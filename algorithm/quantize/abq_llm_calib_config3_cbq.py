# Copyright 2024 ByteDance and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# abq_llm_calib_config3_cbq.py是我让你根据abq_llm_calib_config2.py进行的修改。
# 引入了大小为2的重叠滑动窗口，一次对窗口内的所有DecoderLayer层进行量化校准，如第一轮对[0,1]层校准、第二轮对[1,2]层校准、第三轮对[2,3]层校准、最后一轮只对最后一层校准。
# 但是abq_llm_calib_config3_cbq.py的校准后的量化模型精度很低，帮我检查代码错误，并进行修改

import torch
import torch.nn as nn
from models.int_llama_layer_nomatquant import QuantLlamaDecoderLayer, QuantLlamaAttention
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
from torch.nn import functional as F
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from quantize.utils_calibration import let_parameters, lwc_parameters, get_abq_parameters,com_parameters, \
                            abq_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
import pickle

quant_scheme = ['w4a4', 'w4a8', 'w8a8']

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def get_quant_scheme_for_module(quant_map, layer_idx, module_name):
    if quant_map is None:
        return None
    layer_config = quant_map.get(str(layer_idx), {})
    return layer_config.get(module_name, None)

def get_quant_params_by_scheme(base_params, scheme_idx):
    if scheme_idx is None:
        return base_params
    
    scheme = quant_scheme[scheme_idx]
    new_params = copy.deepcopy(base_params)
    if scheme == 'w4a4':
        new_params['n_bits'] = 4
    elif scheme == 'w4a8':
        new_params['n_bits'] = 8 if 'act_quant_params' in str(base_params) else 4
    elif scheme == 'w8a8':
        new_params['n_bits'] = 8
    return new_params

def abqllm(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # 加载量化配置映射
    quant_map = None
    if args.quant_map:
        with open(args.quant_map, 'rb') as f:
            quant_map = pickle.load(f)
            logger.info(f"Loaded quant_map from {args.quant_map}")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"fc2"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1",
            "fc2":"fc2"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps                                                               # 量化模型输入输出
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input              # 输入输出使用fp模型
    fp_inps_2 = copy.deepcopy(inps) # take output of quantization model as input    # 输入上一层的量化输出，fp模型的输出
    attn_fp = torch.zeros([args.nsamples, 2048, 2048]).to(fp_inps.device)
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='mean')
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        abq_parameters = torch.load(args.resume)
    else:
        abq_parameters = {}

    # 使用滑动窗口进行校准
    window_size = args.window_size  # 滑动窗口大小
    
    # 预处理所有层，创建量化层
    qlayers = []
    for i in range(len(layers)):
        layer = layers[i]
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    scheme_idx = get_quant_scheme_for_module(quant_map, i, name)
                    weight_params = get_quant_params_by_scheme(args.weight_quant_params, scheme_idx)
                    act_params = get_quant_params_by_scheme(args.act_quant_params, scheme_idx)
                    quantlinear = QuantLinear(module, weight_params, act_params)
                    add_new_module(name, qlayer, quantlinear)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
            if quant_map is not None and str(i) in quant_map:
                logger.info(f'quant_map[{i}]: {quant_map[str(i)]}')
                for name, module in qlayer.named_modules():
                    if isinstance(module, QuantLinear) and name in quant_map[str(i)]:
                        scheme = quant_scheme[quant_map[str(i)][name]]
                        wbit, abits = int(scheme[1]), int(scheme[3])
                        logger.info(f"layer {i} module {name} scheme {scheme} wbit {wbit} abits {abits}")
                        module.change_n_bits(wbit, abits)
        
        # 初始化LET参数
        qlayer.let = args.let
        use_shift = False
        if is_llama or args.abits == 16:
            use_shift = False
            
        if args.let:
            # 初始化channel-wise scaling和shift
            qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(torch.ones(layer.self_attn.k_proj.out_features, device="cpu", dtype=dtype)))
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device="cpu", dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            shift = torch.zeros_like(scale)

                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))
                    
                    if args.compensation_calibration and ('down_proj' in name or 'fc2' in name) and (i<=2 or i >= len(layers)-4): 
                        name_tmp = name.replace(".", "_")
                        compensation_left = torch.zeros(module.weight.shape[0], 1).to("cpu")
                        compensation_right = torch.ones(1, module.weight.shape[1]).to("cpu")
                        qlayer.register_parameter(f"{name_tmp}_compensation_left", torch.nn.Parameter(compensation_left))
                        qlayer.register_parameter(f"{name_tmp}_compensation_right", torch.nn.Parameter(compensation_right))
        
        if args.resume and i in abq_parameters:
            qlayer.load_state_dict(abq_parameters[i], strict=False)
            
        qlayers.append(qlayer)
    
    # 存储每层的输出，用于后续窗口的输入
    layer_outputs = {}
    layer_outputs[-1] = quant_inps  # 初始输入
    
    # 使用滑动窗口进行校准
    for window_start in range(0, len(layers) - window_size + 1):
        window_end = window_start + window_size
        logger.info(f"=== Start quantize layers {window_start} to {window_end-1} ===")
        
        # 将窗口内的层移到GPU
        active_qlayers = []
        for i in range(window_start, window_end):
            qlayers[i] = qlayers[i].to(dev)
            active_qlayers.append(qlayers[i])
        
        # 计算FP模型在这个窗口的输出
        if args.epochs > 0:
            with torch.no_grad():
                # 设置为不量化模式
                for qlayer in active_qlayers:
                    set_quant_state(qlayer, weight_quant=False, act_quant=False)
                
                # 计算FP模型输出
                with torch.cuda.amp.autocast():
                    # 获取窗口输入
                    window_input = layer_outputs[window_start-1].to(dev)
                    
                    # 存储窗口内每层的FP输出
                    fp_layer_outputs = {}
                    fp_layer_outputs[window_start-1] = window_input
                    
                    # 计算窗口内每层的FP输出 - 内存优化：分批处理
                    max_batch_size = min(args.batch_size, 4)  # 降低单次处理的批次大小
                    
                    for layer_idx, qlayer in enumerate(active_qlayers, window_start):
                        layer_input = fp_layer_outputs[layer_idx-1]
                        layer_attn_outputs = []
                        layer_outputs_batch = []
                        
                        # 分批处理样本
                        for j in range(0, args.nsamples, max_batch_size):
                            batch_end = min(j + max_batch_size, args.nsamples)
                            batch_input = layer_input[j:batch_end]
                            
                            # 计算FP输出和注意力图
                            fp_out, attn_fp_out = qlayer(batch_input, 
                                                        attention_mask=attention_mask, 
                                                        position_ids=position_ids, 
                                                        output_attentions=True)
                            
                            layer_outputs_batch.append(fp_out.cpu())  # 立即移到CPU
                            if layer_idx == window_end - 1:  # 只保存最后一层的注意力图
                                layer_attn_outputs.append(attn_fp_out[0].cpu())
                            
                            # 清理中间变量
                            del fp_out, attn_fp_out
                            torch.cuda.empty_cache()
                        
                        # 合并批次结果
                        fp_layer_outputs[layer_idx] = torch.cat([batch.to(dev) for batch in layer_outputs_batch], dim=0)
                        del layer_outputs_batch  # 释放内存
                        
                        # 存储注意力图
                        if layer_idx == window_end - 1:  # 只保存窗口最后一层的注意力图
                            window_attn_fp = torch.cat([attn.to(dev) for attn in layer_attn_outputs], dim=0)
                            del layer_attn_outputs  # 释放内存
                        
                        # 如果不是最后一层，则移除前一层的输出以节省内存
                        if layer_idx > window_start:
                            del fp_layer_outputs[layer_idx-2]  # 不再需要前一层的前一层输出
                            torch.cuda.empty_cache()
                    
                    # 计算使用量化输入的FP输出
                    quant_input = layer_outputs[window_start-1].to(dev)
                    fp_inps_2_outputs = []
                    
                    # 分批处理样本
                    for j in range(0, args.nsamples, max_batch_size):
                        batch_end = min(j + max_batch_size, args.nsamples)
                        batch_input = quant_input[j:batch_end]
                        
                        # 通过窗口内所有层
                        for qlayer in active_qlayers:
                            batch_input = qlayer(batch_input, 
                                                attention_mask=attention_mask, 
                                                position_ids=position_ids)[0]
                        
                        fp_inps_2_outputs.append(batch_input.cpu())
                        
                        # 清理中间变量
                        del batch_input
                        torch.cuda.empty_cache()
                    
                    # 合并批次结果
                    window_fp_inps_2 = torch.cat([batch.to(dev) for batch in fp_inps_2_outputs], dim=0)
                    del fp_inps_2_outputs  # 释放内存
                    
                    # 保存窗口输出，用于后续训练
                    window_fp_inps = fp_layer_outputs[window_end-1]
                    # 不再需要的中间层输出
                    del fp_layer_outputs
                    torch.cuda.empty_cache()
        
        # 设置为量化模式
        for qlayer in active_qlayers:
            set_quant_state(qlayer, weight_quant=False, act_quant=True)
            qlayer.float()  # 为AMP训练做准备
        
        if args.epochs > 0:
            # 创建优化器，同时优化窗口内所有层的参数
            all_params_list = []
            for qlayer in active_qlayers:
                params = {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr}
                all_params_list.append(params)
                
                if args.compensation_calibration:
                    params = {"params": com_parameters(qlayer, use_shift), "lr": 1e-2}
                    all_params_list.append(params)
                
                if args.lwc:
                    params = {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}
                    all_params_list.append(params)
            
            optimizer = torch.optim.AdamW(all_params_list, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 训练循环
            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                
                # 内存优化：减少每批次处理的样本数
                max_batch_size = min(args.batch_size, 4)  # 可根据实际情况调整
                
                for j in range(0, args.nsamples, max_batch_size):
                    batch_end = min(j + max_batch_size, args.nsamples)
                    batch_size = batch_end - j
                    
                    # 获取量化模型的输出
                    with traincast():
                        # 量化所有层的权重
                        for qlayer in active_qlayers:
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 获取窗口输入
                        batch_quant_inps = layer_outputs[window_start-1][j:batch_end].to(dev)
                        
                        # 通过窗口内所有层
                        for layer_idx, qlayer in enumerate(active_qlayers, window_start):
                            if layer_idx == window_end - 1:  # 最后一层
                                quant_out, attn_out = qlayer(batch_quant_inps, 
                                                            attention_mask=attention_mask_batch[:batch_size] if attention_mask_batch is not None else None, 
                                                            position_ids=position_ids, 
                                                            output_attentions=True)
                            else:
                                batch_quant_inps = qlayer(batch_quant_inps, 
                                                        attention_mask=attention_mask_batch[:batch_size] if attention_mask_batch is not None else None, 
                                                        position_ids=position_ids)[0]
                        
                        # 计算损失
                        if args.compensation_calibration:
                            # 注意力图损失
                            teacher_output_log_prob = F.log_softmax(window_attn_fp[j:batch_end], dim=-1)
                            student_output_soft = F.softmax(attn_out, dim=-1)
                            loss = torch.abs(kl_loss(teacher_output_log_prob, student_output_soft))
                            
                            student_output_log_prob = F.log_softmax(attn_out, dim=-1)
                            teacher_output_soft = F.softmax(window_attn_fp[j:batch_end], dim=-1)
                            loss += torch.abs(kl_loss(student_output_log_prob, teacher_output_soft))
                            
                            # 输出层引导损失
                            loss += loss_func(window_fp_inps[j:batch_end], quant_out)
                            loss += loss_func(window_fp_inps_2[j:batch_end], quant_out)
                            
                            # 余弦相似度损失
                            loss_attn_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_attn_fun(quant_out, window_fp_inps[j:batch_end]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_attn_fun(quant_out, window_fp_inps_2[j:batch_end]).mean().abs()
                            loss -= torch.log(cos2)
                        else:
                            # 输出层引导损失
                            loss = loss_func(window_fp_inps[j:batch_end], quant_out)
                            loss += loss_func(window_fp_inps_2[j:batch_end], quant_out)
                            
                            # 余弦相似度损失
                            loss_attn_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_attn_fun(quant_out, window_fp_inps[j:batch_end]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_attn_fun(quant_out, window_fp_inps_2[j:batch_end]).mean().abs()
                            loss -= torch.log(cos2)
                    
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    
                    # 获取所有层的参数
                    all_params = []
                    for qlayer in active_qlayers:
                        all_params.extend(get_abq_parameters(qlayer, use_shift))
                    
                    norm = loss_scaler(loss, optimizer, parameters=all_params).cpu()
                    norm_list.append(norm.data)
                    
                    # 明确释放不再需要的张量
                    del batch_quant_inps, quant_out, attn_out, loss
                    torch.cuda.empty_cache()
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layers {window_start}-{window_end-1} epoch {epoch} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                
                # 在每个epoch结束后明确释放内存
                del loss_list, norm_list
                torch.cuda.empty_cache()
            
            # 清理
            del optimizer, loss_scaler
            for qlayer in active_qlayers:
                clear_temp_variable(qlayer)
            
            # 释放不再需要的变量
            del window_fp_inps, window_fp_inps_2, window_attn_fp, all_params_list, all_params
            torch.cuda.empty_cache()
        
        # 实际平滑和量化
        for i, qlayer in enumerate(active_qlayers, window_start):
            smooth_and_quant_inplace(qlayer, args, is_llama)
        
        # 更新量化模型的输出
        with torch.no_grad():
            with traincast():
                # 计算窗口输出，用于下一个窗口的输入
                
                # 内存优化：减少每批次处理的样本数
                max_batch_size = min(args.batch_size, 4)  # 可根据实际情况调整
                
                # 初始化层输出列表
                for layer_idx in range(window_start, window_end):
                    layer_outputs[layer_idx] = []
                
                # 分批处理样本
                for j in range(0, args.nsamples, max_batch_size):
                    batch_end = min(j + max_batch_size, args.nsamples)
                    batch_input = layer_outputs[window_start-1][j:batch_end].to(dev)
                    
                    # 通过窗口内所有层
                    for layer_idx, qlayer in enumerate(active_qlayers, window_start):
                        batch_input = qlayer(batch_input, 
                                            attention_mask=attention_mask, 
                                            position_ids=position_ids)[0]
                        
                        # 保存每层的输出到CPU
                        layer_outputs[layer_idx].append(batch_input.cpu())
                    
                    # 释放GPU内存
                    del batch_input
                    torch.cuda.empty_cache()
                
                # 合并每层的输出
                for layer_idx in range(window_start, window_end):
                    layer_outputs[layer_idx] = torch.cat(layer_outputs[layer_idx], dim=0)
        
        # 保存参数
        for i, qlayer in enumerate(active_qlayers, window_start):
            register_scales_and_zeros(qlayer)
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            abq_parameters[i] = abq_state_dict(qlayer)

        logger.info(f"Saving abq_parameters for block {window_start}-{window_end-1}")
        torch.save(abq_parameters, os.path.join(args.output_dir, f"abq_parameters.pth"))
        
        # 清理GPU内存
        for i in range(window_start, window_end):
            qlayers[i] = qlayers[i].to("cpu")
        
        # 移除不再需要的输入，只保留当前窗口结束层的输出
        if window_start > 0 and window_start-2 in layer_outputs:
            del layer_outputs[window_start-2]  # 保留window_start-1作为下一个窗口的输入
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # 最终清理
    del inps
    if -1 in layer_outputs:
        del layer_outputs[-1]
    del layer_outputs
    if 'quant_inps' in locals():
        del quant_inps
    if 'fp_inps' in locals():
        del fp_inps
    if 'fp_inps_2' in locals():
        del fp_inps_2
    torch.cuda.empty_cache()
    
    # 添加最终内存清理
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    model.config.use_cache = use_cache
    return model

