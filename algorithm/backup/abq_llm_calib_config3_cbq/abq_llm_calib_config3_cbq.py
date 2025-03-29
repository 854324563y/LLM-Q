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
    window_size = 2  # 滑动窗口大小
    
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
            if str(i) in quant_map:
                print('quant_map[i]: ', quant_map[str(i)])
                for name, module in qlayer.named_modules():
                    if isinstance(module, QuantLinear) and name in quant_map[str(i)]:
                        scheme = quant_scheme[quant_map[str(i)][name]]
                        wbit, abits = int(scheme[1]), int(scheme[3])
                        print(f"layer {i} module {name} scheme {scheme} wbit {wbit} abits {abits}")
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
                    window_fp_inps = []
                    window_fp_inps_2 = []
                    window_attn_fp = []
                    
                    # 初始输入
                    if window_start == 0:
                        current_fp_inps = fp_inps
                        current_quant_inps = quant_inps
                    else:
                        # 使用前一个窗口的输出作为输入
                        current_fp_inps = fp_inps
                        current_quant_inps = quant_inps
                    
                    # 计算窗口内每层的FP输出
                    for j in range(args.nsamples):
                        layer_fp_inps = current_fp_inps[j].unsqueeze(0)
                        layer_quant_inps = current_quant_inps[j].unsqueeze(0)
                        
                        # 计算窗口内所有层的输出
                        for layer_idx, qlayer in enumerate(active_qlayers):
                            layer_fp_out, layer_attn_fp = qlayer(layer_fp_inps, attention_mask=attention_mask, position_ids=position_ids, output_attentions=True)
                            layer_fp_inps = layer_fp_out
                            
                            # 只保存最后一层的输出
                            if layer_idx == len(active_qlayers) - 1:
                                window_fp_inps.append(layer_fp_out)
                                window_attn_fp.append(layer_attn_fp)
                        
                        # 计算使用量化输入的FP输出
                        layer_fp_inps_2 = layer_quant_inps
                        for qlayer in active_qlayers:
                            layer_fp_inps_2 = qlayer(layer_fp_inps_2, attention_mask=attention_mask, position_ids=position_ids)[0]
                        window_fp_inps_2.append(layer_fp_inps_2)
                    
                    # 转换为张量
                    window_fp_inps = torch.cat(window_fp_inps, dim=0)
                    window_fp_inps_2 = torch.cat(window_fp_inps_2, dim=0)
                    window_attn_fp = torch.stack([attn[0] for attn in window_attn_fp])
        
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
                
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    
                    # 获取量化模型的输出
                    with traincast():
                        # 量化所有层的权重
                        for qlayer in active_qlayers:
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 前向传播
                        batch_quant_inps = quant_inps[index:index+args.batch_size]
                        
                        # 通过窗口内所有层
                        for layer_idx, qlayer in enumerate(active_qlayers):
                            if layer_idx == len(active_qlayers) - 1:  # 最后一层
                                quant_out, attn_out = qlayer(batch_quant_inps, attention_mask=attention_mask_batch, position_ids=position_ids, output_attentions=True)
                            else:
                                batch_quant_inps = qlayer(batch_quant_inps, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        
                        # 计算损失
                        if args.compensation_calibration:
                            # 注意力图损失
                            teacher_output_log_prob = F.log_softmax(window_attn_fp[index:index+args.batch_size], dim=-1)
                            student_output_soft = F.softmax(attn_out, dim=-1)
                            loss = torch.abs(kl_loss(teacher_output_log_prob, student_output_soft))
                            
                            student_output_log_prob = F.log_softmax(attn_out, dim=-1)
                            teacher_output_soft = F.softmax(window_attn_fp[index:index+args.batch_size], dim=-1)
                            loss += torch.abs(kl_loss(student_output_log_prob, teacher_output_soft))
                            
                            # 输出层引导损失
                            loss += loss_func(window_fp_inps[index:index+args.batch_size], quant_out)
                            loss += loss_func(window_fp_inps_2[index:index+args.batch_size], quant_out)
                            
                            # 余弦相似度损失
                            loss_attn_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_attn_fun(quant_out, window_fp_inps[index:index+args.batch_size]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_attn_fun(quant_out, window_fp_inps_2[index:index+args.batch_size]).mean().abs()
                            loss -= torch.log(cos2)
                        else:
                            # 输出层引导损失
                            loss = loss_func(window_fp_inps[index:index+args.batch_size], quant_out)
                            loss += loss_func(window_fp_inps_2[index:index+args.batch_size], quant_out)
                            
                            # 余弦相似度损失
                            loss_attn_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_attn_fun(quant_out, window_fp_inps[index:index+args.batch_size]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_attn_fun(quant_out, window_fp_inps_2[index:index+args.batch_size]).mean().abs()
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
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layers {window_start}-{window_end-1} epoch {epoch} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            
            # 清理
            del optimizer
            for qlayer in active_qlayers:
                clear_temp_variable(qlayer)
        
        # 实际平滑和量化
        for i, qlayer in enumerate(active_qlayers, window_start):
            smooth_and_quant_inplace(qlayer, args, is_llama)
        
        # 更新量化模型的输入
        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    # 更新量化输入
                    for j in range(args.nsamples):
                        current_input = quant_inps[j].unsqueeze(0)
                        for qlayer in active_qlayers:
                            current_input = qlayer(current_input, attention_mask=attention_mask, position_ids=position_ids)[0]
                        quant_inps[j] = current_input.squeeze(0)
                    
                    # 更新FP输入
                    for j in range(args.nsamples):
                        current_input = fp_inps[j].unsqueeze(0)
                        for qlayer in active_qlayers:
                            current_input = qlayer(current_input, attention_mask=attention_mask, position_ids=position_ids)[0]
                        fp_inps[j] = current_input.squeeze(0)
            
            # 保存参数
            for i, qlayer in enumerate(active_qlayers, window_start):
                register_scales_and_zeros(qlayer)
                qlayer.half()
                layers[i] = qlayer.to("cpu")
                abq_parameters[i] = abq_state_dict(qlayer)
            
            torch.save(abq_parameters, os.path.join(args.output_dir, f"abq_parameters.pth"))
        else:
            for i, qlayer in enumerate(active_qlayers, window_start):
                qlayer.half()
                register_scales_and_zeros(qlayer)
                layers[i] = qlayer.to("cpu")
        
        # 清理GPU内存
        for i in range(window_start, window_end):
            qlayers[i] = qlayers[i].to("cpu")
        torch.cuda.empty_cache()
    
    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

