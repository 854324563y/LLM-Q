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

    # 加载块配置
    blocks = None
    if args.blocks_pkl:
        with open(args.blocks_pkl, 'rb') as f:
            blocks = pickle.load(f)
            logger.info(f"Loaded blocks from {args.blocks_pkl}: {blocks}")
            # 将每个块的范围转换为实际的层索引列表
            processed_blocks = []
            for start, end in blocks:
                processed_blocks.append(list(range(start, end)))
            blocks = processed_blocks
            logger.info(f"Processed blocks: {blocks}")

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
    
    # # save inps
    # with open('/workspace/volume/yangzhe/ABQ-LLM/algorithm/cache/inps.pt', 'wb') as f:
    #     torch.save(inps, f)

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

    
    # 如果没有提供块配置，则按照原来的方式逐层校准
    if blocks is None:
        blocks = [[i] for i in range(len(layers))]
        logger.info("No blocks configuration provided, using layer-by-layer calibration")
    
    # 按块进行校准
    for block_idx, block in enumerate(blocks):
        logger.info(f"=== Start quantize block {block_idx} with layers {block} ===")
        
        # 将块中的所有层移到设备上
        qlayers = []
        for i in block:
            layer = layers[i].to(dev)
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
                print('quant_map[i]: ', quant_map[str(i)])
                for name, module in qlayer.named_modules():
                    if isinstance(module, QuantLinear) and name in quant_map[str(i)]:
                        scheme = quant_scheme[quant_map[str(i)][name]]
                        wbit, abits = int(scheme[1]), int(scheme[3])
                        print(f"layer {i} module {name} scheme {scheme} wbit {wbit} abits {abits}")
                        module.change_n_bits(wbit, abits)
            qlayer = qlayer.to(dev)
            
            # 初始化平滑参数
            set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
            qlayer.let = args.let
            
            use_shift = False
            if is_llama or args.abits == 16:
                use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
            
            if args.let:
                # 初始化通道级缩放和偏移
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.k_proj.out_features,device=dev, dtype=dtype)))
                for name,module in qlayer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                                shift = torch.zeros_like(scale)

                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                        
                        # 补偿向量
                        if args.compensation_calibration and ('down_proj' in name or 'fc2' in name) and (i<=2 or i >= len(layers)-4): 
                            logger.info('use compensation vector')
                            name_tmp = name.replace(".","_")
                            compensation_left = torch.zeros(module.weight.shape[0], 1).to(dev)
                            compensation_right = torch.ones(1, module.weight.shape[1]).to(dev)
                            qlayer.register_parameter(f"{name_tmp}_compensation_left",torch.nn.Parameter(compensation_left))
                            qlayer.register_parameter(f"{name_tmp}_compensation_right",torch.nn.Parameter(compensation_right))
            
            if args.resume and i in abq_parameters:
                qlayer.load_state_dict(abq_parameters[i], strict=False)
            
            qlayers.append(qlayer)
        
        # 获取全精度模型的输出
        # 使用块中最后一层的索引
        last_layer_idx = block[-1]
        
        # 初始化输入输出
        if block_idx == 0:
            # 第一个块使用原始输入
            quant_inps_block = copy.deepcopy(inps)
            fp_inps_block = copy.deepcopy(inps)
            fp_inps_2_block = copy.deepcopy(inps)
        else:
            # 后续块使用前一个块的输出作为输入
            quant_inps_block = copy.deepcopy(quant_inps)
            fp_inps_block = copy.deepcopy(fp_inps)
            fp_inps_2_block = copy.deepcopy(fp_inps_2)
        
        attn_fp = torch.zeros([args.nsamples, 2048, 2048]).to(fp_inps_block.device)
        
        # 获取全精度模型的输出
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        # 通过块中所有层的全精度前向传播
                        temp_fp = fp_inps_block[j].unsqueeze(0)
                        temp_fp2 = quant_inps_block[j].unsqueeze(0)
                        
                        for layer_idx, qlayer in zip(block, qlayers):
                            set_quant_state(qlayer, weight_quant=False, act_quant=False)
                            if layer_idx == last_layer_idx:
                                # 最后一层获取注意力输出
                                temp_fp, attn_temp = qlayer(temp_fp, attention_mask=attention_mask, position_ids=position_ids, output_attentions=True)
                                attn_fp[j] = attn_temp
                                temp_fp2 = qlayer(temp_fp2, attention_mask=attention_mask, position_ids=position_ids)[0]
                            else:
                                temp_fp = qlayer(temp_fp, attention_mask=attention_mask, position_ids=position_ids)[0]
                                temp_fp2 = qlayer(temp_fp2, attention_mask=attention_mask, position_ids=position_ids)[0]
                        
                        fp_inps_block[j] = temp_fp
                        fp_inps_2_block[j] = temp_fp2
        
        # 如果需要训练
        if args.epochs > 0:
            # 准备优化器参数
            all_params_let = []
            all_params_com = []
            all_params_lwc = []
            
            for qlayer in qlayers:
                with torch.no_grad():
                    qlayer.float()  # required for AMP training
                
                # 收集所有层的参数
                all_params_let.extend(let_parameters(qlayer, use_shift))
                if args.compensation_calibration:
                    all_params_com.extend(com_parameters(qlayer, use_shift))
                if args.lwc:
                    all_params_lwc.extend(lwc_parameters(qlayer))
            
            # 创建优化器
            params_list = [{"params": all_params_let, "lr": args.let_lr}]
            if args.compensation_calibration:
                params_list.append({"params": all_params_com, "lr": 1e-2})
            if args.lwc:
                params_list.append({"params": all_params_lwc, "lr": args.lwc_lr})
            
            optimizer = torch.optim.AdamW(params_list, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 训练循环
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):
                    index = j * args.batch_size
                    
                    # 前向传播
                    with traincast():
                        # 对块中所有层应用临时量化
                        for qlayer in qlayers:
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 通过块中所有层的前向传播
                        temp_quant = quant_inps_block[index:index+args.batch_size]
                        all_attn_outputs = []  # 收集所有层的注意力输出
                        
                        for layer_idx, qlayer in zip(block, qlayers):
                            # 对每一层都获取注意力输出
                            temp_quant, attn_out = qlayer(temp_quant, attention_mask=attention_mask_batch, 
                                                          position_ids=position_ids, output_attentions=True)
                            all_attn_outputs.append(attn_out)
                        
                        quant_out = temp_quant  # 最终输出
                        
                        # 计算损失
                        loss = 0.0
                        
                        # 添加每一层的注意力损失
                        for layer_idx, (attn_out, layer_i) in enumerate(zip(all_attn_outputs, block)):
                            # 获取对应层的参考注意力
                            layer_attn_fp = attn_fp[index:index+args.batch_size]
                            
                            if args.compensation_calibration:
                                # 注意力图损失
                                teacher_output_log_prob = F.log_softmax(layer_attn_fp, dim=-1)
                                student_output_soft = F.softmax(attn_out, dim=-1)
                                layer_loss = torch.abs(kl_loss(teacher_output_log_prob, student_output_soft))
                                
                                student_output_log_prob = F.log_softmax(attn_out, dim=-1)
                                teacher_output_soft = F.softmax(layer_attn_fp, dim=-1)
                                layer_loss += torch.abs(kl_loss(student_output_log_prob, teacher_output_soft))
                                
                                loss += layer_loss
                        
                        # 添加最终输出的损失
                        if args.compensation_calibration:
                            # down_proj引导损失
                            loss += loss_func(fp_inps_block[index:index+args.batch_size], quant_out)
                            loss += loss_func(fp_inps_2_block[index:index+args.batch_size], quant_out)
                            
                            loss_attn_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_attn_fun(quant_out, fp_inps_block[index:index+args.batch_size]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_attn_fun(quant_out, fp_inps_2_block[index:index+args.batch_size]).mean().abs()
                            loss -= torch.log(cos2)
                    
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    
                    # 获取所有块中层的参数
                    all_params = []
                    for qlayer in qlayers:
                        all_params.extend(get_abq_parameters(qlayer, use_shift))
                    
                    norm = loss_scaler(loss, optimizer, parameters=all_params).cpu()
                    norm_list.append(norm.data)
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"block {block_idx} (layers {block}) iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            
            # 清理
            for qlayer in qlayers:
                clear_temp_variable(qlayer)
            del optimizer
        
        # 实际平滑和量化
        for i, qlayer in zip(block, qlayers):
            smooth_and_quant_inplace(qlayer, args, is_llama)
            
            if args.epochs > 0:
                # 保存参数
                register_scales_and_zeros(qlayer)
                qlayer.half()
                layers[i] = qlayer.to("cpu")
                abq_parameters[i] = abq_state_dict(qlayer)
            else:
                qlayer.half()
                register_scales_and_zeros(qlayer)
                layers[i] = qlayer.to("cpu")
            
            # 释放原始层
            layers[i].to("cpu")
        
        # 更新下一个块的输入
        if args.epochs > 0:
            with torch.no_grad():
                for j in range(args.nsamples):
                    # 确保输入和模型使用相同的数据类型
                    temp_quant = quant_inps[j].unsqueeze(0)
                    for i in block:
                        qlayer = layers[i].to(dev)
                        # 确保输入和模型使用相同的数据类型
                        if temp_quant.dtype != qlayer.self_attn.q_proj.weight.dtype:
                            temp_quant = temp_quant.to(qlayer.self_attn.q_proj.weight.dtype)
                        temp_quant = qlayer(temp_quant, attention_mask=attention_mask, position_ids=position_ids)[0]
                        qlayer.to("cpu")
                    quant_inps[j] = temp_quant
                    
                    # 更新fp_inps和fp_inps_2
                    fp_inps[j] = fp_inps_block[j]
                    fp_inps_2[j] = fp_inps_2_block[j]

        for layer_idx, qlayer in zip(block, qlayers):
            set_quant_state(qlayer, weight_quant=False, act_quant=True)
        
        # 保存当前块的参数
        logger.info(f"Saving abq_parameters for block {block_idx}, block: {block}")
        torch.save(abq_parameters, os.path.join(args.output_dir, f"abq_parameters.pth"))
        
        # 清理内存
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

