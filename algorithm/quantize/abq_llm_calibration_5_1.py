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
from tqdm import tqdm



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

def abqllm(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
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

    
    # 添加新参数
    look_ahead_layers = args.look_ahead_layers if hasattr(args, 'look_ahead_layers') else 0
    analyze_per_layer_mse = args.analyze_per_layer_mse if hasattr(args, 'analyze_per_layer_mse') else False
    
    # 如果需要分析每层的MSE，创建存储结果的字典
    layer_mse_results = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # 将量化层添加到列表中，用于前瞻性量化
        qlayers_list = []
        qlayers_list.append(qlayer)

        # 获取全精度模型的输出
        set_quant_state(qlayer, weight_quant=False, act_quant=False)

        if analyze_per_layer_mse:
            fp_inps_back = copy.deepcopy(fp_inps)

        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        # 此时设置了False, False，所以不量化weight和act
                        fp_inps[j], attn_fp[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids, output_attentions= True)
                        fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        # 在smooth_and_quant_temporary和smooth_and_quant_inplace里会量化weight
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        
        ## 不考虑shift
        # use_shift = True
        use_shift = False

        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.k_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            ## 不考虑shift，初始化为0
                            shift = torch.zeros_like(scale)

                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                    
                    # 暂不考虑compensation vector
                    if args.compensation_calibration and ('down_proj' in name or 'fc2' in name) and (i<=2 or i >= len(layers)-4): 
                        logger.info('use compensation vector')
                        name_tmp = name.replace(".","_")
                        compensation_left = torch.zeros(module.weight.shape[0], 1).to(dev)
                        compensation_right = torch.ones(1, module.weight.shape[1]).to(dev)
                        qlayer.register_parameter(f"{name_tmp}_compensation_left",torch.nn.Parameter(compensation_left))
                        qlayer.register_parameter(f"{name_tmp}_compensation_right",torch.nn.Parameter(compensation_right))
                              
        if args.resume:
            qlayer.load_state_dict(abq_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer

            ## 添加了params_list，可以分别优化
            params_list = [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}]
            if args.compensation_calibration:
                params_list.append({"params":com_parameters(qlayer, use_shift),"lr":1e-2})
            if args.lwc:
                params_list.append({"params":lwc_parameters(qlayer),"lr":args.lwc_lr})
            optimizer = torch.optim.AdamW(
                params_list,weight_decay=args.wd)
            
            # [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":com_parameters(qlayer, use_shift),"lr":1e-2}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}]
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 如果使用前瞻性量化，准备后续n层
            look_ahead_outputs = None
            if look_ahead_layers > 0 and i + look_ahead_layers < len(layers):
                # 加载后续n层到设备
                look_ahead_fp_layers = []
                for k in range(1, look_ahead_layers + 1):
                    if i + k < len(layers):
                        look_ahead_layer = layers[i + k].to(dev)
                        look_ahead_fp_layers.append(look_ahead_layer)
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # 获取量化模型的输出
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 当前层的量化输出
                        quant_out, attn_out = qlayer(quant_inps[index:index+args.batch_size,], 
                                                        attention_mask=attention_mask_batch,
                                                        position_ids=position_ids, 
                                                        output_attentions=True)
                        
                        # 基本损失计算
                        if args.compensation_calibration:
                            # 注意力图损失
                            teacher_output_log_prob = F.log_softmax(attn_fp[index:index+args.batch_size,],dim=-1)
                            student_output_soft = F.softmax(attn_out,dim=-1)
                            loss = torch.abs(kl_loss(teacher_output_log_prob, student_output_soft))
                            
                            student_output_log_prob = F.log_softmax(attn_out,dim=-1)
                            teacher_output_soft = F.softmax(attn_fp[index:index+args.batch_size,],dim=-1)
                            loss += torch.abs(kl_loss(student_output_log_prob, teacher_output_soft))
                        else:
                            loss = 0
                        
                        # 如果使用前瞻性量化，计算后续n层的损失
                        if look_ahead_layers > 0 and i + look_ahead_layers < len(layers):
                            # 计算全精度模型在后续n层的输出
                            fp_look_ahead = fp_inps[index:index+args.batch_size,].clone()
                            quant_look_ahead = quant_out.clone()
                            
                            # 通过后续n层传播
                            for k, look_ahead_layer in enumerate(look_ahead_fp_layers):
                                # 确保数据类型匹配
                                if fp_look_ahead.dtype != look_ahead_layer.self_attn.q_proj.weight.dtype:
                                    fp_look_ahead = fp_look_ahead.to(look_ahead_layer.self_attn.q_proj.weight.dtype)
                                
                                fp_look_ahead = look_ahead_layer(fp_look_ahead, 
                                                               attention_mask=attention_mask_batch,
                                                               position_ids=position_ids)[0]
                                
                                # 对于量化路径，使用已经量化的层（如果有）或全精度层
                                if i + k + 1 < len(qlayers_list):
                                    ## 不会进入此分支
                                    ## print('i + k + 1 < len(qlayers_list)')
                                    next_qlayer = qlayers_list[i + k + 1]
                                    set_quant_state(next_qlayer, weight_quant=False, act_quant=False)
                                    
                                    # 确保数据类型匹配
                                    if quant_look_ahead.dtype != next_qlayer.self_attn.q_proj.weight.dtype:
                                        quant_look_ahead = quant_look_ahead.to(next_qlayer.self_attn.q_proj.weight.dtype)
                                        
                                    quant_look_ahead = next_qlayer(quant_look_ahead, 
                                                                 attention_mask=attention_mask_batch,
                                                                 position_ids=position_ids)[0]
                                else:
                                    # 确保数据类型匹配
                                    if quant_look_ahead.dtype != look_ahead_layer.self_attn.q_proj.weight.dtype:
                                        quant_look_ahead = quant_look_ahead.to(look_ahead_layer.self_attn.q_proj.weight.dtype)
                                        
                                    quant_look_ahead = look_ahead_layer(quant_look_ahead, 
                                                                      attention_mask=attention_mask_batch,
                                                                      position_ids=position_ids)[0]
                            
                            # 添加前瞻性损失
                            # 将张量转换为float32再计算损失
                            fp_look_ahead_fp32 = fp_look_ahead.float()
                            quant_look_ahead_fp32 = quant_look_ahead.float()
                            look_ahead_loss = loss_func(fp_look_ahead_fp32, quant_look_ahead_fp32)
                            # 移除调试打印
                            # print('fp_look_ahead: ', fp_look_ahead)
                            # print('quant_look_ahead: ', quant_look_ahead)
                            # print('look_ahead_loss: ', look_ahead_loss)
                            loss += look_ahead_loss
                            
                            # 余弦相似度损失也应该使用float32计算
                            loss_cos_fun = nn.CosineSimilarity(dim=2)
                            cos_look_ahead = loss_cos_fun(quant_look_ahead_fp32, fp_look_ahead_fp32).mean().abs()
                            loss -= torch.log(cos_look_ahead)
                        else:
                            # 传统的逐层损失
                            loss += loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                            
                            loss_cos_fun = nn.CosineSimilarity(dim=2)
                            cos1 = loss_cos_fun(quant_out, fp_inps[index:index+args.batch_size,]).mean().abs()
                            loss -= torch.log(cos1)
                            cos2 = loss_cos_fun(quant_out, fp_inps_2[index:index+args.batch_size,]).mean().abs()
                            loss -= torch.log(cos2)
                        
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters= get_abq_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            
            # 清理内存
            clear_temp_variable(qlayer)
            del optimizer
            
            # 如果使用前瞻性量化，释放后续层
            if look_ahead_layers > 0 and i + look_ahead_layers < len(layers):
                for look_ahead_layer in look_ahead_fp_layers:
                    look_ahead_layer = look_ahead_layer.to("cpu")
                del look_ahead_fp_layers
                torch.cuda.empty_cache()
        
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:

            # 如果需要分析每层的端到端MSE，计算当前层量化后的端到端MSE
            if analyze_per_layer_mse:
                logger.info(f"=== Analyzing end-to-end MSE for layer {i} ===")
                
                # 初始化累积的MSE和余弦相似度
                total_mse = 0.0
                total_cos_sim = 0.0
                total_batches = 0
                
                with torch.no_grad():
                    for j in tqdm(range(args.nsamples//args.batch_size)):
                        index = j * args.batch_size
                        current_batch_size = args.batch_size
                        
                        # 获取当前批次的输入
                        current_batch = fp_inps_back[index:index+current_batch_size].to(dev)
                        
                        with traincast():
                            # 计算量化路径的输出
                            quant_end_to_end = current_batch.clone()
                            
                            # 第i层使用量化层
                            quant_end_to_end = qlayer(quant_end_to_end, 
                                                    attention_mask=attention_mask_batch,
                                                    position_ids=position_ids)[0]
                            
                            # i+1到最后一层使用原始层
                            for k in range(i+1, len(layers)):
                                quant_end_to_end = layers[k].float().to(dev)(quant_end_to_end, 
                                                           attention_mask=attention_mask_batch,
                                                           position_ids=position_ids)[0]
                                # layers[k].to('cpu')  # 立即将层移回CPU
                                torch.cuda.empty_cache()

                            # 计算全精度路径的输出
                            fp_end_to_end = current_batch.clone()
                            for k in range(i, len(layers)):
                                fp_end_to_end = layers[k].float().to(dev)(fp_end_to_end, 
                                                        attention_mask=attention_mask_batch,
                                                        position_ids=position_ids)[0]
                                # layers[k].to('cpu')
                                torch.cuda.empty_cache()

                            # 计算当前批次的MSE和余弦相似度
                            batch_mse = loss_func(fp_end_to_end, quant_end_to_end).item()
                            batch_cos_sim = F.cosine_similarity(
                                fp_end_to_end.float().view(current_batch_size, -1),
                                quant_end_to_end.float().view(current_batch_size, -1),
                                dim=1
                            ).mean().item()

                            # 更新累积值
                            total_mse += batch_mse * current_batch_size
                            total_cos_sim += batch_cos_sim * current_batch_size
                            total_batches += current_batch_size

                            # 清理当前批次的内存
                            del quant_end_to_end
                            del fp_end_to_end
                            torch.cuda.empty_cache()

                # 计算平均MSE和余弦相似度
                avg_mse = total_mse / total_batches
                avg_cos_sim = total_cos_sim / total_batches
                
                # 记录结果
                layer_mse_results[i] = {
                    'layer_idx': i,
                    'look_ahead_layers': look_ahead_layers,
                    'end_to_end_mse': avg_mse,
                    'cosine_similarity': avg_cos_sim,
                    'calibration_method': 'look_ahead' if look_ahead_layers > 0 else 'traditional'
                }
                
                logger.info(f"Layer {i} Results:")
                logger.info(f"  Look-ahead layers: {look_ahead_layers}")
                logger.info(f"  End-to-End MSE: {avg_mse:.6f}")
                logger.info(f"  Cosine Similarity: {avg_cos_sim:.6f}")
                
                # 保存结果
                result_filename = f"layer_mse_results_look_ahead_{look_ahead_layers}.pth"
                torch.save(layer_mse_results, os.path.join(args.output_dir, result_filename))
                
                # 保存为可读文本
                txt_filename = f"layer_mse_results_look_ahead_{look_ahead_layers}.txt"
                with open(os.path.join(args.output_dir, txt_filename), 'a') as f:
                    f.write(f"\nLayer {i} Results:\n")
                    f.write(f"Look-ahead layers: {look_ahead_layers}\n")
                    f.write(f"End-to-End MSE: {avg_mse:.6f}\n")
                    f.write(f"Cosine Similarity: {avg_cos_sim:.6f}\n")
                    f.write("-" * 50 + "\n")
                
                torch.cuda.empty_cache()

            # update input of quantization model
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            qlayer.half()
            layer = layer.to("cpu")
            layers[i] = qlayer.to("cpu")
            abq_parameters[i] = abq_state_dict(qlayer)
            torch.save(abq_parameters, os.path.join(args.output_dir, f"abq_parameters.pth"))
            
        else:
            qlayer.half()
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

