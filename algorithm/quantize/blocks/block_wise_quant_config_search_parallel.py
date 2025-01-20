import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
from models.LMClass import LMClass
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from quantize.blocks.int_linear import QuantLinear
import copy
import sys
import torch.multiprocessing as mp
import cvxpy as cp
import logging
from multiprocessing import get_logger, log_to_stderr
# 设置多进程启动方法为spawn
mp.set_start_method('spawn', force=True)

quant_scheme = ['w4a4', 'w4a8', 'w8a8']

types_to_quant = (torch.nn.Linear)


def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    if isinstance(model, LMClass):
        m = model.model
    else:
        m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

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


def kldiv(quant_out, out):
    # # calculate kl divergence between out and quant_out
    # out = F.log_softmax(out, dim=-1)
    # quant_out = F.log_softmax(quant_out, dim=-1)
    # kl = F.kl_div(out, quant_out.exp(), reduction='batchmean', log_target=False)
    # return kl.item()
    """
    计算 KL 散度，修正使用 torch.nn.functional.kl_div 的问题。
    Args:
        quant_out (torch.Tensor): 量化模型输出 (logits)。
        out (torch.Tensor): 原始模型输出 (logits)。
    Returns:
        float: KL 散度的值。
    """
    # 计算概率分布（使用 softmax 转为概率）
    # log_out = F.log_softmax(out, dim=-1)  # log(P)
    # quant_probs = F.softmax(quant_out, dim=-1)  # Q

    log_out = F.log_softmax(out - out.max(dim=-1, keepdim=True).values, dim=-1)
    quant_probs = F.softmax(quant_out - quant_out.max(dim=-1, keepdim=True).values, dim=-1)

    print(torch.sum(F.softmax(quant_out, dim=-1), dim=-1))  # 每行应该接近 1
    print(torch.sum(F.softmax(out, dim=-1), dim=-1))  # 每行应该接近 1
    # KL 散度计算
    kl = F.kl_div(log_out, quant_probs, reduction='batchmean', log_target=False)

    if kl.item() < 0:
        print(quant_out)
        print(out)
        sys.exit()
    return kl.item()

def mse(quant_out, out):
    quant_out = quant_out.to(torch.float32)
    out = out.to(torch.float32)
    res = F.mse_loss(quant_out, out)
    if torch.isinf(res):
        print(quant_out)
        print(out)
        sys.exit()
    return res.item()

def get_linear_bitops(model, module_name, abits, wbits):
    m = getModuleByName(model, module_name)
    if isinstance(m, torch.nn.Linear):
        n_muls = m.in_features * m.out_features  # 乘法操作数
        n_accs = m.in_features - 1  # 累加操作数
        bitops_per_mul = abits * wbits  # 每个乘法的位操作数
        bitops_per_acc = max(abits, wbits) + 5  # 每个累加的位操作数，+5是为了处理进位
        return n_muls * bitops_per_mul + n_accs * bitops_per_acc
    return 0

def quant_and_forward(args, quant_config, layers, block, inputs):
    """
    Args:
        inputs: dict containing:
            - hidden_states: input tensor
            - attention_mask_batch: attention mask
            - position_ids: position ids
    """
    # print('quant_config: ', quant_config)

    quant_out = copy.deepcopy(inputs['hidden_states'])

    for module_name, scheme in quant_config.items():
        w_bits = int(scheme[1])
        a_bits = int(scheme[3])
        module = getModuleByName(layers, module_name)
        assert isinstance(module, QuantLinear), 'module is not a QuantLinear'
        module.change_n_bits(w_bits, a_bits)
        module.set_quant_state(weight_quant=True, act_quant=True)

    # # replace certain nn.Linear modules with QuantLinear modules
    # ori_modules = {}
    # new_modules = []
    # for module_name, scheme in quant_config.items():
    #     w_bits = int(scheme[1])
    #     a_bits = int(scheme[3])
    #     ori_module = getModuleByName(layers, module_name)
    #     print('ori_module: ', ori_module, type(ori_module))
    #     if isinstance(ori_module, torch.nn.Linear):
    #         new_module = QuantLinear(ori_module, weight_quant_params=args.weight_quant_params,
    #                                 act_quant_params=args.act_quant_params, wbits=w_bits, abits=a_bits)
    #         new_module.set_quant_state(weight_quant=True, act_quant=True)
    #         ori_modules[module_name] = ori_module
    #         new_modules.append(new_module)
    #         setattr(layers, module_name, new_module)
    #         print('new_module: ', type(new_module)) # <class 'quantize.blocks.int_linear.QuantLinear'>
    #         print('new_module: ', type(getModuleByName(layers, module_name))) # 但这里还是<class 'torch.nn.Linear'>

    with torch.no_grad():
        # forward inference only block layers
        attention_mask_batch = inputs['attention_mask_batch']
        position_ids = inputs['position_ids']

        for layer in block:
            for j in range(args.nsamples//args.batch_size):    
                index = j * args.batch_size
                quant_out[index:index+args.batch_size] = layer(quant_out[index:index+args.batch_size], attention_mask=attention_mask_batch,position_ids=position_ids)[0]

    for module_name, scheme in quant_config.items():
        w_bits = int(scheme[1])
        a_bits = int(scheme[3])
        module = getModuleByName(layers, module_name)
        assert isinstance(module, QuantLinear), 'module is not a QuantLinear'
        module.set_quant_state(weight_quant=False, act_quant=False)

    # # restore original modules
    # for module_name, ori_module in ori_modules.items():
    #     setattr(layers, module_name, ori_module)
    # for new_module in new_modules:
    #     del new_module
    return quant_out

def mixed_quant_optimze(hm, module_bitops, module_size, schemes_per_module=3,bitops_bound=np.inf,size_bound=np.inf, PSD=False):
    def generate_psd_matrix(dim):
        A = np.random.rand(dim, dim)  # 生成随机矩阵
        psd_matrix = np.dot(A, A.T)  # A * A^T
        return psd_matrix

    if hm.__class__ == torch.Tensor:
        hm = hm.cpu().numpy()
    x = cp.Variable(hm.shape[0], boolean=True)
    assert hm.shape[0]%schemes_per_module == 0, 'schemes_per_module must be a divisor of L'
    num_modules = hm.shape[0]//schemes_per_module

    # hm = generate_psd_matrix(hm.shape[0])
    # print('hm: ', hm)

    if PSD:
        es, us = np.linalg.eig(hm)
        es[es<0] = 0
        hm = us@np.diag(es)@us.T
    hm = (hm + hm.T) / 2

    # print('hm: ', hm)
    hm = cp.atoms.affine.wraps.psd_wrap(hm)
    objective = cp.Minimize(cp.quad_form(x,hm))

    # objective = cp.Minimize(np.diagonal(hm)@x)
    
    equality_constraint_matrix = []
    for i in range(num_modules):
        col = np.zeros(hm.shape[0])
        col[i*schemes_per_module:(i+1)*schemes_per_module] = 1
        equality_constraint_matrix.append(col)
    equality_constraint_matrix = np.array(equality_constraint_matrix)
    
    constraints = [equality_constraint_matrix@x == np.ones((num_modules,)),
                   module_bitops@x/10**9<=bitops_bound,
                   module_size@x/8/1024/1024<=size_bound]

    # constraints = [equality_constraint_matrix@x == np.ones((num_modules,))]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    # Print result.
    print("Solution status", prob.status)
    print("A solution x is")
    print(x.value)
    # print(f"bitops: {x.value@module_bitops}")

    ans = x.value.tolist()

    def conver_list(lst):
        result = []
        for i in range(0, len(lst), schemes_per_module):
            group = lst[i:i+schemes_per_module]  # 取出每组的三个元素
            # 找到1的位置并添加到结果列表中
            result.append(group.index(1))
        return result
    return conver_list(ans)

def get_block_input(block_index, block_layers, layers, prev_input, attention_mask, position_ids, args, dev):
    """获取指定block的输入输出"""
    with torch.no_grad():
        # 获取当前block中实际要处理的层（去掉最后一个索引，因为是左闭右开）
        current_layers = list(range(block_layers[0], block_layers[-1]))
        current_block = [layers[i].to(dev) for i in current_layers]
        
        current_output = prev_input.clone().to(dev)
        for layer in current_block:
            for j in range(args.nsamples):
                current_output[j] = layer(current_output[j].unsqueeze(0), 
                                       attention_mask=attention_mask,
                                       position_ids=position_ids)[0]
        
        current_block = [layer.to('cpu') for layer in current_block]
        torch.cuda.empty_cache()
        
        return current_output.to('cpu')

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # 添加处理器到logger
    logger.addHandler(console_handler)
    return logger

def process_single_block(block_index, block_layers, layers, fp_inps, attention_mask, attention_mask_batch, position_ids, args, dev, logger, return_fp_flag=False):
    """处理单个block的搜索任务"""
    # 设置进程特定的logger
    process_logger = setup_logger(f'Block_{block_index}')
    # 确保当前进程使用指定的GPU
    torch.cuda.set_device(dev)
    process_logger.info(f"Process for block {block_index} started on device {dev}")
    
    try:
        # 将输入数据移动到指定设备
        fp_inps = fp_inps.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        if attention_mask_batch is not None:
            attention_mask_batch = attention_mask_batch.to(dev)
        if position_ids is not None:
            position_ids = position_ids.to(dev)

        # 获取当前block中实际要处理的层（去掉最后一个索引，因为是左闭右开）
        current_layers = list(range(block_layers[0], block_layers[-1]))
        block = [layers[i].to(dev) for i in current_layers]
        process_logger.info(f"Block {block_index}: Successfully moved layers to device {dev}")
        process_logger.info(f'Block {block_index} layers: {current_layers}')

        # update module_layer_map and module_index
        module_layer_map = {}
        module_index = {}
        for layer_idx in current_layers:
            layer = layers[layer_idx]
            for name, module in layer.named_modules():
                if isinstance(module, types_to_quant) and 'gate' not in name:
                    module_layer_map[str(layer_idx) + '.' + name] = layer_idx
        
        cnt = 0
        for module_name in module_layer_map:
            for s in quant_scheme:
                module_index[module_name + '_' + s] = cnt
                cnt += 1
        L = cnt

        # update index2modulescheme
        index2modulescheme = [None for i in range(L)]
        for name in module_index:
            index = module_index[name]
            module_name = name[:-5]
            scheme = name[-4:]
            index2modulescheme[index] = (module_name,scheme)

        # update module_size and module_bitops
        module_size = np.array([0 for i in range(len(module_index))])
        module_bitops = np.array([0 for i in range(len(module_index))])
        for module_name in module_index:
            index = module_index[module_name]
            module_name, scheme = index2modulescheme[index]
            abits, wbits = int(scheme[3]), int(scheme[1])
            module_size[index] = torch.numel(getModuleByName(layers, module_name).weight) * int(wbits)
            module_bitops[index] = get_linear_bitops(layers, module_name, abits, wbits)

        layer_cost = {'module_index': module_index, 'module_size': module_size, 'module_bitops': module_bitops}

        # Forward inference to get full-precision output of this block
        fp_out = copy.deepcopy(fp_inps)
        with torch.no_grad():
            for layer in block:
                for j in range(args.nsamples):
                    fp_out[j] = layer(fp_out[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        for layer in block:
            for name, module in layer.named_modules():
                if isinstance(module, types_to_quant) and not 'gate' in name:
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, layer, quantlinear)

        # calculate sensitivity
        s_time = time.time()
        cached = {}

        # Prepare inputs dict for quant_and_forward
        inputs = {
            'hidden_states': fp_inps,
            'attention_mask_batch': attention_mask_batch,
            'position_ids': position_ids,
        }

        for n in tqdm(module_layer_map):
            for m in module_layer_map:
                for naw in quant_scheme:
                    for maw in quant_scheme:
                        if (n,m,naw,maw) not in cached:
                            loss = 0
                            quant_out = None
                            if n == m:
                                if naw == maw:
                                    quant_out = quant_and_forward(args,{n:naw},layers,block,inputs)
                                    loss = mse(quant_out, fp_out)
                                else:
                                    loss = 0
                            else:
                                quant_out = quant_and_forward(args,{n:naw,m:maw},layers,block,inputs)
                                loss = mse(quant_out, fp_out)
                            process_logger.info(f'Block {block_index} - {n} {m} {naw} {maw} loss: {loss}')
                            del quant_out
                            cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = loss
        process_logger.info(f'Block {block_index} - {time.time()-s_time:.2f} seconds elapsed')

        hm = np.zeros((L,L))
        for n in module_layer_map:
            for m in module_layer_map:
                for naw in quant_scheme:
                    for maw in quant_scheme:
                        hm[module_index[n + '_' + naw], module_index[m + '_' + maw]] = cached[(n,m,naw,maw)]
        
        hm_modify = np.zeros((L,L))
        for i in range(L):
            for j in range(L):
                module_i, scheme_i = index2modulescheme[i]
                module_j, scheme_j = index2modulescheme[j]
                if module_i == module_j:
                    if scheme_i == scheme_j:
                        hm_modify[i, j] = hm_modify[j, i] = 2 * hm[i, j]
                    else:
                        hm_modify[i, j] = hm_modify[j, i] = 0
                else:
                    hm_modify[i, j] = hm_modify[j, i] = hm[i,j] - hm[i,i] - hm[j,j]

        hm_info = {'hm': hm_modify, 'module_index': module_index, 'index2modulescheme': index2modulescheme}
        
        # 清理GPU内存
        for layer in block:
            layer.cpu()
        del block
        del cached, hm
        torch.cuda.empty_cache()
        if return_fp_flag:
            return block_index, layer_cost, hm_info, fp_out.cpu()
        else:
            del fp_out
            return block_index, layer_cost, hm_info, None

    except Exception as e:
        process_logger.error(f"Error processing block {block_index} on device {dev}: {str(e)}")
        # 发生异常时也要清理GPU内存
        torch.cuda.empty_cache()
        raise e

def search_quant_config_parallel(lm, blocks_config, args, dataloader, logger):
    """并行版本的量化配置搜索函数，使用动态调度"""
    from torch.cuda import device_count
    from collections import deque
    
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    dtype = torch.float16
    layers = model.model.layers
    
    # 获取第一个block的输入
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0}
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
    del dataloader
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1)
    else:
        logger.info("No attention mask caught from the first layer. Seems that model's attention works without a mask.")
        attention_mask_batch = None
    position_ids = cache["position_ids"] if is_llama else None

    # 设置可用的GPU数量和并行处理的block数量
    num_gpus = min(device_count(), len(blocks_config))
    parallel_blocks = num_gpus  # 同时处理的block数量
    logger.info(f"Using {num_gpus} GPUs for parallel search")

    # 初始化结果存储
    layer_cost = {}
    hm_info = {}
    
    # 使用队列存储block输入
    input_cache = {}
    current_input = inps

    # 按批次处理blocks，一次处理parallel_blocks个block
    for batch_start in range(0, len(blocks_config), parallel_blocks):
        batch_end = min(batch_start + parallel_blocks, len(blocks_config))
        current_batch = list(range(batch_start, batch_end))
        logger.info(f"Starting new batch: blocks {batch_start} to {batch_end-1}")
        logger.info(f"Available GPUs: {num_gpus}")

        # 获取当前批次所需的输入
        for block_idx in current_batch:
            if block_idx == batch_start:
                input_cache[block_idx] = current_input
            else:
                prev_block_layers = blocks_config[block_idx-1]
                input_cache[block_idx] = get_block_input(
                    block_idx-1, prev_block_layers, layers,
                    input_cache[block_idx-1], attention_mask,
                    position_ids, args, dev
                )
            logger.info(f"Got input for block {block_idx}")

        # 创建进程池处理当前批次
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(num_gpus)
        results = []

        # 分配任务给不同GPU
        for block_idx in current_batch:
            gpu_id = (block_idx - batch_start) % num_gpus
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Preparing to assign block {block_idx} to {device}")
            
            block_layers = blocks_config[block_idx]
            result = pool.apply_async(
                process_single_block,
                args=(block_idx, block_layers, layers, input_cache[block_idx],
                     attention_mask, attention_mask_batch, position_ids, args, device, logger, (block_idx==current_batch[-1]))
            )
            results.append(result)
            logger.info(f"Successfully queued block {block_idx} for processing on {device}")

        # 收集当前批次的结果
        for i, result in enumerate(results):
            try:
                block_idx, block_layer_cost, block_hm_info, fp_out = result.get()
                logger.info(f"Successfully processed block {block_idx}")
                layer_cost[block_idx] = block_layer_cost
                hm_info[block_idx] = block_hm_info
                if fp_out is not None:
                    current_input = fp_out
            except Exception as e:
                logger.error(f"Error processing result {i}: {str(e)}")
                raise e

        pool.close()
        pool.join()

        # # 更新下一批次的起始输入
        # if batch_end < len(blocks_config):
        #     prev_block_layers = blocks_config[batch_end-1]
        #     current_input = get_block_input(
        #         batch_end-1, prev_block_layers, layers,
        #         input_cache[batch_end-1], attention_mask,
        #         position_ids, args, dev
        #     )

        # 清理当前批次的输入缓存
        for block_idx in current_batch[:-1]:  # 保留最后一个block的输入用于下一批次
            if block_idx in input_cache:
                del input_cache[block_idx]
        torch.cuda.empty_cache()

    # 保存中间结果
    with open(f'{args.output_dir}/layer_cost_{args.net}.pkl', 'wb') as f:
        pickle.dump(layer_cost, f)
    with open(f'{args.output_dir}/hm_info_{args.net}.pkl', 'wb') as f:
        pickle.dump(hm_info, f)

    # 计算最终的量化结果
    quant_result = {}
    quant_map = {}

    for i in range(len(blocks_config)):
        hm = hm_info[i]['hm']
        module_bitops = layer_cost[i]['module_bitops']
        module_size = layer_cost[i]['module_size']
        size = module_size.sum()/(4+8+8)/1024/1024
        size_bound = size * args.size_bound_factor
        ans = mixed_quant_optimze(hm, module_bitops, module_size, bitops_bound=np.inf, size_bound=size_bound, PSD=True)
        print('block ', i, ' ans: ', ans)
        quant_result[i] = ans

    with open(f'{args.output_dir}/quant_result_{args.net}_{args.size_bound_factor}.pkl', 'wb') as f:
        pickle.dump(quant_result, f)

    for i in range(len(quant_result)):
        module_index = {v: k for k, v in hm_info[i]['module_index'].items()}
        print(type(quant_result[i]))
        for j in range(len(quant_result[i])):
            if int(quant_result[i][j]) == 1:
                modulename = module_index[j][:-5]
                layer = modulename.split('.')[0]
                modulename = modulename.split('.', 1)[1]
                if layer not in quant_map:
                    quant_map[layer] = {}
                quant_map[layer][modulename] = j % 3
    with open(f'{args.output_dir}/quant_map_{args.net}.pkl', 'wb') as f:
        pickle.dump(quant_map, f)

    # 恢复原始use_cache设置
    model.config.use_cache = use_cache
    return quant_result