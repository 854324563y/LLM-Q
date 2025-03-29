import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
from models.LMClass import LMClass
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from quantize.adaptive.int_linear import QuantLinear
import copy
import sys

import cvxpy as cp

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
        # print(module_name,type(module), module.use_weight_quant, module.use_act_quant)

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
        # print(module_name,type(module), module.use_weight_quant, module.use_act_quant)

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
    # ans = x.value

    # def conver_list(lst):
    #     result = []
    #     for i in range(0, len(lst), schemes_per_module):
    #         group = lst[i:i+schemes_per_module]  # 取出每组的三个元素
    #         # 找到1的位置并添加到结果列表中
    #         result.append(group.index(1))
    #     return result
    return ans

def search_quant_config(lm, blocks_config, args, dataloader, logger):
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
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")

    layers[0] = layers[0].to(dev)
    dtype = torch.float16
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
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1)
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    types_to_quant = (torch.nn.Linear)
    quant_map = {}

    for i in range(len(layers)):
        logger.info(f"=== Start Search layer {i} ===")
        layer = layers[i].to(dev)

        # Forward inference to get full-precision output of this layer
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # Get all linear modules in this layer
        module_dict = {}
        for name, module in layer.named_modules():
            # gate这里是不量化的
            if isinstance(module, types_to_quant) and not 'gate' in name:
                module_dict[name] = module

        # Calculate sensitivity for each module
        cached = {}
        for n in tqdm(module_dict):
            for s in quant_scheme:
                loss = 0
                quant_out = None
                quantlinear = QuantLinear(module_dict[n], args.weight_quant_params, args.act_quant_params)
                w_bits = int(s[1])
                a_bits = int(s[3])
                quantlinear.change_n_bits(w_bits, a_bits)
                quantlinear.set_quant_state(weight_quant=True, act_quant=True)
                
                # Replace the original module with quantized one
                add_new_module(n, layer, quantlinear)
                
                # Forward with quantized module
                with torch.no_grad():
                    for j in range(args.nsamples):
                        quant_out = layer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        loss += mse(quant_out, fp_inps[j])
                
                # Restore original module
                add_new_module(n, layer, module_dict[n])
                
                cached[(n,s)] = loss / args.nsamples
                logger.info(f'{n} {s} loss: {loss/args.nsamples}')
                del quant_out

        # Select best scheme for each module
        layer_quant_map = {}
        for n in module_dict:
            min_loss = float('inf')
            best_scheme = None
            for s in quant_scheme:
                if cached[(n,s)] < min_loss:
                    min_loss = cached[(n,s)]
                    best_scheme = s
            layer_quant_map[n] = quant_scheme.index(best_scheme)
        
        quant_map[str(i)] = layer_quant_map
        
        # Update input for next layer
        with torch.no_grad():
            for j in range(args.nsamples):
                quant_inps[j] = layer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        layer = layer.to('cpu')
        torch.cuda.empty_cache()

    with open(f'{args.output_dir}/quant_map_{args.net}.pkl', 'wb') as f:
        pickle.dump(quant_map, f)

    model.config.use_cache = use_cache
    return model

def load_search_quant_config(lm, blocks_config, args, dataloader, logger):
    with open(f'{args.output_dir}/layer_cost_{args.net}.pkl', 'rb') as f:
        layer_cost = pickle.load(f)
    with open(f'{args.output_dir}/hm_info_{args.net}.pkl', 'rb') as f:
        hm_info = pickle.load(f)

    quant_result = {}
    quant_map = {}

    # def mixed_quant_optimze(hm, module_bitops, module_size, schemes_per_module=3,bitops_bound=np.inf,size_bound=np.inf, PSD=False):
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