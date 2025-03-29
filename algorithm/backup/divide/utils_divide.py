import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from quantize.int_linear import QuantLinear
import logging

logger = logging.getLogger(__name__)

def determine_blocks(args, quant_errors, layer_similarities, sensitivities):
    """确定block划分
    
    基于量化误差、层间相似度和Hessian敏感度三个指标来划分block。
    当任一指标不满足条件时，就在该处进行分割。
    
    Args:
        args: 配置参数
        quant_errors: 每层的量化误差列表
        layer_similarities: 层间相似度列表 
        sensitivities: 每层的Hessian敏感度列表
        
    Returns:
        list: block划分列表，每个元素为(start_idx, end_idx)
    """
    error_threshold = args.error_threshold
    similarity_threshold = args.similarity_threshold 
    sensitivity_threshold = args.sensitivity_threshold
    max_block_size = args.max_block_size
    
    n_layers = len(quant_errors)
    assert len(layer_similarities) == n_layers
    assert len(sensitivities) == n_layers
    
    blocks = []
    current_block_start = 0
    
    def should_split(start, current):
        """判断是否应该在current位置分割block"""
        if current - start >= max_block_size:
            return True
            
        # 1. 检查累计量化误差
        error_sum = sum(quant_errors[start:current+1])
        if error_sum > error_threshold:
            return True
            
        # 2. 检查相邻层的相似度
        if layer_similarities[current] < similarity_threshold:
            return True
            
        # 3. 检查相邻层的敏感度差异
        if abs(sensitivities[current] - sensitivities[current-1]) > sensitivity_threshold:
            return True
            
        return False
    
    # 遍历所有层，根据条件决定在哪里分割
    for i in range(1, n_layers):
        if should_split(current_block_start, i):
            # 如果当前block只有一层，尝试与下一层合并
            if i - current_block_start == 1 and i < n_layers - 1:
                # 检查与下一层合并是否可行
                if not should_split(current_block_start, i+1):
                    continue
            
            blocks.append((current_block_start, i))
            current_block_start = i
    
    # 处理最后一个block
    if current_block_start < n_layers:
        blocks.append((current_block_start, n_layers))
    
    # 优化block大小：合并过小的block
    optimized_blocks = []
    i = 0
    while i < len(blocks):
        start, end = blocks[i]
        
        # 如果当前block太小且不是最后一个block
        if end - start < max_block_size // 2 and i < len(blocks) - 1:
            next_start, next_end = blocks[i + 1]
            merged_size = next_end - start
            
            # 检查合并后的block是否满足条件
            if merged_size <= max_block_size:
                error_sum = sum(quant_errors[start:next_end])
                if error_sum <= error_threshold:
                    # 合并blocks
                    optimized_blocks.append((start, next_end))
                    i += 2
                    continue
        
        optimized_blocks.append((start, end))
        i += 1
    
    # 打印每个block的指标信息用于调试
    for start, end in optimized_blocks:
        error_sum = sum(quant_errors[start:end])
        
        # 处理单层block的情况
        if end - start == 1:
            min_similarity = layer_similarities[start] if start > 0 else 1.0
            max_sensitivity_diff = abs(sensitivities[start] - sensitivities[start-1]) if start > 0 else 0.0
        else:
            min_similarity = min(layer_similarities[start+1:end])
            max_sensitivity_diff = max(abs(sensitivities[i] - sensitivities[i-1]) 
                                    for i in range(start+1, end))
        
        logger.info(f"Block {start}-{end}: size={end-start}, "
                   f"error_sum={error_sum:.4f}, "
                   f"min_similarity={min_similarity:.4f}, "
                   f"max_sensitivity_diff={max_sensitivity_diff:.4f}")
    
    return optimized_blocks


def compute_quant_error(layer, dataloader, dev):
    """计算单层的量化误差
    
    通过比较量化前后的输出来计算量化误差。
    
    Args:
        layer: decoder layer
        dataloader: 校准数据集的dataloader
        dev: 设备(GPU/CPU)
        
    Returns:
        float: 归一化后的量化误差分数(0-1之间)
    """
    layer.eval()
    total_error = 0
    num_batches = 0
    
    # 保存原始权重
    if hasattr(layer, 'named_modules'):
        orig_weights = {}
        for name, module in layer.named_modules():
            if isinstance(module, (nn.Linear, QuantLinear)):
                orig_weights[name] = module.weight.data.clone()
    
    with torch.no_grad():
        for batch in dataloader:
            # 将输入数据移到指定设备
            inputs = batch[0].to(dev)
            batch_size = inputs.size(0)
            
            # 获取原始输出
            orig_output = layer(inputs)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
                
            # 对层进行量化
            if hasattr(layer, 'named_modules'):
                for name, module in layer.named_modules():
                    if isinstance(module, (nn.Linear, QuantLinear)):
                        weight = module.weight.data
                        # 使用简单的对称量化
                        max_val = torch.max(torch.abs(weight))
                        scale = max_val / 127.0  # 对于8位量化
                        quant_weight = torch.round(weight / scale) * scale
                        module.weight.data = quant_weight
            
            # 获取量化后的输出
            quant_output = layer(inputs)
            if isinstance(quant_output, tuple):
                quant_output = quant_output[0]
                
            # 计算MSE误差
            error = F.mse_loss(quant_output, orig_output)
            total_error += error.item() * batch_size
            num_batches += batch_size
            
            # 恢复原始权重
            if hasattr(layer, 'named_modules'):
                for name, module in layer.named_modules():
                    if isinstance(module, (nn.Linear, QuantLinear)):
                        module.weight.data = orig_weights[name]
            
            # 只使用少量batch来估计误差
            if num_batches >= 32:
                break
    
    avg_error = total_error / num_batches if num_batches > 0 else 0
    
    # 将误差归一化到0-1范围
    normalized_error = 2.0 / (1.0 + math.exp(-avg_error)) - 1.0
    
    return normalized_error

def compute_layer_similarity(layer1, layer2, bin_num=256):
    """计算两层之间的分布相似度
    
    通过比较两层中对应linear层的权重分布相似度来衡量层间相似度。
    使用KL散度来度量分布的相似程度。
    
    Args:
        layer1: 第一个decoder layer
        layer2: 第二个decoder layer
        bin_num: 直方图的bin数量
        
    Returns:
        float: 两层的相似度分数(0-1之间,1表示完全相似)
    """
    similarities = []
    
    # 获取所有linear层的权重
    linear_weights1 = []
    linear_weights2 = []
    
    for name1, module1 in layer1.named_modules():
        if isinstance(module1, (nn.Linear, QuantLinear)):
            linear_weights1.append(module1.weight.data.float())
            
    for name2, module2 in layer2.named_modules():
        if isinstance(module2, (nn.Linear, QuantLinear)):
            linear_weights2.append(module2.weight.data.float())
            
    # 确保两层有相同数量的linear层
    assert len(linear_weights1) == len(linear_weights2)
    
    # 计算每对对应linear层之间的相似度
    for w1, w2 in zip(linear_weights1, linear_weights2):
        # 将权重展平并移到CPU
        w1_flat = w1.flatten().cpu()
        w2_flat = w2.flatten().cpu()
        
        # 计算数值范围以确定bin边界
        min_val = min(w1_flat.min().item(), w2_flat.min().item())
        max_val = max(w1_flat.max().item(), w2_flat.max().item())
        bins = torch.linspace(min_val, max_val, bin_num+1)
        
        # 计算直方图
        hist1 = torch.histogram(w1_flat, bins=bins, density=True)[0]
        hist2 = torch.histogram(w2_flat, bins=bins, density=True)[0]
        
        # 添加平滑项避免log(0)
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # 归一化
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # 计算KL散度
        kl_div = torch.sum(hist1 * torch.log(hist1 / hist2))
        
        # 将KL散度转换为相似度分数(0-1)
        similarity = torch.exp(-kl_div).item()
        similarities.append(similarity)
    
    # 返回所有linear层相似度的平均值
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    logger.info(f"Layer similarity: {mean_similarity}")
    return mean_similarity

def compute_hessian_sensitivity(layer, num_samples=10):
    """计算层的Hessian敏感度
    
    使用Hutchinson估计器来近似计算Hessian矩阵的迹，
    用来评估该层对量化的敏感程度。
    
    Args:
        layer: decoder layer
        num_samples (int): 随机向量采样数
        
    Returns:
        float: 该层的Hessian敏感度分数
    """
    def compute_trace_estimate(weight, num_samples):
        """计算Hessian迹估计的平均值
        
        Args:
            weight: shape [out_features, in_features] 的权重矩阵
            num_samples: 随机采样数量
        """
        trace_estimates = []
        for _ in range(num_samples):
            # 生成与权重同形状的随机向量
            v = torch.randn_like(weight, device=weight.device)
            v = v / torch.norm(v)
            
            # 计算 Hv：对于线性层 W ∈ R^{m×n}，Hessian近似为 W^T W ∈ R^{n×n}
            # 1. 先计算 Wv: [m×n] × [m×n] -> [m×n]
            Wv = weight * v  # 逐元素相乘
            # 2. 再计算 W^T(Wv): [m×n] × [m×n] -> [m×n]
            Hv = weight * Wv  # 逐元素相乘
            
            # 计算迹：tr(H) = tr(W^T W) ≈ v^T Hv
            trace_estimate = torch.sum(v * Hv).item()
            trace_estimates.append(trace_estimate)
            
        return sum(trace_estimates) / num_samples

    sensitivities = []
    for name, module in layer.named_modules():
        if isinstance(module, (nn.Linear, QuantLinear)) and hasattr(module, "weight"):
            weight = module.weight.data.float()
            sensitivity = compute_trace_estimate(weight, num_samples)
            sensitivities.append(sensitivity)

    if sensitivities:
        avg_sensitivity = sum(sensitivities) / len(sensitivities)
        normalized_sensitivity = 1.0 / (1.0 + math.exp(-avg_sensitivity))  # Sigmoid归一化
        logger.info(f"Hessian sensitivity: {normalized_sensitivity}")
        return normalized_sensitivity
    else:
        logger.info("No linear layers found in the layer.")
        return 0.0

def visualize_layer_metrics(quant_errors, layer_similarities, sensitivities, save_path):
    """可视化层间指标
    
    将量化误差、层间相似度和Hessian敏感度绘制成图表
    
    Args:
        quant_errors: 每层的量化误差列表
        layer_similarities: 层间相似度列表
        sensitivities: 每层的Hessian敏感度列表
        save_path: 图表保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制量化误差
    plt.subplot(131)
    plt.plot(quant_errors, 'b-', label='Quantization Error')
    plt.title('Layer-wise Quantization Error')
    plt.xlabel('Layer Index')
    plt.ylabel('Error')
    plt.legend()
    
    # 绘制层间相似度
    plt.subplot(132)
    plt.plot(layer_similarities, 'g-', label='Layer Similarity')
    plt.title('Layer-wise Similarity')
    plt.xlabel('Layer Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    
    # 绘制Hessian敏感度
    plt.subplot(133)
    plt.plot(sensitivities, 'r-', label='Hessian Sensitivity')
    plt.title('Layer-wise Hessian Sensitivity')
    plt.xlabel('Layer Index')
    plt.ylabel('Sensitivity Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def com_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('compensation') > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_abq_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find('compensation') > -1:
            params.append(m)
    return iter(params)  

def abq_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            # 对activation的缩放融入到上一层的layernorm中
            # 对weight直接缩放
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift, self_attn=model.self_attn)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale, self_attn=model.self_attn)
            smooth_fc_fc_temporary(model.mlp.up_proj,model.mlp.down_proj,model.fc2_smooth_scale,model.fc2_smooth_shift) # balance up & down
            # model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
            # smooth_fc_fc_temporary(model.fc1,model.fc2,model.fc2_smooth_scale, model.fc2_smooth_shift)
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # if hasattr(module, "temp_weight"):
            #     module.temp_weight = module.weight_quantizer(module.temp_weight)
            # else:
            #     module.temp_weight = module.weight_quantizer(module.weight)
            if hasattr(module, "temp_weight"):
                temp_weight = module.temp_weight
            else:
                temp_weight = module.weight
            name_tmp = name.replace(".","_")
            if hasattr(model, f"{name_tmp}_compensation_left"):
                compensation_left = getattr(model, f"{name_tmp}_compensation_left")
                compensation_right = getattr(model, f"{name_tmp}_compensation_right")
                temp_weight = temp_weight + compensation_left @ compensation_right

            module.temp_weight = module.weight_quantizer(temp_weight)
            # temp_weight = module.weight_quantizer(temp_weight)
            # module.temp_weight = torch.nn.Parameter(temp_weight)

            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift, self_attn=model.self_attn)
            smooth_fc_fc_inplace(model.mlp.up_proj,model.mlp.down_proj,model.fc2_smooth_scale,model.fc2_smooth_shift) # balance up & down
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            # smooth_fc_fc_inplace(model.fc1,model.fc2,model.fc2_smooth_scale, model.fc2_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale, self_attn=model.self_attn)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            weight = module.weight
            name_tmp = name.replace(".","_")
            if hasattr(model, f"{name_tmp}_compensation_left"):
                compensation_left = getattr(model, f"{name_tmp}_compensation_left")
                compensation_right = getattr(model, f"{name_tmp}_compensation_right")
                weight = weight + compensation_left @ compensation_right
            module.weight = module.weight_quantizer(weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        # 实际QuantMatMul不会用到weight_quant、act_quant
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)