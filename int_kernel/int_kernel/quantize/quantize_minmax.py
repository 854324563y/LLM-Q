import torch
import numpy as np

def quantize_per_tensor_absmax(tensor, num_bits=8):
    """
    对tensor进行per-tensor absmax量化
    
    Args:
        tensor (torch.Tensor): 输入tensor
        num_bits (int): 量化位数,默认8bit
        
    Returns:
        tuple: (量化后的tensor, scale)
        注意：当num_bits=4时，返回的tensor类型为torch.uint8，其他情况为torch.int8
    """
    scale = tensor.abs().max() / (2**(num_bits - 1) - 1)
    # use inplace operation to save memory
    output = tensor / scale
    
    if num_bits == 4:
        # 确保K维度是偶数
        if output.size(-1) % 2 != 0:
            raise ValueError("For int4 quantization, the last dimension must be even")
            
        # 将值限制在[-8, 7]范围内
        output = output.round_().clamp_(-8, 7)
        
        # 重塑tensor以便打包
        shape = output.shape
        output = output.view(-1, shape[-1])
        
        # 创建packed tensor
        packed = torch.empty((output.size(0), output.size(1)//2), 
                           dtype=torch.uint8, device=output.device)
        
        # 打包相邻的两个4-bit值到一个8-bit值
        even_elements = output[:, ::2].to(torch.int8)
        odd_elements = output[:, 1::2].to(torch.int8)
        
        # 对于负数，我们需要只取最低4位
        even_elements = even_elements & 0xF
        odd_elements = odd_elements & 0xF
        
        # 使用uint8类型存储打包的值
        packed = (even_elements | (odd_elements << 4)).to(torch.uint8)
        
        # 恢复原始形状
        output = packed.view(*shape[:-1], shape[-1]//2)
    else:
        output = output.round_().to(torch.int8)
        
    return output, scale.to(torch.float32)

def dequantize_tensor_absmax(q_tensor, input_scale, weight_scale):
    # 如果是打包的int4数据，需要先解包
    if q_tensor.dtype == torch.uint8:
        # 解包int4数据
        shape = q_tensor.shape
        q_tensor = q_tensor.view(-1, shape[-1])
        
        # 创建解包后的tensor
        unpacked = torch.empty((q_tensor.size(0), q_tensor.size(1)*2), 
                             dtype=torch.int8, device=q_tensor.device)
        
        # 解包8-bit到两个4-bit值，保持符号
        unpacked[:, ::2] = (q_tensor & 0xF)
        unpacked[:, 1::2] = ((q_tensor >> 4) & 0xF)
        
        # 处理负数：如果最高位(第3位)为1，则将值转换为负数
        unpacked = unpacked.where(unpacked < 8, unpacked - 16)
        
        # 恢复原始形状
        q_tensor = unpacked.view(*shape[:-1], shape[-1]*2)
        
    output = q_tensor.float() * input_scale * weight_scale
    return output.to(torch.float16)