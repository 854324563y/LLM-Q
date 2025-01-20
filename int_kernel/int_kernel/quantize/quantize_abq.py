import torch
import torch

def quantize_weight_per_channel(weight, scales, zeros, num_bits=8):
    """
    对weight进行per-channel量化
    
    Args:
        tensor (torch.Tensor): 输入tensor
        num_bits (int): 量化位数,默认8bit
        
    Returns:
        tuple: (量化后的tensor, scale, zero_point)
        注意：当num_bits=4时，返回的tensor类型为torch.uint8，其他情况为torch.int8
    """
    output = (weight / scales).round()
    output.add_(zeros)
    
    print('output', output)
    sys.exit()
    
    # 量化
    output = tensor / scale + zero_point
    if num_bits == 4:
        # 确保K维度是偶数
        if output.size(-1) % 2 != 0:
            raise ValueError("For int4 quantization, the last dimension must be even")
            
        # 将值限制在[-8, 7]范围内
        output = torch.clamp(output, -8, 7)
        
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
        output = torch.clamp(output, -2**(num_bits - 1), 2**(num_bits - 1) - 1).to(torch.int8)
    
    # 返回量化结果和参数
    return output

#def dequantize_per_channel