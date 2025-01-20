import torch

def quantize_per_tensor(tensor, num_bits=8):
    """
    对tensor进行per-tensor量化
    
    Args:
        tensor (torch.Tensor): 输入tensor
        num_bits (int): 量化位数,默认8bit
        
    Returns:
        tuple: (量化后的tensor, scale, zero_point)
        注意：当num_bits=4时，返回的tensor类型为torch.uint8，其他情况为torch.int8
    """
    # 计算range
    min_val = tensor.min()
    max_val = tensor.max()
    
    # 计算scale和zero_point
    scale = (max_val - min_val) / (2**num_bits - 1)
    zero_point = -2**(num_bits - 1) - min_val / scale
    
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
    return output, scale.to(torch.float32), zero_point.to(torch.float32)

def dequantize_tensor(q_tensor, input_scale, input_zero_point, weight_scale, weight_zero_point, K=None):
    """
    对量化后的GEMM结果进行反量化
    
    Args:
        q_tensor (torch.Tensor): 量化后的GEMM结果tensor
        input_scale (torch.Tensor): 输入tensor的量化scale
        input_zero_point (torch.Tensor): 输入tensor的量化zero point
        weight_scale (torch.Tensor): 权重tensor的量化scale
        weight_zero_point (torch.Tensor): 权重tensor的量化zero point
        K (int, optional): 矩阵乘法中的K维度,用于zero point校正
        
    Returns:
        torch.Tensor: 反量化后的tensor
    """
    # 如果是打包的int4数据，需要先解包
    if q_tensor.dtype == torch.uint8 and K is not None and K % 2 == 0:
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

    if K is not None:
        # 考虑zero point的反量化
        # output = (q_tensor - K * input_zp * weight_zp) * input_scale * weight_scale
        zp_correction = float(K) * input_zero_point * weight_zero_point
        output = (q_tensor.float() - zp_correction) * input_scale * weight_scale
    else:
        output = q_tensor.float() * input_scale * weight_scale
        
    return output.to(torch.float16)

def quantize_per_token(tensor, num_bits=8):
    """
    对tensor进行per-token量化
    
    Args:
        tensor (torch.Tensor): 输入tensor [M, K]
        num_bits (int): 量化位数,默认8bit
        
    Returns:
        tuple: (量化后的tensor, scale, zero_point)
    """
    M = tensor.size(0)
    
    # 初始化输出
    q_tensor = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(M, 1, dtype=torch.float32, device=tensor.device)
    zero_point = torch.empty(M, 1, dtype=torch.float32, device=tensor.device)
    
    # 对每个token进行量化
    for i in range(M):
        row = tensor[i]
        min_val = row.min()
        max_val = row.max()
        
        # 计算scale和zero point
        scale[i] = (max_val - min_val) / (2**num_bits - 1)
        zero_point[i] = -128 - min_val / scale[i]
        
        # 量化
        q_row = row / scale[i] + zero_point[i]
        q_tensor[i] = torch.clamp(q_row, -128, 127)
        
    return q_tensor, scale, zero_point

def quantize_per_channel(tensor, num_bits=8):
    """
    对tensor进行per-channel量化
    
    Args:
        tensor (torch.Tensor): 输入tensor [N, K]
        num_bits (int): 量化位数,默认8bit
        
    Returns:
        tuple: (量化后的tensor, scale, zero_point)
    """
    N = tensor.size(0)
    
    # 初始化输出
    q_tensor = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(N, 1, dtype=torch.float32, device=tensor.device)
    zero_point = torch.empty(N, 1, dtype=torch.float32, device=tensor.device)
    
    # 对每个channel进行量化
    for i in range(N):
        channel = tensor[i]
        min_val = channel.min()
        max_val = channel.max()
        
        # 计算scale和zero point
        scale[i] = (max_val - min_val) / (2**num_bits - 1)
        zero_point[i] = -128 - min_val / scale[i]
        
        # 量化
        q_channel = channel / scale[i] + zero_point[i]
        q_tensor[i] = torch.clamp(q_channel, -128, 127)
        
    return q_tensor, scale, zero_point 