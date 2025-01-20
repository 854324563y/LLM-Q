import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantize.quantize import (
    quantize_per_tensor,
    dequantize_tensor,
)
from ..quantize.quantize_minmax import (
    quantize_per_tensor_absmax,
    dequantize_tensor_absmax,
)
from ..quantize.quantize_abq import quantize_weight_per_channel
from .._CUDA import s4t_s4n_f16t_gemm, s8t_s8n_f16t_gemm, s8t_s4n_f16t_gemm
# from .._CUDA import s8t_s8n_f16t_gemm

class W8A8O16Linear(nn.Module):
    """量化版本的Linear层
    
    使用int8量化来加速计算,同时保持FP16的精度
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化参数
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale, input_zp = quantize_per_tensor(x)
        q_weight, weight_scale, weight_zp = quantize_per_tensor(self.weight)

        # 执行量化GEMM
        output = s8t_s8n_f16t_gemm(q_input, q_weight)
        # 反量化结果
        output = dequantize_tensor(
            output,
            input_scale, input_zp,
            weight_scale, weight_zp,
            self.in_features
        )        
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
                    
        return output
    
    def forward_absmax(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale = quantize_per_tensor_absmax(x)
        q_weight, weight_scale = quantize_per_tensor_absmax(self.weight)

        # 执行量化GEMM
        output = s8t_s8n_f16t_gemm(q_input, q_weight)

        
        # 反量化结果
        output = dequantize_tensor_absmax(
            output,
            input_scale,
            weight_scale,
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
            
        return output

    def forward_abq(self, x):
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale, input_zp = quantize_per_tensor(x)
        q_weight = quantize_weight_per_channel(self.weight, self.scales, self.zeros)

        # 执行量化GEMM
        output = s8t_s8n_f16t_gemm(q_input, q_weight)
        # 反量化结果
        output = dequantize_tensor(
            output,
            input_scale, input_zp,
            weight_scale, weight_zp,
            self.in_features
        )        
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
                    
        return output

class W4A4O16Linear(nn.Module):
    """量化版本的Linear层
    
    使用int4量化来加速计算,同时保持FP16的精度
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化参数
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale, input_zp = quantize_per_tensor(x, num_bits=4)
        q_weight, weight_scale, weight_zp = quantize_per_tensor(self.weight, num_bits=4)

        # 执行量化GEMM
        print('q_input', q_input.dtype, q_input.shape)
        print('q_weight', q_weight.dtype, q_weight.shape)
        output = s4t_s4n_f16t_gemm(q_input, q_weight)
        
        # 反量化结果
        output = dequantize_tensor(
            output,
            input_scale, input_zp,
            weight_scale, weight_zp,
            self.in_features
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
                    
        return output
    
    def forward_absmax(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale = quantize_per_tensor_absmax(x, num_bits=4)
        q_weight, weight_scale = quantize_per_tensor_absmax(self.weight, num_bits=4)

        # 执行量化GEMM
        output = s4t_s4n_f16t_gemm(q_input, q_weight)
        
        # 反量化结果
        output = dequantize_tensor_absmax(
            output,
            input_scale,
            weight_scale,
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
            
        return output

class W4A8O16Linear(nn.Module):
    """量化版本的Linear层
    
    使用int8激活和int4权重来加速计算,同时保持FP16的精度
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化参数
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale, input_zp = quantize_per_tensor(x, num_bits=8)  # 8-bit activation
        q_weight, weight_scale, weight_zp = quantize_per_tensor(self.weight, num_bits=4)  # 4-bit weight

        # 执行量化GEMM
        output = s8t_s4n_f16t_gemm(q_input, q_weight)
        
        # 反量化结果
        output = dequantize_tensor(
            output,
            input_scale, input_zp,
            weight_scale, weight_zp,
            self.in_features
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
                    
        return output
    
    def forward_absmax(self, x):
        """前向传播
        
        Args:
            x: 输入tensor [*, in_features]
            
        Returns:
            输出tensor [*, out_features]
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, self.in_features)
            
        # 量化输入和权重
        q_input, input_scale = quantize_per_tensor_absmax(x, num_bits=8)  # 8-bit activation
        q_weight, weight_scale = quantize_per_tensor_absmax(self.weight, num_bits=4)  # 4-bit weight

        # 执行量化GEMM
        output = s8t_s4n_f16t_gemm(q_input, q_weight)
        
        # 反量化结果
        output = dequantize_tensor_absmax(
            output,
            input_scale,
            weight_scale,
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
            
        # 恢复原始形状
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)
            
        return output
