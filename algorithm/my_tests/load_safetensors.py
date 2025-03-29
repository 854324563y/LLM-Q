import safetensors.torch
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def quant1():
    from quantize.quantizer import UniformAffineQuantizer


    model_path = '/workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf/model-00001-of-00002.safetensors'
    model = safetensors.torch.load_file(model_path)
    weight = model['model.layers.0.self_attn.q_proj.weight']
    
    print(weight)

    weight_quant_params = {
        "n_bits": 4,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": None,
        "lwc": False,
        "disable_zero_point": False
    }

    # 使用UniformAffineQuantizer
    quantizer = UniformAffineQuantizer(**weight_quant_params, shape=weight.shape)
    # 执行量化和反量化
    dequantized_weight_uaq = quantizer(weight)
    print(dequantized_weight_uaq)

def quant2():
    from quantize.quantizer import UniformAffineQuantizer
    # from quantize.blocks.quantizer import UniformAffineQuantizer

    model_path = '/workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf/model-00001-of-00002.safetensors'

    model = safetensors.torch.load_file(model_path)
    weight = model['model.layers.0.self_attn.q_proj.weight']

    def quantize_per_channel(x, num_bits=8):
        # 计算每个输出通道的最大最小值
        # 这里和UniformAffineQuantizer一样
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]

        
        # 计算量化范围
        # 这里scale和UniformAffineQuantizer一样
        scale = (max_val - min_val) / (2**num_bits - 1)
        # UniformAffineQuantizer 中 zero_point 是 -(xmin) / (self.scale)，有所区别
        zero_point = min_val
        # 量化
        x_int = torch.round((x - zero_point) / scale)
        x_int = torch.clamp(x_int, 0, 2**num_bits - 1)
        
        # 反量化
        x_dequant = x_int * scale + zero_point
        
        return x_dequant, x_int, scale, zero_point

    print("======= 方案1: 简单的per-channel量化 =======")
    # 执行量化和反量化
    dequantized_weight, quantized_weight, scale, zero_point = quantize_per_channel(weight)

    print(weight)
    print(dequantized_weight)

    # 计算量化误差
    abs_error = torch.abs(weight - dequantized_weight)
    relative_error = abs_error / (torch.abs(weight) + 1e-10)
    mse = torch.mean((weight - dequantized_weight) ** 2)
    max_error = torch.max(abs_error)

    print(f"均方误差 (MSE): {mse.item():.6f}")
    print(f"最大绝对误差: {max_error.item():.6f}")
    print(f"平均相对误差: {torch.mean(relative_error).item():.6f}")

    print("\n======= 方案2: UniformAffineQuantizer量化 =======")

    weight_quant_params = {
        "n_bits": 8,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": None,
        "lwc": False,
        "disable_zero_point": False
    }

    # 使用UniformAffineQuantizer
    quantizer = UniformAffineQuantizer(**weight_quant_params, shape=weight.shape)
    # 执行量化和反量化
    dequantized_weight_uaq = quantizer(weight)

    print(dequantized_weight_uaq)

    # 计算量化误差
    abs_error_uaq = torch.abs(weight - dequantized_weight_uaq)
    relative_error_uaq = abs_error_uaq / (torch.abs(weight) + 1e-10)
    mse_uaq = torch.mean((weight - dequantized_weight_uaq) ** 2)
    max_error_uaq = torch.max(abs_error_uaq)

    print(f"均方误差 (MSE): {mse_uaq.item():.6f}")
    print(f"最大绝对误差: {max_error_uaq.item():.6f}")
    print(f"平均相对误差: {torch.mean(relative_error_uaq).item():.6f}")

    # 打印两种方案的量化结果样例
    print("\n======= 权重样例对比 =======")
    print("原始权重前5个值:")
    print(weight[0, :5])
    print("\n方案1量化后的权重前5个值:")
    print(dequantized_weight[0, :5])
    print("\n方案2量化后的权重前5个值:")
    print(dequantized_weight_uaq[0, :5])





def load_safetensors():
    

    model_path = './log-calibration-compensation/quant/Llama-2-7b-hf-w8a8/model-00001-of-00003.safetensors'
    model = safetensors.torch.load_file(model_path)
    weight = model['model.layers.0.self_attn.q_proj.weight']
    print(weight)

    model_path = './log-calibration-compensation-lwc/quant/Llama-2-7b-hf-w8a8/model-00001-of-00003.safetensors'
    model = safetensors.torch.load_file(model_path)
    weight = model['model.layers.0.self_attn.q_proj.weight']
    print(weight)

# quant1()
load_safetensors()