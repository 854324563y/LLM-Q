#!/usr/bin/env python3
import os
import subprocess
import torch
from datetime import datetime

def get_gpu_count():
    """获取系统中可用的GPU数量"""
    return torch.cuda.device_count()

def run_experiment(gpu_id, bitops_factor):
    """在指定GPU上运行实验"""
    # model = "Llama-2-13b-hf"
    model = "llama-13b-hf"
    log_dir = f"./log-adaptive-calibration/{model}_{bitops_factor}"
    quant_map = f"log-adaptive/{model}/quant_map_{model}_{bitops_factor}.pkl"
    
    # 构建命令
    cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} nohup python main_calib_config2.py \
        --model /workspace/volume/inference-soft-data/AE/llm/models/{model} \
        --epochs 20 \
        --output_dir {log_dir} \
        --eval_ppl \
        --wbits 4 \
        --abits 4 \
        --let \
        --lwc \
        --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande \
        --compensation_calibration \
        --quant_map {quant_map} &"""
    
    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 运行命令
    print(f"Starting experiment on GPU {gpu_id} with bitops_factor {bitops_factor}")
    subprocess.run(cmd, shell=True)

def main():
    # 定义要测试的bitops_bound_factors
    #bitops_bound_factors = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    bitops_bound_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #bitops_bound_factors = [0.9, 0.95]
    
    # 获取可用的GPU数量
    gpu_count = get_gpu_count()
    print(f"Found {gpu_count} available GPUs")
    
    # 为每个任务分配GPU
    for i, bitops_factor in enumerate(bitops_bound_factors):
        gpu_id = i % gpu_count  # 循环使用GPU
        run_experiment(gpu_id, bitops_factor)
        print(f"Launched experiment {i+1}/{len(bitops_bound_factors)} on GPU {gpu_id}")

if __name__ == "__main__":
    main() 
