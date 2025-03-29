#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime

def run_experiment(bitops_bound_factor, size_bound_factor=1.0):
    # 创建日志目录
    model = "llama-7b-hf"
    #model = "Llama-2-7b-hf"
    #log_dir = f"./log-adaptive/{model}"
    log_dir = f"./log-divide-adaptive/{model}"

    blocks_pkl = f"./log-divide/{model}-w4a4/{model}_blocks.pkl"



    # 构建命令
    cmd = [
        "python", "main_quant_config.py",
        "--model", f"/workspace/volume/inference-soft-data/AE/llm/models/{model}",
        "--output_dir", log_dir,
        "--blocks_pkl", blocks_pkl,
        "--nsamples", "128",
        "--reload",
        "--bitops_bound_factor", str(bitops_bound_factor),
    ]
    
    # 运行命令并将输出重定向到日志文件
    log_file = f"{log_dir}/log_reload_{bitops_bound_factor}.txt"
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    # 定义网格搜索参数
    bitops_bound_factors = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #bitops_bound_factors = [0.95]
    size_bound_factor = 1.0  # 固定值
    
    # 执行网格搜索
    for bitops_factor in bitops_bound_factors:
        print(f"Running experiment with bitops_bound_factor = {bitops_factor}")
        run_experiment(bitops_factor, size_bound_factor)
        print(f"Completed experiment with bitops_bound_factor = {bitops_factor}")

if __name__ == "__main__":
    main() 

    # # pip install cvxpy gurobipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
    # 5.2