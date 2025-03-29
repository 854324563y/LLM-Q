import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import csv
from int_kernel._CUDA import s8t_s8n_f16t_gemm, s8t_s4n_f16t_gemm, s4t_s4n_f16t_gemm
from tabulate import tabulate
import numpy as np

def bench_func_latency(func, args, num_iter=1000):
    """测量函数的平均执行时间
    
    Args:
        func: 要测试的函数
        args: 函数参数列表
        num_iter: 迭代次数，默认1000
    
    Returns:
        float: 平均执行时间(ms)
    """
    cudnn.benchmark = True
    # 预热
    for i in range(100):
        func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for i in range(num_iter):
        func(*args)
    end.record()
    
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / num_iter
    print(f"Average inference time: {avg_time:.3f} ms")
    return avg_time

def test_s8t_s8n_benchmark(M=128, N=4096, K=4096):
    """测试s8t_s8n kernel的性能"""
    print(f"\n=== Testing s8t_s8n kernel (M={M}, N={N}, K={K}) ===")
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    return bench_func_latency(s8t_s8n_f16t_gemm, [x, w])

def test_s8t_s4n_benchmark(M=128, N=4096, K=4096):
    """测试s8t_s4n kernel的性能"""
    print(f"\n=== Testing s8t_s4n kernel (M={M}, N={N}, K={K}) ===")
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(0, 255, (N, K//2), dtype=torch.uint8, device="cuda")
    return bench_func_latency(s8t_s4n_f16t_gemm, [x, w])

def test_s4t_s4n_benchmark(M=128, N=4096, K=4096):
    """测试s4t_s4n kernel的性能"""
    print(f"\n=== Testing s4t_s4n kernel (M={M}, N={N}, K={K}) ===")
    x = torch.randint(0, 255, (M, K//2), dtype=torch.uint8, device="cuda")
    w = torch.randint(0, 255, (N, K//2), dtype=torch.uint8, device="cuda")
    return bench_func_latency(s4t_s4n_f16t_gemm, [x, w])

def test_fp16_linear_benchmark(M=128, N=4096, K=4096):
    """测试FP16 nn.Linear的性能作为基准"""
    print(f"\n=== Testing FP16 Linear (M={M}, N={N}, K={K}) ===")
    linear = nn.Linear(K, N).half().cuda()
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    return bench_func_latency(linear, [x])

def calculate_metrics(M, N, K, latency_ms):
    """计算性能指标
    
    Args:
        M, N, K: 矩阵维度
        latency_ms: 执行时间(ms)
    
    Returns:
        dict: 包含TFLOPS和Memory BW等指标
    """
    # 计算FLOPs (2*M*N*K 用于gemm运算)
    flops = 2 * M * N * K
    tflops = (flops / (latency_ms * 1e-3)) / 1e12
    
    # 估算内存带宽 (假设每个元素读一次)
    bytes_accessed = M * K + N * K + M * N  # 输入、权重和输出
    memory_bw = (bytes_accessed / (latency_ms * 1e-3)) / 1e9  # GB/s
    
    return {
        "TFLOPS": tflops,
        "Memory_BW_GBs": memory_bw
    }

def run_all_benchmarks():
    """运行所有benchmark测试"""
    # 定义测试维度
    batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    matrix_configs = [
        (4096, 4096),    # 标准大小
        (4096, 11008),   # 扩展K
        (11008, 4096),   # 扩展N
    ]
    
    results = {}
    for M in batch_sizes:
        for N, K in matrix_configs:
            print(f"\n=== Running benchmarks for M={M}, N={N}, K={K} ===")
            results[(M,N,K)] = {
                "fp16_linear": test_fp16_linear_benchmark(M, N, K),
                "s8t_s8n": test_s8t_s8n_benchmark(M, N, K),
                "s8t_s4n": test_s8t_s4n_benchmark(M, N, K),
                "s4t_s4n": test_s4t_s4n_benchmark(M, N, K)
            }
    
    # 打印汇总结果
    print("\n=== Performance Summary ===")
    for size, times in results.items():
        M, N, K = size
        print(f"\nMatrix size: M={M}, N={N}, K={K}")
        
        # 准备表格数据
        table_data = []
        headers = ["Kernel", "Latency(ms)", "TFLOPS", "Memory BW(GB/s)"]
        
        for kernel, time in times.items():
            metrics = calculate_metrics(M, N, K, time)
            table_data.append([
                kernel,
                f"{time:.3f}",
                f"{metrics['TFLOPS']:.2f}",
                f"{metrics['Memory_BW_GBs']:.2f}"
            ])
        
        # 使用tabulate打印美化的表格
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存结果到CSV文件
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'Batch_Size', 'N', 'K',
            'Kernel', 'Latency_ms', 'TFLOPS', 'Memory_BW_GBs',
            'Matrix_Size', 'Total_Params'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for size, times in results.items():
            M, N, K = size
            for kernel, time in times.items():
                metrics = calculate_metrics(M, N, K, time)
                writer.writerow({
                    'Batch_Size': M,
                    'N': N,
                    'K': K,
                    'Kernel': kernel,
                    'Latency_ms': f"{time:.3f}",
                    'TFLOPS': f"{metrics['TFLOPS']:.2f}",
                    'Memory_BW_GBs': f"{metrics['Memory_BW_GBs']:.2f}",
                    'Matrix_Size': f"{M}x{N}x{K}",
                    'Total_Params': N * K
                })
    
    print("\nDetailed results have been saved to benchmark_results.csv")

if __name__ == "__main__":
    run_all_benchmarks() 