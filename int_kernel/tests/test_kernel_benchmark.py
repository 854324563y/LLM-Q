import torch
import torch.backends.cudnn as cudnn
import csv
from int_kernel._CUDA import s8t_s8n_f16t_gemm, s8t_s4n_f16t_gemm, s4t_s4n_f16t_gemm

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

def run_all_benchmarks():
    """运行所有benchmark测试"""
    # 测试不同的矩阵大小
    sizes = [
        (1, 4096, 4096),     # 批大小为1
        (1, 4096, 11008),
        (1, 11008, 4096),
        (128, 4096, 4096),   # 标准大小
        (128, 4096, 11008),
        (128, 11008, 4096),
    ]
    
    results = {}
    for M, N, K in sizes:
        print(f"\n=== Running benchmarks for M={M}, N={N}, K={K} ===")
        results[(M,N,K)] = {
            "s8t_s8n": test_s8t_s8n_benchmark(M, N, K),
            "s8t_s4n": test_s8t_s4n_benchmark(M, N, K),
            "s4t_s4n": test_s4t_s4n_benchmark(M, N, K)
        }
    
    # 打印汇总结果
    print("\n=== Summary ===")
    for size, times in results.items():
        M, N, K = size
        print(f"\nMatrix size: M={M}, N={N}, K={K}")
        print("Kernel      | Latency (ms)")
        print("-----------|-------------")
        for kernel, time in times.items():
            print(f"{kernel:10s} | {time:10.3f}")
    
    # 保存结果到CSV文件
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['M', 'N', 'K', 'Kernel', 'Latency_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for size, times in results.items():
            M, N, K = size
            for kernel, time in times.items():
                writer.writerow({
                    'M': M,
                    'N': N,
                    'K': K,
                    'Kernel': kernel,
                    'Latency_ms': f"{time:.3f}"
                })
    
    print("\nResults have been saved to benchmark_results.csv")

if __name__ == "__main__":
    run_all_benchmarks() 