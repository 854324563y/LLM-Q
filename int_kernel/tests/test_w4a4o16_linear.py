import torch
import pytest
from torch import nn

from int_kernel.nn.linear import W4A4O16Linear

# @pytest.mark.parametrize("batch_size", [16])
# @pytest.mark.parametrize("in_features", [4096])
# @pytest.mark.parametrize("out_features", [4096])
# @pytest.mark.parametrize("bias", [True])
def test_w4a4_o16_linear(batch_size, in_features, out_features, bias):
    """测试W4A4O16Linear的功能正确性
    
    将结果与torch.nn.Linear的结果进行对比
    """
    # 创建输入
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    
    # 创建量化Linear和普通Linear
    q_linear = W4A4O16Linear(in_features, out_features, bias=bias).cuda().half()
    ref_linear = nn.Linear(in_features, out_features, bias=bias).cuda().half()
    
    # 复制权重和偏置
    with torch.no_grad():
        ref_linear.weight.copy_(q_linear.weight)
        if bias:
            ref_linear.bias.copy_(q_linear.bias)
    
    # 前向传播
    with torch.no_grad():
        q_out = q_linear(x)
        q_out_absmax = q_linear.forward_absmax(x)
        ref_out = ref_linear(x)
    
    print('q_out', q_out)
    print('q_out_absmax', q_out_absmax)
    print('ref_out', ref_out)
    
    # 检查输出形状
    assert q_out.shape == ref_out.shape

    # 计算均方误差
    error = (q_out - ref_out).pow(2).mean()
    print('square error', error)

    # 计算均方误差
    error = (q_out_absmax - ref_out).pow(2).mean()
    print('absmax square error', error)

    # 检查数值误差
    # 由于4-bit量化会导致更大的精度损失,我们使用相对宽松的误差阈值
    rtol = 0.2  # 相对误差阈值
    atol = 0.2  # 绝对误差阈值
    torch.testing.assert_close(q_out, ref_out, rtol=rtol, atol=atol)

    # 检查absmax输出
    rtol = 0.2  # 相对误差阈值
    atol = 0.2  # 绝对误差阈值
    torch.testing.assert_close(q_out_absmax, ref_out, rtol=rtol, atol=atol)


def test_w4a4_o16_linear2(M, N, K):
    from int_kernel._CUDA import s4t_s4n_f16t_gemm

    # torch.manual_seed(0)
    def unpack_uint8_to_int4(x):
        """将 uint8 解包成两个 int4 (-8 到 7 的范围)"""
        # 计算低 4 位和高 4 位
        low = x & 0xF
        high = (x >> 4) & 0xF
        # 转为张量并调整范围
        low = torch.tensor(low, dtype=torch.int8)
        high = torch.tensor(high, dtype=torch.int8)
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        # 拼接为单个张量
        return torch.stack([high, low], dim=-1).view(-1, x.shape[-1]*2)

    def unpack_uint8_to_int4_reverse(x):
        """将 uint8 解包成两个 int4 (-8 到 7 的范围)"""
        # 计算低 4 位和高 4 位
        low = x & 0xF
        high = (x >> 4) & 0xF
        # 转为张量并调整范围
        low = torch.tensor(low, dtype=torch.int8)
        high = torch.tensor(high, dtype=torch.int8)
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        # 拼接为单个张量
        return torch.stack([low, high], dim=-1).view(-1, x.shape[-1]*2)

    # 但看kernel，其精度是正常的


    #x_test = torch.full((M, K//2), 17, dtype=torch.uint8, device="cuda")
    #w_test = torch.full((N, K//2), 17, dtype=torch.uint8, device="cuda")
    x_test = torch.randint(0, 255, (M, K//2), dtype=torch.uint8, device="cuda")
    w_test = torch.randint(0, 255, (N, K//2), dtype=torch.uint8, device="cuda")

    print('x_test', x_test.shape, x_test)
    print('w_test', w_test.shape, w_test)
    print("x_test前几个元素:", x_test[0, :5])
    for elem in x_test[0, :5]:
        print(bin(elem.item()), end=' ')
    print("\nw_test前几个元素:", w_test[0, :5])
    for elem in w_test[0, :5]:
        print(bin(elem.item()), end=' ')

    cuda_output = s4t_s4n_f16t_gemm(x_test, w_test)

    # 解包并转换为fp16进行计算
    x_unpacked = unpack_uint8_to_int4(x_test).to(torch.float32)
    w_unpacked = unpack_uint8_to_int4(w_test).to(torch.float32)
    # 打印解包后的值
    print("\n解包后的值:")
    print("x_unpacked前几个元素:", x_unpacked.shape, x_unpacked[0, :10])
    print("w_unpacked前几个元素:", w_unpacked.shape, w_unpacked[0, :10])

    # PyTorch参考计算
    torch_output = torch.matmul(x_unpacked, w_unpacked.t())

    # 打印结果对比
    print("\nCUDA kernel输出:")
    print(cuda_output, cuda_output.shape)
    print("\nPyTorch参考输出:")
    print(torch_output, torch_output.shape)

    # 计算误差
    abs_diff = torch.abs(cuda_output - torch_output)
    rel_diff = abs_diff / (torch.abs(torch_output) + 1e-6)
    
    print(f"\n精度统计:")
    print(f"最大绝对误差: {abs_diff.max().item():.6f}")
    print(f"平均绝对误差: {abs_diff.mean().item():.6f}")
    print(f"最大相对误差: {rel_diff.max().item():.6f}")
    print(f"平均相对误差: {rel_diff.mean().item():.6f}")

if __name__ == '__main__':
    test_w4a4_o16_linear(batch_size=16, in_features=256, out_features=128, bias=True)
    #test_w4a4_o16_linear2(1, 4096, 4096) # (M,N,K)
    #test_w4a4_o16_linear2(48, 4096, 4096) # (M,N,K)
