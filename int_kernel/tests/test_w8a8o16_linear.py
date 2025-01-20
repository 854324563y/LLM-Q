import torch
import pytest
from torch import nn

from int_kernel.nn.linear import W8A8O16Linear

@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("in_features", [4096])
@pytest.mark.parametrize("out_features", [4096, 11088])
@pytest.mark.parametrize("bias", [True])
def test_w8a8_o16_linear(batch_size, in_features, out_features, bias):
    """测试W8A8O16Linear的功能正确性
    
    将结果与torch.nn.Linear的结果进行对比
    """
    # 创建输入
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    
    # 创建量化Linear和普通Linear
    q_linear = W8A8O16Linear(in_features, out_features, bias=bias).cuda().half()
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
    
    # calculate mean square error
    error = (q_out - ref_out).pow(2).mean()
    print('square error', error)

    # calculate mean square error
    error = (q_out_absmax - ref_out).pow(2).mean()
    print('absmax square error', error)

    # # 检查数值误差
    # # 由于量化会导致精度损失,我们使用相对宽松的误差阈值
    # rtol = 0.1  # 相对误差阈值
    # atol = 0.1  # 绝对误差阈值
    # torch.testing.assert_close(q_out, ref_out, rtol=rtol, atol=atol)

    # 检查absmax输出
    rtol = 0.1  # 相对误差阈值
    atol = 0.1  # 绝对误差阈值
    torch.testing.assert_close(q_out_absmax, ref_out, rtol=rtol, atol=atol)



def test_w8a8_o16_linear2():
    from int_kernel._CUDA import s8t_s8n_f16t_gemm
    
    x_test = torch.randint(-128, 127, (4, 256), dtype=torch.int8, device="cuda")
    w_test = torch.randint(-128, 127, (128, 256), dtype=torch.int8, device="cuda")

    print('torch output: ', torch.matmul(x_test.to(torch.float), w_test.T.to(torch.float))[0])

    output = s8t_s8n_f16t_gemm(x_test, w_test)
    # torch.cuda.synchronize()
    print('cuda output: ', output.dtype, output.shape, output[0])

if __name__ == '__main__':
    test_w8a8_o16_linear2()
    pass
