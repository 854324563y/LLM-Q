#ifndef INT_KERNEL_H
#define INT_KERNEL_H

#include <torch/extension.h>

// 主要接口声明
extern "C" {
    torch::Tensor w8a8_gemm_with_quant(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor input_scale,
        torch::Tensor input_zero_point,
        torch::Tensor weight_scale,
        torch::Tensor weight_zero_point,
        torch::Tensor output_scale,
        torch::Tensor output_zero_point);

    torch::Tensor w4a8_gemm_with_quant(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor input_scale,
        torch::Tensor input_zero_point,
        torch::Tensor weight_scale,
        torch::Tensor weight_zero_point,
        torch::Tensor output_scale,
        torch::Tensor output_zero_point);

    torch::Tensor w4a4_gemm_with_quant(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor input_scale,
        torch::Tensor input_zero_point,
        torch::Tensor weight_scale,
        torch::Tensor weight_zero_point,
        torch::Tensor output_scale,
        torch::Tensor output_zero_point);
}

#endif // INT_KERNEL_H