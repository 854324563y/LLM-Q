#pragma once

#include <torch/extension.h>

// 主接口：int8激活-int4权重 GEMM计算
torch::Tensor s8t_s4n_f16t_gemm(
    torch::Tensor q_input,          // int8 [M, K]
    torch::Tensor q_weight         // int4 [N, K]
); 