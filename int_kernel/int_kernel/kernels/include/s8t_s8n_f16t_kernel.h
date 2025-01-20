#pragma once

#include <torch/extension.h>

// 主接口：int8 GEMM计算
torch::Tensor s8t_s8n_f16t_gemm(
    torch::Tensor q_input,          // int8 [M, K]
    torch::Tensor q_weight         // int8 [N, K]
); 