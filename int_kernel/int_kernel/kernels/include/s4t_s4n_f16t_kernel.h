#pragma once

#include <torch/extension.h>

// 主接口：int4权重-int4激活 GEMM计算
torch::Tensor s4t_s4n_f16t_gemm(
    torch::Tensor q_input,          // int4 [M, K]
    torch::Tensor q_weight         // int4 [N, K]
); 