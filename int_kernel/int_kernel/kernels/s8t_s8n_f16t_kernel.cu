#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <torch/extension.h>

#include "include/s8t_s8n_f16t_kernel.h"
#include "include/cuda_kernels.cuh"

// 主接口：int8 GEMM计算
torch::Tensor s8t_s8n_f16t_gemm(
    torch::Tensor q_input,          // int8 [M, K]
    torch::Tensor q_weight         // int8 [N, K]
) {
    TORCH_CHECK(q_input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(q_weight.dim() == 2, "Weight must be 2D tensor");
    TORCH_CHECK(q_input.size(1) == q_weight.size(1), "Input and weight dimensions mismatch");
    
    const int M = q_input.size(0);
    const int K = q_input.size(1);
    const int N = q_weight.size(0);
    
    // 配置GEMM
    using ElementOutput = float;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>;
    
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;
    
    // 创建输出tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(q_input.device()));
    
    // 设置GEMM参数
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    typename Gemm::Arguments arguments{
        problem_size,
        {q_input.data_ptr<int8_t>(), K},
        {q_weight.data_ptr<int8_t>(), K},
        {output.data_ptr<float>(), N},
        {output.data_ptr<float>(), N},
        {1.0f, 0.0f},
        1
    };
    
    // 初始化并运行GEMM
    Gemm gemm_op;
    auto status = gemm_op.initialize(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");
    
    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
    
    return output;
} 