#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <torch/extension.h>

#include "include/s4t_s4n_f16t_kernel.h"
#include "include/cuda_kernels.cuh"

// 主接口：int4权重-int4激活 GEMM计算
torch::Tensor s4t_s4n_f16t_gemm(
    torch::Tensor q_input,          // int4 [M, K]
    torch::Tensor q_weight         // int4 [N, K]
) {
    TORCH_CHECK(q_input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(q_weight.dim() == 2, "Weight must be 2D tensor");
    TORCH_CHECK(q_input.size(1) == q_weight.size(1), "Input and weight dimensions mismatch");
    
    const int M = q_input.size(0);
    const int K = q_input.size(1) * 2;
    const int N = q_weight.size(0);
    
    // 配置GEMM
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
        cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 256>,
        cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

    // 创建输出tensor
    // auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(q_input.device()));
    auto output = torch::empty({M, N}, torch::dtype(torch::kInt32).device(q_input.device()));

    // 设置GEMM参数
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    typename Gemm::Arguments arguments{
        problem_size,
        {reinterpret_cast<const cutlass::int4b_t*>(q_input.data_ptr<uint8_t>()), K},   // 使用 uint8_t 作为中间类型
        // 有点疑问为什么这里weight的stride是K，而不是K/2
        {reinterpret_cast<const cutlass::int4b_t*>(q_weight.data_ptr<uint8_t>()), K},  // 使用 uint8_t 作为中间类型
        {output.data_ptr<int32_t>(), N},
        {output.data_ptr<int32_t>(), N},
        {1, 0},
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
