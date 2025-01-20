#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <torch/extension.h>

#include "include/s8t_s4n_f16t_kernel.h"
#include "include/cuda_kernels.cuh"

// 主接口：int8激活-int4权重 GEMM计算
torch::Tensor s8t_s4n_f16t_gemm(
    torch::Tensor q_input,          // int8 [M, K]
    torch::Tensor q_weight         // int4 [N, K/2]
) {
    TORCH_CHECK(q_input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(q_weight.dim() == 2, "Weight must be 2D tensor");
    // TORCH_CHECK(q_input.size(1) == q_weight.size(1) * 2, "Input and weight dimensions mismatch");
    
    const int M = q_input.size(0);
    const int K = q_input.size(1);
    const int N = q_weight.size(0);
    
    // 配置GEMM
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;
    using ElementInputA = int8_t;
    // 注意CUDA内核在处理权重解包时，会优先处理低4位，再处理高4位
    using ElementInputB = cutlass::int4b_t;  // 使用4位整数类型
    
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>;
    
    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,    // Stages
        16,   // AlignmentA // 128/cutlass::sizeof_bits<ElementA>::value
        32,   // AlignmentB // 128/cutlass::sizeof_bits<ElementB>::value
        cutlass::arch::OpMultiplyAddMixedInputUpcast,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone
    >;
    
    // 创建输出tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kInt32).device(q_input.device()));
    
    // 设置GEMM参数
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    int batch_count = 1;
    ElementCompute alpha = ElementCompute(1);
    ElementCompute beta = ElementCompute(0);
    
    typename Gemm::Arguments arguments{
        mode,                                   // mode
        problem_size,                          // problem size
        batch_count,                           // batch count
        {alpha, beta},                         // alpha, beta
        q_input.data_ptr<int8_t>(),           // ptr A
        reinterpret_cast<const cutlass::int4b_t*>(q_weight.data_ptr<uint8_t>()),// ptr B
        output.data_ptr<int32_t>(),           // ptr C
        output.data_ptr<int32_t>(),           // ptr D
        M * K,                                // batch stride A
        N * K,                                // batch stride B
        M * N,                                // batch stride C
        M * N,                                // batch stride D
        K,                                    // stride A
        K,                                    // stride B
        N,                                    // stride C
        N                                     // stride D
    };
    
    // 初始化并运行GEMM
    Gemm gemm_op;
    auto status = gemm_op.initialize(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");
    
    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
    
    return output;
} 