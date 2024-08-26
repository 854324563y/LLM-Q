// Copyright (C) 2024 ByteDance and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//          http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_kernel.h"

struct AqBWMMAOpState {
    size_t shared_mem_size;
    dim3 gridDim;
    dim3 blockDim;
    bool initSuccess = false;
    struct Argument_t {
        int M, N, K;
        int *X;
        int *W;
        half *C;
        int *D;
        bool bias = false;
    } args;
};

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage>
class AqBWMMAOp {
    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
    static constexpr bool SIGNED = QuantType::SIGNED;
    using AccumulatorType = int32_t;
    using ASwizzle = SwizzleIdentity;
    using BSwizzle = SwizzleIdentity;
    using CSwizzle = SwizzleIdentity;
    // launch state
public:
    AqBWMMAOpState state;
    using KernelImpl =
        AqBWMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, AccumulatorType,
                      ASwizzle, BSwizzle, CSwizzle, true, true, false>;
    void initialize(int *X, int *W, int M, int N, int K, int *D, half *C, bool bias);
    void operator()(cudaStream_t stream = NULL);
};

// *** device kernel ***
template <typename KernelImpl>
__global__ void launchAqBWMMAKernel(typename AqBWMMAOpState::Argument_t args)
{
    extern __shared__ int shared_mem_workspace[];
    KernelImpl k;
    k.mainLoop(args.M, args.N, args.K, args.X, args.W, shared_mem_workspace);
    k.epilogue(args.M, args.N, args.D, shared_mem_workspace, args.C, args.bias);
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage>
void AqBWMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    int *X, int *W, int M, int N, int K, int *D, half *C, bool bias)
{
    assert(!bias && "Bias operation is not supported temporarily\n");
    // set argument
    this->state.args = AqBWMMAOpState::Argument_t({ M, N, K, X, W, C, D, bias });
    // compute shared memory buffer size
    size_t input_buffer_size_dyn = 0;
    size_t input_buffer_size = input_buffer_size_dyn + KernelImpl::input_buffer_size_static;
    size_t output_buffer_size_dyn = 0;
    size_t output_buffer_size = output_buffer_size_dyn + KernelImpl::output_buffer_size_static;
    this->state.shared_mem_size = max(input_buffer_size, output_buffer_size);
    // printf("\ninput_buffer_size:%d\n", input_buffer_size);
    // printf("output_buffer_size_dyn:%d\n", output_buffer_size_dyn);
    // printf("output_buffer_size:%d\n", KernelImpl::output_buffer_size_static);
    // printf("shared_mem_size:%d\n", this->state.shared_mem_size);
    if (this->state.shared_mem_size >= 32 * 1024) {
        // set kernel attribute
        if (cudaSuccess != cudaFuncSetAttribute(launchAqBWMMAKernel<KernelImpl>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                this->state.shared_mem_size) ||
            cudaSuccess != cudaFuncSetAttribute(launchAqBWMMAKernel<KernelImpl>,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                100)) {
            int max_shared_mem;
            cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err) << std::endl;
            std::cerr
                << "Kernel required " << this->state.shared_mem_size
                << " shared memory but the max shared memory per block optin is: " << max_shared_mem
                << std::endl;
            this->state.initSuccess = false;
        }
    }
    // printf("dyn shared_mem_size:%d\n", this->state.shared_mem_size);
    // calculate launch configuration
    int gdimX = KernelImpl::GridMapping ? (CEIL(M, KernelImpl::BLOCK_M)) :
                                          (CEIL(N, KernelImpl::BLOCK_N));
    int gdimY = KernelImpl::GridMapping ? (CEIL(N, KernelImpl::BLOCK_N)) :
                                          (CEIL(M, KernelImpl::BLOCK_M));
    this->state.gridDim = dim3(gdimX, gdimY, 1);
    this->state.blockDim = dim3(KernelImpl::blockDims, 1, 1);
    // printf("gdimX:%d gdimY:%d, KernelImpl::blockDim:%d\n", gdimX, gdimY, KernelImpl::blockDims);
    this->state.initSuccess = true;
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage>
void AqBWMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage>::operator()(
    cudaStream_t stream)
{
    launchAqBWMMAKernel<KernelImpl>
        <<<this->state.gridDim, this->state.blockDim, this->state.shared_mem_size, stream>>>(
            this->state.args);
}

// pure-function version of the original c++-object Op
// function handle easy for benchmarking, testing
template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage>
AqBWMMAOpState AqBWMMAInitFn(int *X, int *W, int M, int N, int K, int *D, half *C, bool bias)
{
    AqBWMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage> op;
    op.initialize(X, W, M, N, K, D, C, bias);
    return op.state;
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage>
void AqBWMMAExecFn(AqBWMMAOpState &state, cudaStream_t stream = NULL)
{
    using KernelImpl =
        typename AqBWMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage>::KernelImpl;
    launchAqBWMMAKernel<KernelImpl>
        <<<state.gridDim, state.blockDim, state.shared_mem_size, stream>>>(state.args);
}

typedef AqBWMMAOpState (*AqBWMMAInitFn_t)(int *, int *, int, int, int, int *, half *, bool);
typedef void (*AqBWMMAExecFn_t)(AqBWMMAOpState &, cudaStream_t);
