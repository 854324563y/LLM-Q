#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

namespace utils {
    __device__ __forceinline__ void pack_int4(const int8_t* input, uint8_t* output) {
        uint8_t low = static_cast<uint8_t>(input[0] & 0xF);
        uint8_t high = static_cast<uint8_t>(input[1] & 0xF);
        *output = (high << 4) | low;
    }

    __device__ __forceinline__ void unpack_int4(const uint8_t* input, int8_t* output) {
        output[0] = static_cast<int8_t>(*input & 0xF);
        output[1] = static_cast<int8_t>((*input >> 4) & 0xF);
    }

    template<typename T>
    __device__ __forceinline__ T clamp(float val, float min_val, float max_val) {
        val = min(max(val, min_val), max_val);
        return static_cast<T>(round(val));
    }
}

// 改进的量化kernel
template<typename T_IN, typename T_OUT>
static __global__ void quantize_tensor_kernel(
    const T_IN* input,
    T_OUT* output,
    const float* scale,
    const float* zero_point,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input[idx]);
        val = val / scale[0] + zero_point[0];
        output[idx] = utils::clamp<T_OUT>(
            val,
            static_cast<float>(std::numeric_limits<T_OUT>::min()),
            static_cast<float>(std::numeric_limits<T_OUT>::max())
        );
    }
}

namespace cuda_kernels {
    static __global__ void quantize_tensor_int4_kernel(
        const at::Half* input,
        uint8_t* output,
        const float* scale,
        const float* zero_point,
        int size) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size/2) {  // 每个线程处理2个值
            float val1 = __half2float(input[idx*2]);
            float val2 = __half2float(input[idx*2 + 1]);
            
            val1 = val1 / scale[0] + zero_point[0];
            val2 = val2 / scale[0] + zero_point[0];
            
            int8_t q_vals[2];
            q_vals[0] = utils::clamp<int8_t>(val1, -8.0f, 7.0f);  // 4-bit范围
            q_vals[1] = utils::clamp<int8_t>(val2, -8.0f, 7.0f);
            
            utils::pack_int4(q_vals, &output[idx]);
        }
    }

    static __global__ void dequantize_tensor_int4_kernel(
        const uint8_t* input,
        at::Half* output,
        const float* scale,
        const float* zero_point,
        int size) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size/2) {  // 每个线程处理2个值
            int8_t q_vals[2];
            utils::unpack_int4(&input[idx], q_vals);
            
            float val1 = static_cast<float>(q_vals[0]);
            float val2 = static_cast<float>(q_vals[1]);
            
            val1 = (val1 - zero_point[0]) * scale[0];
            val2 = (val2 - zero_point[0]) * scale[0];
            
            output[idx*2] = __float2half(val1);
            output[idx*2 + 1] = __float2half(val2);
        }
    }
}

template<typename T_IN>
static __global__ void dequantize_tensor_kernel(
    const T_IN* input,
    at::Half* output,
    const float* scale,
    const float* zero_point,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input[idx]);
        val = (val - zero_point[0]) * scale[0];
        output[idx] = __float2half(val);
    }
}

#endif // CUDA_KERNELS_CUH 