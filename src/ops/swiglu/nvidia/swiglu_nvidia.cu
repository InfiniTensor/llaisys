#include "swiglu_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// 设备端的 silu 激活函数实现
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// --- F32 Kernel ---
__global__ void swiglu_kernel_f32(float *c, const float *a, const float *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = silu(a[idx]) * b[idx];
    }
}

// --- F16 Kernel ---
__global__ void swiglu_kernel_f16(void *c, const void *a, const void *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float fa = __half2float(reinterpret_cast<const __half*>(a)[idx]);
        float fb = __half2float(reinterpret_cast<const __half*>(b)[idx]);
        reinterpret_cast<__half*>(c)[idx] = __float2half(silu(fa) * fb);
    }
}

// --- BF16 Kernel ---
__global__ void swiglu_kernel_bf16(void *c, const void *a, const void *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
#if __CUDACC_VER_MAJOR__ >= 11
        float fa = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(a)[idx]);
        float fb = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(b)[idx]);
        reinterpret_cast<__nv_bfloat16*>(c)[idx] = __float2bfloat16(silu(fa) * fb);
#endif
    }
}

void swiglu(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    int threads_per_block = 256;
    int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel_f32<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float*>(c),
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(b),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel_f16<<<blocks_per_grid, threads_per_block>>>(c, a, b, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(c, a, b, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia