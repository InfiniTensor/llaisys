#include "add_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
__global__ void add_kernel_f32(float *c, const float *a, const float *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}

// --- F16 Kernel ---
__global__ void add_kernel_f16(void *c, const void *a, const void *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __half ha = reinterpret_cast<const __half*>(a)[idx];
        __half hb = reinterpret_cast<const __half*>(b)[idx];
        // 转换为 float 相加后再转回 half
        reinterpret_cast<__half*>(c)[idx] = __float2half(__half2float(ha) + __half2float(hb));
    }
}

// --- BF16 Kernel ---
__global__ void add_kernel_bf16(void *c, const void *a, const void *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
#if __CUDACC_VER_MAJOR__ >= 11
        __nv_bfloat16 ha = reinterpret_cast<const __nv_bfloat16*>(a)[idx];
        __nv_bfloat16 hb = reinterpret_cast<const __nv_bfloat16*>(b)[idx];
        reinterpret_cast<__nv_bfloat16*>(c)[idx] = __float2bfloat16(__bfloat162float(ha) + __bfloat162float(hb));
#endif
    }
}

// C++ 路由入口：配置线程并启动 Kernel
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    int threads_per_block = 256;
    int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel_f32<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float*>(c),
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(b),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel_f16<<<blocks_per_grid, threads_per_block>>>(c, a, b, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(c, a, b, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia