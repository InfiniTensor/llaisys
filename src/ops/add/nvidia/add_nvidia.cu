#include "add_nvidia.cuh"

#include "../../../utils.hpp"

#include "../../../utils/gpu_utils.hpp"

__global__ void add_f32_kernel(float *c, const float *a, const float *b, size_t n) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        float4 reg_a = LOAD_FLOAT4(a[idx]);
        float4 reg_b = LOAD_FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        STORE_FLOAT4(c[idx]) = reg_c;
    }
}

__global__ void add_f16_kernel(half *c, const half *a, const half *b, size_t n) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        half2 reg_a = LOAD_HALF2(a[idx]);
        half2 reg_b = LOAD_HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        STORE_HALF2(c[idx]) = reg_c;
    }
}

__global__ void add_bf16_kernel(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        __nv_bfloat162 reg_a = LOAD_BFLOAT2(a[idx]);
        __nv_bfloat162 reg_b = LOAD_BFLOAT2(b[idx]);
        __nv_bfloat162 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        STORE_BFLOAT2(c[idx]) = reg_c;
    }
}

void config_launch(dim3 &block, dim3 &grid, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        block = dim3(256);
        grid = dim3(CEIL(CEIL(numel,4), 256));
        break;
    case LLAISYS_DTYPE_F16:
        block = dim3(256);
        grid = dim3(CEIL(CEIL(numel,2), 256));
        break;
    case LLAISYS_DTYPE_BF16:
        block = dim3(256);
        grid = dim3(CEIL(CEIL(numel,2), 256));
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    if (numel == 0) {
        return;
    }

    dim3 block{0};
    dim3 grid{0};
    config_launch(block, grid, type, numel);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                                        reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_f16_kernel<<<grid, block>>>(reinterpret_cast<half *>(c),
                                        reinterpret_cast<const half *>(a),
                                        reinterpret_cast<const half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_bf16_kernel<<<grid, block>>>(reinterpret_cast<__nv_bfloat16 *>(c),
                                         reinterpret_cast<const __nv_bfloat16 *>(a),
                                         reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
