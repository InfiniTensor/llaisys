#include "add_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// F32：使用 float4 宽加载（A100 128-bit 访存对齐）
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_f32_kernel(float *__restrict__ c,
                               const float *__restrict__ a,
                               const float *__restrict__ b,
                               size_t n4, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = reinterpret_cast<const float4 *>(a)[idx];
        float4 vb = reinterpret_cast<const float4 *>(b)[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4 *>(c)[idx] = vc;
    }
    // 处理尾部未对齐元素
    size_t tail_start = n4 * 4;
    if (idx == 0) {
        for (size_t i = tail_start; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// F16：使用 __half2 一次处理两个 fp16 元素
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_f16_kernel(__half *__restrict__ c,
                               const __half *__restrict__ a,
                               const __half *__restrict__ b,
                               size_t n2, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 va = reinterpret_cast<const __half2 *>(a)[idx];
        __half2 vb = reinterpret_cast<const __half2 *>(b)[idx];
        reinterpret_cast<__half2 *>(c)[idx] = __hadd2(va, vb);
    }
    if (idx == 0 && n % 2 != 0) {
        c[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BF16：使用 __nv_bfloat162 一次处理两个 bf16 元素
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_bf16_kernel(__nv_bfloat16 *__restrict__ c,
                                const __nv_bfloat16 *__restrict__ a,
                                const __nv_bfloat16 *__restrict__ b,
                                size_t n2, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __nv_bfloat162 va = reinterpret_cast<const __nv_bfloat162 *>(a)[idx];
        __nv_bfloat162 vb = reinterpret_cast<const __nv_bfloat162 *>(b)[idx];
        reinterpret_cast<__nv_bfloat162 *>(c)[idx] = __hadd2(va, vb);
    }
    if (idx == 0 && n % 2 != 0) {
        c[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

namespace llaisys::ops::nvidia {

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    constexpr int BLOCK = 256;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        size_t n4 = numel / 4;
        int grid = static_cast<int>((n4 + BLOCK - 1) / BLOCK);
        if (grid == 0) {
            grid = 1;
        }
        add_f32_kernel<<<grid, BLOCK>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            n4, numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        size_t n2 = numel / 2;
        int grid = static_cast<int>((n2 + BLOCK - 1) / BLOCK);
        if (grid == 0) {
            grid = 1;
        }
        add_f16_kernel<<<grid, BLOCK>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            n2, numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        size_t n2 = numel / 2;
        int grid = static_cast<int>((n2 + BLOCK - 1) / BLOCK);
        if (grid == 0) {
            grid = 1;
        }
        add_bf16_kernel<<<grid, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            n2, numel);
        break;
    }
    default:
        throw std::runtime_error("add CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
