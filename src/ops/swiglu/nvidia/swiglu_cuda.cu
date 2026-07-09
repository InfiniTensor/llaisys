#include "swiglu_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU kernel：out = up * silu(gate) = up * (gate / (1 + exp(-gate)))
// 使用 __expf 快速单精度指数，A100 上比双精度快 ~4×
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void swiglu_kernel(T *__restrict__ out,
                              const T *__restrict__ gate,
                              const T *__restrict__ up,
                              size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float g_val, u_val;
    if constexpr (std::is_same_v<T, __half>) {
        g_val = __half2float(gate[idx]);
        u_val = __half2float(up[idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        g_val = __bfloat162float(gate[idx]);
        u_val = __bfloat162float(up[idx]);
    } else {
        g_val = static_cast<float>(gate[idx]);
        u_val = static_cast<float>(up[idx]);
    }

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_val = g_val / (1.0f + __expf(-g_val));
    float result = u_val * silu_val;

    if constexpr (std::is_same_v<T, __half>) {
        out[idx] = __float2half(result);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        out[idx] = __float2bfloat16(result);
    } else {
        out[idx] = static_cast<T>(result);
    }
}

// float4 向量化版本（f32，numel 为 4 的倍数时使用，减少 kernel launch overhead）
__global__ void swiglu_f32x4_kernel(float *__restrict__ out,
                                    const float *__restrict__ gate,
                                    const float *__restrict__ up,
                                    size_t n4) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n4) {
        return;
    }

    float4 g4 = reinterpret_cast<const float4 *>(gate)[idx];
    float4 u4 = reinterpret_cast<const float4 *>(up)[idx];
    float4 r4;
    r4.x = u4.x * (g4.x / (1.0f + __expf(-g4.x)));
    r4.y = u4.y * (g4.y / (1.0f + __expf(-g4.y)));
    r4.z = u4.z * (g4.z / (1.0f + __expf(-g4.z)));
    r4.w = u4.w * (g4.w / (1.0f + __expf(-g4.w)));
    reinterpret_cast<float4 *>(out)[idx] = r4;
}

namespace llaisys::ops::nvidia {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    constexpr int BLOCK = 256;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        size_t n4 = numel / 4;
        size_t tail = numel % 4;
        if (n4 > 0) {
            int grid = static_cast<int>((n4 + BLOCK - 1) / BLOCK);
            swiglu_f32x4_kernel<<<grid, BLOCK>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(gate),
                reinterpret_cast<const float *>(up),
                n4);
        }
        if (tail > 0) {
            int grid = static_cast<int>((tail + BLOCK - 1) / BLOCK);
            size_t offset = n4 * 4;
            swiglu_kernel<float><<<grid, BLOCK>>>(
                reinterpret_cast<float *>(out) + offset,
                reinterpret_cast<const float *>(gate) + offset,
                reinterpret_cast<const float *>(up) + offset,
                tail);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        int grid = static_cast<int>((numel + BLOCK - 1) / BLOCK);
        swiglu_kernel<__half><<<grid, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        int grid = static_cast<int>((numel + BLOCK - 1) / BLOCK);
        swiglu_kernel<__nv_bfloat16><<<grid, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            numel);
        break;
    }
    default:
        throw std::runtime_error("swiglu CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
