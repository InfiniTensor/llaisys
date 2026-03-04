#include "rope_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

namespace llaisys::ops::nvidia {

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<half>(half v) { return __half2float(v); }
template<> __device__ inline float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline half from_float<half>(float v) { return __float2half(v); }
// 修复拼写错误: nv_bloat16 -> nv_bfloat16
template<> __device__ inline nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void rope_kernel(
    T *out, const T *in, const int64_t *pos_ids,
    size_t seq_len, size_t n_head, size_t head_dim, float theta,
    size_t in_s0, size_t in_s1, size_t in_s2,
    size_t out_s0, size_t out_s1, size_t out_s2,
    size_t pos_s0
) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_head * half_dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t l = tid % half_dim;
    size_t h = (tid / half_dim) % n_head;
    size_t s = tid / (half_dim * n_head);

    float pos = (float)pos_ids[s * pos_s0];
    float exponent = 2.0f * (float)l / (float)head_dim;
    float angle = pos / powf(theta, exponent);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    size_t idx_a = s * in_s0 + h * in_s1 + l * in_s2;
    size_t idx_b = s * in_s0 + h * in_s1 + (l + half_dim) * in_s2;

    float val_a = to_float(in[idx_a]);
    float val_b = to_float(in[idx_b]);

    size_t out_idx_a = s * out_s0 + h * out_s1 + l * out_s2;
    size_t out_idx_b = s * out_s0 + h * out_s1 + (l + half_dim) * out_s2;

    out[out_idx_a] = from_float<T>(val_a * cos_val - val_b * sin_val);
    out[out_idx_b] = from_float<T>(val_b * cos_val + val_a * sin_val);
}

template<typename T>
void launch_rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
                 size_t seq_len, size_t n_head, size_t head_dim, float theta,
                 size_t in_s0, size_t in_s1, size_t in_s2,
                 size_t out_s0, size_t out_s1, size_t out_s2,
                 size_t pos_s0) {
    T* d_out = reinterpret_cast<T*>(out);
    const T* d_in = reinterpret_cast<const T*>(in);

    size_t total = seq_len * n_head * (head_dim / 2);
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    rope_kernel<T><<<blocks, threads>>>(
        d_out, d_in, pos_ids,
        seq_len, n_head, head_dim, theta,
        in_s0, in_s1, in_s2,
        out_s0, out_s1, out_s2,
        pos_s0
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA RoPE Kernel Launch failed: %s\n", cudaGetErrorString(err));
    }
}

// 修改函数签名以匹配新的 .hpp
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_head, size_t head_dim, float theta) {
    
    size_t in_s0 = n_head * head_dim;
    size_t in_s1 = head_dim;
    size_t in_s2 = 1;

    size_t out_s0 = n_head * head_dim;
    size_t out_s1 = head_dim;
    size_t out_s2 = 1;

    size_t pos_s0 = 1;

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_rope<float>(out, in, pos_ids, seq_len, n_head, head_dim, theta,
                               in_s0, in_s1, in_s2, out_s0, out_s1, out_s2, pos_s0);
            break;
        case LLAISYS_DTYPE_F16:
            launch_rope<half>(out, in, pos_ids, seq_len, n_head, head_dim, theta,
                              in_s0, in_s1, in_s2, out_s0, out_s1, out_s2, pos_s0);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_rope<nv_bfloat16>(out, in, pos_ids, seq_len, n_head, head_dim, theta,
                                     in_s0, in_s1, in_s2, out_s0, out_s1, out_s2, pos_s0);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA RoPE: %d\n", dtype);
            break;
    }
}

} // namespace llaisys::ops::nvidia