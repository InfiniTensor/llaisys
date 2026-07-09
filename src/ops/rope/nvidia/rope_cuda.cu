#include "rope_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// RoPE kernel
// grid:  [seq_len, n_heads, half_dim/BLOCK_DIM] — 三维网格充分并行
// block: BLOCK_DIM threads 处理同一 (s, h) 的维度对
//
// 对每个 (s, h, j)：
//   freq = pos_ids[s] / theta^(2j/d)
//   a' = a*cos(freq) - b*sin(freq)
//   b' = b*cos(freq) + a*sin(freq)
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void rope_kernel(T *__restrict__ out,
                            const T *__restrict__ in,
                            const int64_t *__restrict__ pos_ids,
                            size_t n_heads,
                            size_t head_dim,
                            float theta) {
    size_t s = blockIdx.x;                            // 序列位置
    size_t h = blockIdx.y;                            // attention head
    size_t j = blockIdx.z * blockDim.x + threadIdx.x; // 维度对索引

    size_t half_dim = head_dim / 2;
    if (j >= half_dim) {
        return;
    }

    int64_t pos = pos_ids[s];

    // freq = pos / theta^(2j/d)
    float freq_exp = (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
    float freq = static_cast<float>(pos) / powf(theta, freq_exp);
    float cos_freq, sin_freq;
    sincosf(freq, &sin_freq, &cos_freq);

    size_t a_idx = s * n_heads * head_dim + h * head_dim + j;
    size_t b_idx = a_idx + half_dim;

    float a_val, b_val;
    if constexpr (std::is_same_v<T, __half>) {
        a_val = __half2float(in[a_idx]);
        b_val = __half2float(in[b_idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        a_val = __bfloat162float(in[a_idx]);
        b_val = __bfloat162float(in[b_idx]);
    } else {
        a_val = static_cast<float>(in[a_idx]);
        b_val = static_cast<float>(in[b_idx]);
    }

    float a_prime = a_val * cos_freq - b_val * sin_freq;
    float b_prime = b_val * cos_freq + a_val * sin_freq;

    if constexpr (std::is_same_v<T, __half>) {
        out[a_idx] = __float2half(a_prime);
        out[b_idx] = __float2half(b_prime);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        out[a_idx] = __float2bfloat16(a_prime);
        out[b_idx] = __float2bfloat16(b_prime);
    } else {
        out[a_idx] = static_cast<T>(a_prime);
        out[b_idx] = static_cast<T>(b_prime);
    }
}

namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads,
          size_t head_dim, float theta) {
    if (seq_len == 0 || n_heads == 0 || head_dim == 0) {
        return;
    }

    size_t half_dim = head_dim / 2;
    const int BLOCK = 32; // 每个 block 处理 32 个维度对（一个 warp）
    unsigned z_dim = static_cast<unsigned>((half_dim + BLOCK - 1) / BLOCK);

    dim3 grid(static_cast<unsigned>(seq_len),
              static_cast<unsigned>(n_heads),
              z_dim);
    dim3 blk(static_cast<unsigned>(BLOCK));

    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, blk>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ptr, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<grid, blk>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            pos_ptr, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<grid, blk>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            pos_ptr, n_heads, head_dim, theta);
        break;
    default:
        throw std::runtime_error("rope CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
