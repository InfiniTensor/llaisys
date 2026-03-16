/**
 * NVIDIA CUDA 算子实现：add, embedding, linear, argmax, rms_norm, rope, swiglu, self_attention.
 * 与 include/llaisys/ops_nvidia.h 声明对应；op.cpp 在 LLAISYS_DEVICE_NVIDIA 时调用此处。
 */
#ifdef ENABLE_NVIDIA_API

#include "llaisys/ops_nvidia.h"
#include "utils/types.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace llaisys::ops::nvidia {

// ---- 辅助：bf16/fp16 与 float 互转（device） ----
__device__ __forceinline__ float bf16_to_float(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    return __uint_as_float(u);
}
__device__ __forceinline__ uint16_t float_to_bf16(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t bf16_bits = u >> 16;
    uint32_t remainder = u & 0xFFFFu;
    if (remainder > 0x8000u) bf16_bits++;
    else if (remainder == 0x8000u && (bf16_bits & 1u)) bf16_bits++;
    return (uint16_t)bf16_bits;
}
__device__ __forceinline__ float half_to_float(uint16_t v) {
    uint32_t sign = (v & 0x8000u) << 16;
    uint32_t rest = v & 0x7fffu;
    if (rest >= 0x7c00u) rest = (rest == 0x7c00u) ? 0x7c00u : 0x7e00u;
    uint32_t exp = (rest >= 0x400u) ? ((((rest >> 10) - 15) + 127) << 23) : 0;
    uint32_t mant = (rest & 0x3ffu) << 13;
    return __uint_as_float(sign | exp | mant);
}
__device__ __forceinline__ uint16_t float_to_half(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xff) - 127;
    uint32_t mant = u & 0x7fffffu;
    if (exp >= 15) return sign | 0x7c00u;
    if (exp < -14) return sign;
    uint32_t new_exp = (uint32_t)(exp + 15) << 10;
    return sign | new_exp | (mant >> 13);
}

// ---- Add ----
template<typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_kernel_bf16(uint16_t* c, const uint16_t* a, const uint16_t* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = bf16_to_float(a[i]);
    float fb = bf16_to_float(b[i]);
    c[i] = float_to_bf16(fa + fb);
}

__global__ void add_kernel_f16(uint16_t* c, const uint16_t* a, const uint16_t* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = half_to_float(a[i]);
    float fb = half_to_float(b[i]);
    c[i] = float_to_half(fa + fb);
}

void add(std::byte* c, const std::byte* a, const std::byte* b, llaisysDataType_t dtype, size_t numel) {
    const size_t block = 256;
    size_t grid = (numel + block - 1) / block;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<grid, block>>>((float*)c, (const float*)a, (const float*)b, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16<<<grid, block>>>((uint16_t*)c, (const uint16_t*)a, (const uint16_t*)b, numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel_f16<<<grid, block>>>((uint16_t*)c, (const uint16_t*)a, (const uint16_t*)b, numel);
        break;
    default:
        break;
    }
}

// ---- Embedding (gather rows) ----
__global__ void embedding_kernel_f32(float* out, const float* weight, const int64_t* index, size_t num_index, size_t embed_dim, size_t vocab_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_index * embed_dim) return;
    size_t row = i / embed_dim;
    size_t col = i % embed_dim;
    int64_t idx = index[row];
    if (idx < 0 || (size_t)idx >= vocab_size) { out[i] = 0.f; return; }
    out[i] = weight[(size_t)idx * embed_dim + col];
}

__global__ void embedding_kernel_bf16(uint16_t* out, const uint16_t* weight, const int64_t* index, size_t num_index, size_t embed_dim, size_t vocab_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_index * embed_dim) return;
    size_t row = i / embed_dim;
    size_t col = i % embed_dim;
    int64_t idx = index[row];
    if (idx < 0 || (size_t)idx >= vocab_size) { out[i] = 0; return; }
    out[i] = weight[(size_t)idx * embed_dim + col];
}

void embedding(std::byte* out, const std::byte* weight, const int64_t* index, size_t num_index, size_t embed_dim, size_t vocab_size, size_t elem_size) {
    size_t n = num_index * embed_dim;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    if (elem_size == 4)
        embedding_kernel_f32<<<grid, block>>>((float*)out, (const float*)weight, index, num_index, embed_dim, vocab_size);
    else
        embedding_kernel_bf16<<<grid, block>>>((uint16_t*)out, (const uint16_t*)weight, index, num_index, embed_dim, vocab_size);
}

// ---- Linear: out(B,M) = in(B,K) * weight(M,K)^T + bias ----
// 使用 2D grid (gridM, gridB)，避免大 1D grid 在部分环境下的问题；i=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x
__global__ void linear_kernel_f32(float* out, const float* in, const float* weight, const float* bias, size_t B, size_t M, size_t K) {
    size_t i = (size_t)blockIdx.y;
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B || j >= M) return;
    float sum = 0.f;
    for (size_t k = 0; k < K; k++)
        sum += in[i * K + k] * weight[j * K + k];
    if (bias) sum += bias[j];
    out[i * M + j] = sum;
}

__global__ void linear_kernel_bf16(uint16_t* out, const uint16_t* in, const uint16_t* weight, const uint16_t* bias, size_t B, size_t M, size_t K) {
    size_t i = (size_t)blockIdx.y;
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B || j >= M) return;
    float sum = 0.f;
    for (size_t k = 0; k < K; k++)
        sum += bf16_to_float(in[i * K + k]) * bf16_to_float(weight[j * K + k]);
    if (bias) sum += bf16_to_float(bias[j]);
    out[i * M + j] = float_to_bf16(sum);
}

__global__ void linear_kernel_f16(uint16_t* out, const uint16_t* in, const uint16_t* weight, const uint16_t* bias, size_t B, size_t M, size_t K) {
    size_t i = (size_t)blockIdx.y;
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B || j >= M) return;
    float sum = 0.f;
    for (size_t k = 0; k < K; k++)
        sum += half_to_float(in[i * K + k]) * half_to_float(weight[j * K + k]);
    if (bias) sum += half_to_float(bias[j]);
    out[i * M + j] = float_to_half(sum);
}

void linear(std::byte* out, const std::byte* in, const std::byte* weight, const std::byte* bias, llaisysDataType_t dtype, size_t B, size_t M, size_t K) {
    const unsigned int block = 256;
    const unsigned int gridM = (unsigned int)((M + block - 1) / block);
    const unsigned int gridB = (unsigned int)B;
    const dim3 grid(gridM, gridB);
    const dim3 dimBlock(block);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_kernel_f32<<<grid, dimBlock>>>((float*)out, (const float*)in, (const float*)weight, bias ? (const float*)bias : nullptr, B, M, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel_bf16<<<grid, dimBlock>>>((uint16_t*)out, (const uint16_t*)in, (const uint16_t*)weight, bias ? (const uint16_t*)bias : nullptr, B, M, K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel_f16<<<grid, dimBlock>>>((uint16_t*)out, (const uint16_t*)in, (const uint16_t*)weight, bias ? (const uint16_t*)bias : nullptr, B, M, K);
        break;
    default:
        break;
    }
}

// ---- Argmax (single output) ----
// 单 block 内归约：只处理前 blockDim.x 个元素，供小 n 或第二段使用
__device__ void argmax_block_reduce_f32(float* sh_max, size_t* sh_idx, size_t tid, size_t blockDim_x) {
    __syncthreads();
    for (int s = blockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s && sh_max[tid] < sh_max[tid + s]) {
            sh_max[tid] = sh_max[tid + s];
            sh_idx[tid] = sh_idx[tid + s];
        }
        __syncthreads();
    }
}

__global__ void argmax_kernel_f32(int64_t* max_idx, float* max_val, const float* vals, size_t n) {
    __shared__ float sh_max[256];
    __shared__ size_t sh_idx[256];
    size_t tid = threadIdx.x;
    const size_t block = blockDim.x;
    // 单 block 时必须覆盖全数组：每个线程循环处理 tid, tid+block, tid+2*block, ...
    float my_val = -1e38f;
    size_t my_idx = 0;
    for (size_t i = tid; i < n; i += block) {
        float v = vals[i];
        if (v > my_val) { my_val = v; my_idx = i; }
    }
    sh_max[tid] = my_val;
    sh_idx[tid] = my_idx;
    argmax_block_reduce_f32(sh_max, sh_idx, tid, block);
    if (tid == 0) {
        *max_val = sh_max[0];
        *max_idx = (int64_t)sh_idx[0];
    }
}

__global__ void argmax_kernel_bf16(int64_t* max_idx, uint16_t* max_val, const uint16_t* vals, size_t n) {
    __shared__ float sh_max[256];
    __shared__ size_t sh_idx[256];
    size_t tid = threadIdx.x;
    const size_t block = blockDim.x;
    float my_val = -1e38f;
    size_t my_idx = 0;
    for (size_t i = tid; i < n; i += block) {
        float v = bf16_to_float(vals[i]);
        if (v > my_val) { my_val = v; my_idx = i; }
    }
    sh_max[tid] = my_val;
    sh_idx[tid] = my_idx;
    argmax_block_reduce_f32(sh_max, sh_idx, tid, block);
    if (tid == 0) {
        *max_val = float_to_bf16(sh_max[0]);
        *max_idx = (int64_t)sh_idx[0];
    }
}

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t dtype, size_t numel) {
    if (numel == 0) return;
    const size_t block = 256;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel_f32<<<1, block>>>((int64_t*)max_idx, (float*)max_val, (const float*)vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
    case LLAISYS_DTYPE_F16:
        argmax_kernel_bf16<<<1, block>>>((int64_t*)max_idx, (uint16_t*)max_val, (const uint16_t*)vals, numel);
        break;
    default:
        break;
    }
}

// ---- RMSNorm: out = weight * in / sqrt(mean(in^2) + eps) per row ----
__global__ void rms_norm_kernel_f32(float* out, const float* in, const float* weight, size_t rows, size_t d, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * d;
    float* y = out + row * d;
    __shared__ float sh_sum;
    float sum_sq = 0.f;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x)
        sum_sq += x[i] * x[i];
    __shared__ float sh_sums[256];
    sh_sums[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh_sums[threadIdx.x] += sh_sums[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sh_sum = sqrtf((1.f / (float)d) * sh_sums[0] + eps);
    __syncthreads();
    float rms = sh_sum;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x)
        y[i] = weight[i] * x[i] / rms;
}

__global__ void rms_norm_kernel_bf16(uint16_t* out, const uint16_t* in, const uint16_t* weight, size_t rows, size_t d, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;
    const uint16_t* x = in + row * d;
    uint16_t* y = out + row * d;
    float sum_sq = 0.f;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float v = bf16_to_float(x[i]);
        sum_sq += v * v;
    }
    __shared__ float sh_sums[256];
    sh_sums[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh_sums[threadIdx.x] += sh_sums[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf((1.f / (float)d) * sh_sums[0] + eps);
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float v = bf16_to_float(x[i]) * bf16_to_float(weight[i]) / rms;
        y[i] = float_to_bf16(v);
    }
}

void rms_norm(std::byte* out, const std::byte* in, const std::byte* weight, llaisysDataType_t dtype, size_t rows, size_t dim, float eps) {
    size_t block = (dim < 256) ? (dim > 0 ? (unsigned)dim : 256) : 256;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel_f32<<<rows, block>>>((float*)out, (const float*)in, (const float*)weight, rows, dim, eps);
        break;
    case LLAISYS_DTYPE_BF16:
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel_bf16<<<rows, block>>>((uint16_t*)out, (const uint16_t*)in, (const uint16_t*)weight, rows, dim, eps);
        break;
    default:
        break;
    }
}

// ---- RoPE ----
__global__ void rope_kernel_f32(float* out, const float* in, const int64_t* pos_ids, size_t seq_len, size_t n_head, size_t head_dim, float theta) {
    size_t half_d = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_head * head_dim;
    if (idx >= total) return;
    size_t s = idx / (n_head * head_dim);
    size_t j = idx % head_dim;
    if (j >= half_d) return;
    double inv_freq = 1.0 / pow((double)theta, 2.0 * (double)j / (double)head_dim);
    double phi = (double)pos_ids[s] * inv_freq;
    float cos_phi = (float)cos(phi);
    float sin_phi = (float)sin(phi);
    float a = in[idx];
    float b = in[idx + half_d];
    out[idx] = a * cos_phi - b * sin_phi;
    out[idx + half_d] = b * cos_phi + a * sin_phi;
}

__global__ void rope_kernel_bf16(uint16_t* out, const uint16_t* in, const int64_t* pos_ids, size_t seq_len, size_t n_head, size_t head_dim, float theta) {
    size_t half_d = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_head * head_dim;
    if (idx >= total) return;
    size_t s = idx / (n_head * head_dim);
    size_t j = idx % head_dim;
    if (j >= half_d) return;
    double inv_freq = 1.0 / pow((double)theta, 2.0 * (double)j / (double)head_dim);
    double phi = (double)pos_ids[s] * inv_freq;
    float cos_phi = (float)cos(phi);
    float sin_phi = (float)sin(phi);
    float a = bf16_to_float(in[idx]);
    float b = bf16_to_float(in[idx + half_d]);
    out[idx] = float_to_bf16(a * cos_phi - b * sin_phi);
    out[idx + half_d] = float_to_bf16(b * cos_phi + a * sin_phi);
}

void rope(std::byte* out, const std::byte* in, const int64_t* pos_ids, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    size_t n = seq_len * num_heads * head_dim;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    if (dtype == LLAISYS_DTYPE_F32)
        rope_kernel_f32<<<grid, block>>>((float*)out, (const float*)in, pos_ids, seq_len, num_heads, head_dim, theta);
    else
        rope_kernel_bf16<<<grid, block>>>((uint16_t*)out, (const uint16_t*)in, pos_ids, seq_len, num_heads, head_dim, theta);
}

// ---- SwiGLU: out = up * silu(gate) ----
__device__ __forceinline__ float silu(float x) {
    if (x >= 0.f) return x / (1.f + expf(-x));
    float e = expf(x);
    return x * e / (1.f + e);
}

__global__ void swiglu_kernel_f32(float* out, const float* gate, const float* up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = up[i] * silu(gate[i]);
}

__global__ void swiglu_kernel_bf16(uint16_t* out, const uint16_t* gate, const uint16_t* up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float_to_bf16(bf16_to_float(up[i]) * silu(bf16_to_float(gate[i])));
}

void swiglu(std::byte* out, const std::byte* gate, const std::byte* up, llaisysDataType_t dtype, size_t numel) {
    size_t block = 256;
    size_t grid = (numel + block - 1) / block;
    if (dtype == LLAISYS_DTYPE_F32)
        swiglu_kernel_f32<<<grid, block>>>((float*)out, (const float*)gate, (const float*)up, numel);
    else
        swiglu_kernel_bf16<<<grid, block>>>((uint16_t*)out, (const uint16_t*)gate, (const uint16_t*)up, numel);
}

// ---- Self-attention: causal, single head at a time for simplicity ----
// shared: sh[0..kvlen-1] = scores, sh[kvlen..kvlen+blockDim.x-1] = per-thread max for reduction
__global__ void self_attention_kernel_f32(float* attn_val, const float* q, const float* k, const float* v, float scale, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    int causal_off = (int)kvlen - (int)qlen;
    size_t i = blockIdx.x;
    size_t h = blockIdx.y;
    if (i >= qlen || h >= nh) return;
    size_t kv_h = h * nkvh / nh;
    const float* q_row = q + (i * nh + h) * hd;
    const float* k_base = k + kv_h * hd;
    const float* v_base = v + kv_h * hd;

    extern __shared__ float sh[];
    float* sh_scores = sh;
    float max_s = -1e30f;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float dot = 0.f;
        for (size_t d = 0; d < hd; d++)
            dot += q_row[d] * k_base[j * nkvh * hd + d];
        float s = scale * dot;
        if ((int)j > (int)i + causal_off) s = -1e30f;
        sh_scores[j] = s;
        if (s > max_s) max_s = s;
    }
    __syncthreads();
    sh_scores[kvlen + threadIdx.x] = max_s;
    __syncthreads();
    for (int red = blockDim.x / 2; red > 0; red >>= 1) {
        if (threadIdx.x < red) {
            float other = sh_scores[kvlen + threadIdx.x + red];
            if (other > sh_scores[kvlen + threadIdx.x]) sh_scores[kvlen + threadIdx.x] = other;
        }
        __syncthreads();
    }
    max_s = sh_scores[kvlen];
    __syncthreads();
    float sum_exp = 0.f;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float e = expf(sh_scores[j] - max_s);
        sh_scores[j] = e;
        sum_exp += e;
    }
    __syncthreads();
    sh_scores[kvlen + threadIdx.x] = sum_exp;
    __syncthreads();
    for (int red = blockDim.x / 2; red > 0; red >>= 1) {
        if (threadIdx.x < red) sh_scores[kvlen + threadIdx.x] += sh_scores[kvlen + threadIdx.x + red];
        __syncthreads();
    }
    sum_exp = sh_scores[kvlen];
    __syncthreads();
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x)
        sh_scores[j] /= sum_exp;
    __syncthreads();

    float* out_row = attn_val + (i * nh + h) * hd;
    for (size_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float sum_v = 0.f;
        for (size_t j = 0; j < kvlen; j++)
            sum_v += sh_scores[j] * v_base[j * nkvh * hd + d];
        out_row[d] = sum_v;
    }
}

__global__ void self_attention_kernel_bf16(uint16_t* attn_val, const uint16_t* q, const uint16_t* k, const uint16_t* v, float scale, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    int causal_off = (int)kvlen - (int)qlen;
    size_t i = blockIdx.x;
    size_t h = blockIdx.y;
    if (i >= qlen || h >= nh) return;
    size_t kv_h = h * nkvh / nh;
    const uint16_t* q_row = q + (i * nh + h) * hd;
    const uint16_t* k_base = k + kv_h * hd;
    const uint16_t* v_base = v + kv_h * hd;

    extern __shared__ float sh[];
    float* sh_scores = sh;
    float max_s = -1e30f;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float dot = 0.f;
        for (size_t d = 0; d < hd; d++)
            dot += bf16_to_float(q_row[d]) * bf16_to_float(k_base[j * nkvh * hd + d]);
        float s = scale * dot;
        if ((int)j > (int)i + causal_off) s = -1e30f;
        sh_scores[j] = s;
        if (s > max_s) max_s = s;
    }
    __syncthreads();
    sh_scores[kvlen + threadIdx.x] = max_s;
    __syncthreads();
    for (int red = blockDim.x / 2; red > 0; red >>= 1) {
        if (threadIdx.x < red) {
            float other = sh_scores[kvlen + threadIdx.x + red];
            if (other > sh_scores[kvlen + threadIdx.x]) sh_scores[kvlen + threadIdx.x] = other;
        }
        __syncthreads();
    }
    max_s = sh_scores[kvlen];
    __syncthreads();
    float sum_exp = 0.f;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float e = expf(sh_scores[j] - max_s);
        sh_scores[j] = e;
        sum_exp += e;
    }
    __syncthreads();
    sh_scores[kvlen + threadIdx.x] = sum_exp;
    __syncthreads();
    for (int red = blockDim.x / 2; red > 0; red >>= 1) {
        if (threadIdx.x < red) sh_scores[kvlen + threadIdx.x] += sh_scores[kvlen + threadIdx.x + red];
        __syncthreads();
    }
    sum_exp = sh_scores[kvlen];
    __syncthreads();
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x)
        sh_scores[j] /= sum_exp;
    __syncthreads();

    uint16_t* out_row = attn_val + (i * nh + h) * hd;
    for (size_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float sum_v = 0.f;
        for (size_t j = 0; j < kvlen; j++)
            sum_v += sh_scores[j] * bf16_to_float(v_base[j * nkvh * hd + d]);
        out_row[d] = float_to_bf16(sum_v);
    }
}

void self_attention(std::byte* out, const std::byte* q, const std::byte* k, const std::byte* v, llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t num_heads, size_t nkvh, size_t head_dim, float scale) {
    dim3 grid(qlen, num_heads);
    size_t block = 256;
    size_t shmem = (kvlen + block) * sizeof(float);
    if (dtype == LLAISYS_DTYPE_F32)
        self_attention_kernel_f32<<<grid, block, shmem>>>((float*)out, (const float*)q, (const float*)k, (const float*)v, scale, qlen, kvlen, num_heads, nkvh, head_dim);
    else
        self_attention_kernel_bf16<<<grid, block, shmem>>>((uint16_t*)out, (const uint16_t*)q, (const uint16_t*)k, (const uint16_t*)v, scale, qlen, kvlen, num_heads, nkvh, head_dim);
}

} // namespace llaisys::ops::nvidia

#endif // ENABLE_NVIDIA_API
