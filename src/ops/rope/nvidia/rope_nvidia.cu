#include "rope_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
__global__ void rope_kernel_f32(float* out, const float* in, const int64_t* pos_ids, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 总线程数：seqlen * nhead * half_dim
    if (idx < seqlen * nhead * half_dim) {
        // 解析当前处理的多维坐标
        size_t pair_idx = idx % half_dim;
        size_t head_idx = (idx / half_dim) % nhead;
        size_t seq_idx = idx / (half_dim * nhead);
        
        // 🚨 核心修复：对齐 CPU 版本的内存跳跃步长，前一半和后一半组合！
        size_t idx_a = seq_idx * (nhead * head_dim) + head_idx * head_dim + pair_idx;
        size_t idx_b = idx_a + half_dim;
        
        // 计算旋转频率和角度
        float freq = 1.0f / powf(theta, (2.0f * (float)pair_idx) / (float)head_dim);
        float m_theta = (float)pos_ids[seq_idx] * freq;
        float cos_m = cosf(m_theta);
        float sin_m = sinf(m_theta);

        // 取出相隔 half_dim 的两个特征，执行复数旋转
        float x0 = in[idx_a];
        float x1 = in[idx_b];
        out[idx_a] = x0 * cos_m - x1 * sin_m;
        out[idx_b] = x1 * cos_m + x0 * sin_m;
    }
}

// --- F16 Kernel ---
__global__ void rope_kernel_f16(void* out_ptr, const void* in_ptr, const int64_t* pos_ids, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seqlen * nhead * half_dim) {
        size_t pair_idx = idx % half_dim;
        size_t head_idx = (idx / half_dim) % nhead;
        size_t seq_idx = idx / (half_dim * nhead);
        
        size_t idx_a = seq_idx * (nhead * head_dim) + head_idx * head_dim + pair_idx;
        size_t idx_b = idx_a + half_dim;
        
        float freq = 1.0f / powf(theta, (2.0f * (float)pair_idx) / (float)head_dim);
        float m_theta = (float)pos_ids[seq_idx] * freq;
        float cos_m = cosf(m_theta);
        float sin_m = sinf(m_theta);

        const __half* in = reinterpret_cast<const __half*>(in_ptr);
        __half* out = reinterpret_cast<__half*>(out_ptr);

        float x0 = __half2float(in[idx_a]);
        float x1 = __half2float(in[idx_b]);
        out[idx_a] = __float2half(x0 * cos_m - x1 * sin_m);
        out[idx_b] = __float2half(x1 * cos_m + x0 * sin_m);
    }
}

// --- BF16 Kernel ---
__global__ void rope_kernel_bf16(void* out_ptr, const void* in_ptr, const int64_t* pos_ids, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
#if __CUDACC_VER_MAJOR__ >= 11
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seqlen * nhead * half_dim) {
        size_t pair_idx = idx % half_dim;
        size_t head_idx = (idx / half_dim) % nhead;
        size_t seq_idx = idx / (half_dim * nhead);
        
        size_t idx_a = seq_idx * (nhead * head_dim) + head_idx * head_dim + pair_idx;
        size_t idx_b = idx_a + half_dim;
        
        float freq = 1.0f / powf(theta, (2.0f * (float)pair_idx) / (float)head_dim);
        float m_theta = (float)pos_ids[seq_idx] * freq;
        float cos_m = cosf(m_theta);
        float sin_m = sinf(m_theta);

        const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(in_ptr);
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_ptr);

        float x0 = __bfloat162float(in[idx_a]);
        float x1 = __bfloat162float(in[idx_b]);
        out[idx_a] = __float2bfloat16(x0 * cos_m - x1 * sin_m);
        out[idx_b] = __float2bfloat16(x1 * cos_m + x0 * sin_m);
    }
#endif
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    
    size_t total_pairs = seqlen * nhead * (head_dim / 2);
    int threads_per_block = 256;
    int blocks_per_grid = (total_pairs + threads_per_block - 1) / threads_per_block;
    
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel_f32<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), pos_ptr, seqlen, nhead, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel_f16<<<blocks_per_grid, threads_per_block>>>(out, in, pos_ptr, seqlen, nhead, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(out, in, pos_ptr, seqlen, nhead, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia