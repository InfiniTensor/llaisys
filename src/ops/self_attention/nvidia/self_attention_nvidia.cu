#include "self_attention_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
__global__ void self_attention_kernel_f32(
    float* out, const float* q, const float* k, const float* v,
    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
    size_t d, size_t dv, float scale
) {
    size_t q_idx = blockIdx.x; // 当前的 token 位置
    size_t h_idx = blockIdx.y; // 当前的注意力头
    size_t tid = threadIdx.x;

    // GQA: 映射到对应的 KV 头
    size_t kv_h_idx = h_idx / (nhead / nkvhead);

    // 动态分配的共享内存，用于存储当前 Query 对所有 KV 的打分
    extern __shared__ float scores[];

    // 1. 并行计算点积 (Dot Product)
    for (size_t k_idx = tid; k_idx < total_len; k_idx += blockDim.x) {
        // Causal Mask 逻辑：强制转为 signed long long 防止无符号数下溢出
        if ((long long)k_idx > (long long)q_idx + (long long)total_len - (long long)seqlen) {
            scores[k_idx] = -1e20f; // 设为负无穷
        } else {
            float sum = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                float q_val = q[q_idx * (nhead * d) + h_idx * d + i];
                float k_val = k[k_idx * (nkvhead * d) + kv_h_idx * d + i];
                sum += q_val * k_val;
            }
            scores[k_idx] = sum * scale;
        }
    }
    __syncthreads();

    // 2. Softmax 操作 (由 0 号线程安全处理共享内存数组)
    __shared__ float sum_exp;
    if (tid == 0) {
        float max_score = -1e20f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            if (scores[k_idx] > max_score) max_score = scores[k_idx];
        }
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float exp_val = expf(scores[k_idx] - max_score);
            scores[k_idx] = exp_val;
            sum += exp_val;
        }
        sum_exp = sum;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            scores[k_idx] /= sum_exp;
        }
    }
    __syncthreads();

    // 3. 并行计算 V 的加权和
    for (size_t v_idx = tid; v_idx < dv; v_idx += blockDim.x) {
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float val = v[k_idx * (nkvhead * dv) + kv_h_idx * dv + v_idx];
            sum += scores[k_idx] * val;
        }
        out[q_idx * (nhead * dv) + h_idx * dv + v_idx] = sum;
    }
}

// --- F16 Kernel ---
__global__ void self_attention_kernel_f16(
    void* out_ptr, const void* q_ptr, const void* k_ptr, const void* v_ptr,
    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
    size_t d, size_t dv, float scale
) {
    size_t q_idx = blockIdx.x;
    size_t h_idx = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t kv_h_idx = h_idx / (nhead / nkvhead);

    const __half* q = reinterpret_cast<const __half*>(q_ptr);
    const __half* k = reinterpret_cast<const __half*>(k_ptr);
    const __half* v = reinterpret_cast<const __half*>(v_ptr);
    __half* out = reinterpret_cast<__half*>(out_ptr);

    extern __shared__ float scores[];

    for (size_t k_idx = tid; k_idx < total_len; k_idx += blockDim.x) {
        if ((long long)k_idx > (long long)q_idx + (long long)total_len - (long long)seqlen) {
            scores[k_idx] = -1e20f;
        } else {
            float sum = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                float q_val = __half2float(q[q_idx * (nhead * d) + h_idx * d + i]);
                float k_val = __half2float(k[k_idx * (nkvhead * d) + kv_h_idx * d + i]);
                sum += q_val * k_val;
            }
            scores[k_idx] = sum * scale;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float max_score = -1e20f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            if (scores[k_idx] > max_score) max_score = scores[k_idx];
        }
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float exp_val = expf(scores[k_idx] - max_score);
            scores[k_idx] = exp_val;
            sum += exp_val;
        }
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            scores[k_idx] /= sum;
        }
    }
    __syncthreads();

    for (size_t v_idx = tid; v_idx < dv; v_idx += blockDim.x) {
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float val = __half2float(v[k_idx * (nkvhead * dv) + kv_h_idx * dv + v_idx]);
            sum += scores[k_idx] * val;
        }
        out[q_idx * (nhead * dv) + h_idx * dv + v_idx] = __float2half(sum);
    }
}

// --- BF16 Kernel ---
__global__ void self_attention_kernel_bf16(
    void* out_ptr, const void* q_ptr, const void* k_ptr, const void* v_ptr,
    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
    size_t d, size_t dv, float scale
) {
#if __CUDACC_VER_MAJOR__ >= 11
    size_t q_idx = blockIdx.x;
    size_t h_idx = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t kv_h_idx = h_idx / (nhead / nkvhead);

    const __nv_bfloat16* q = reinterpret_cast<const __nv_bfloat16*>(q_ptr);
    const __nv_bfloat16* k = reinterpret_cast<const __nv_bfloat16*>(k_ptr);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(v_ptr);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_ptr);

    extern __shared__ float scores[];

    for (size_t k_idx = tid; k_idx < total_len; k_idx += blockDim.x) {
        if ((long long)k_idx > (long long)q_idx + (long long)total_len - (long long)seqlen) {
            scores[k_idx] = -1e20f;
        } else {
            float sum = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                float q_val = __bfloat162float(q[q_idx * (nhead * d) + h_idx * d + i]);
                float k_val = __bfloat162float(k[k_idx * (nkvhead * d) + kv_h_idx * d + i]);
                sum += q_val * k_val;
            }
            scores[k_idx] = sum * scale;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float max_score = -1e20f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            if (scores[k_idx] > max_score) max_score = scores[k_idx];
        }
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float exp_val = expf(scores[k_idx] - max_score);
            scores[k_idx] = exp_val;
            sum += exp_val;
        }
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            scores[k_idx] /= sum;
        }
    }
    __syncthreads();

    for (size_t v_idx = tid; v_idx < dv; v_idx += blockDim.x) {
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < total_len; ++k_idx) {
            float val = __bfloat162float(v[k_idx * (nkvhead * dv) + kv_h_idx * dv + v_idx]);
            sum += scores[k_idx] * val;
        }
        out[q_idx * (nhead * dv) + h_idx * dv + v_idx] = __float2bfloat16(sum);
    }
#endif
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    llaisysDataType_t type, 
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv, 
                    float scale) {
    
    // Grid: [seqlen, nhead] 每一个 Block 独立负责一个 Q 向量的完整处理
    dim3 blocks(seqlen, nhead);
    int threads_per_block = 256;
    
    // 动态分配共享内存，存放长度为 total_len 的 attention scores
    size_t shared_mem_size = total_len * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel_f32<<<blocks, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<float*>(attn_val),
            reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel_f16<<<blocks, threads_per_block, shared_mem_size>>>(
            attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel_bf16<<<blocks, threads_per_block, shared_mem_size>>>(
            attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia