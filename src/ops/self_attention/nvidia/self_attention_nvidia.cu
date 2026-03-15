#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cstddef>

namespace {

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T, int BLOCK_SIZE, int MAX_LOCAL_OUT>
__global__ void self_attention_online_kernel(T *__restrict__ out,
                                             const T *__restrict__ q,
                                             const T *__restrict__ k,
                                             const T *__restrict__ v,
                                             size_t seqlen,
                                             size_t nhead,
                                             size_t nkvhead,
                                             size_t d,
                                             size_t dv,
                                             size_t total_len,
                                             float scale) {
    const size_t block_id = static_cast<size_t>(blockIdx.x);
    if (block_id >= seqlen * nhead) {
        return;
    }

    const size_t qi = block_id / nhead;
    const size_t qh = block_id % nhead;
    const size_t kv_head = qh * nkvhead / nhead;

    const T *q_row = q + (qi * nhead + qh) * d;
    T *out_row = out + (qi * nhead + qh) * dv;

    const ptrdiff_t diag = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);
    const ptrdiff_t max_visible_key = static_cast<ptrdiff_t>(qi) + diag;
    if (max_visible_key < 0) {
        for (size_t m = static_cast<size_t>(threadIdx.x); m < dv; m += BLOCK_SIZE) {
            out_row[m] = from_float<T>(0.0f);
        }
        return;
    }
    const size_t visible_len = (static_cast<size_t>(max_visible_key) + 1 < total_len)
                                   ? static_cast<size_t>(max_visible_key) + 1
                                   : total_len;

    // Dynamic shared memory layout: [q_cache(d), score(1)]
    extern __shared__ float smem[];
    float *q_cache = smem;
    float *score_ptr = q_cache + d;

    for (size_t kd = static_cast<size_t>(threadIdx.x); kd < d; kd += BLOCK_SIZE) {
        q_cache[kd] = to_float(q_row[kd]);
    }
    __syncthreads();

    int local_idx[MAX_LOCAL_OUT];
    float local_acc[MAX_LOCAL_OUT];
    int local_n = 0;
    for (size_t m = static_cast<size_t>(threadIdx.x); m < dv && local_n < MAX_LOCAL_OUT; m += BLOCK_SIZE) {
        local_idx[local_n] = static_cast<int>(m);
        local_acc[local_n] = 0.0f;
        ++local_n;
    }

    float row_m = -INFINITY;
    float row_l = 0.0f;

    for (size_t j = 0; j < visible_len; ++j) {
        if (threadIdx.x < 32) {
            const T *k_row = k + (j * nkvhead + kv_head) * d;
            float dot = 0.0f;
            for (size_t kd = static_cast<size_t>(threadIdx.x); kd < d; kd += 32) {
                dot += q_cache[kd] * to_float(k_row[kd]);
            }
            dot = warp_sum(dot);
            if (threadIdx.x == 0) {
                *score_ptr = dot * scale;
            }
        }
        __syncthreads();

        const float score = *score_ptr;
        const float m_new = fmaxf(row_m, score);
        const float alpha = (row_l == 0.0f) ? 0.0f : expf(row_m - m_new);
        const float beta = expf(score - m_new);
        const float l_new = row_l * alpha + beta;

        const T *v_row = v + (j * nkvhead + kv_head) * dv;
        #pragma unroll
        for (int t = 0; t < MAX_LOCAL_OUT; ++t) {
            if (t < local_n) {
                local_acc[t] = local_acc[t] * alpha + beta * to_float(v_row[local_idx[t]]);
            }
        }
        row_m = m_new;
        row_l = l_new;
        __syncthreads();
    }

    const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
    #pragma unroll
    for (int t = 0; t < MAX_LOCAL_OUT; ++t) {
        if (t < local_n) {
            out_row[local_idx[t]] = from_float<T>(local_acc[t] * inv_l);
        }
    }

    // Rare fallback for very large dv.
    for (size_t m = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(BLOCK_SIZE * MAX_LOCAL_OUT); m < dv;
         m += BLOCK_SIZE) {
        float acc = 0.0f;
        for (size_t j = 0; j < visible_len; ++j) {
            const T *k_row = k + (j * nkvhead + kv_head) * d;
            float dot = 0.0f;
            for (size_t kd = 0; kd < d; ++kd) {
                dot += q_cache[kd] * to_float(k_row[kd]);
            }
            const float prob = (row_l > 0.0f) ? expf(dot * scale - row_m) * inv_l : 0.0f;
            acc += prob * to_float(v[(j * nkvhead + kv_head) * dv + m]);
        }
        out_row[m] = from_float<T>(acc);
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t type,
                    size_t seqlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t d,
                    size_t dv,
                    size_t total_len,
                    float scale) {
    if (seqlen == 0 || nhead == 0 || nkvhead == 0 || d == 0 || dv == 0 || total_len == 0) {
        return;
    }

    const int grid_size = static_cast<int>(seqlen * nhead);
    constexpr int block_size = 128;
    constexpr int max_local_out = 8;
    const size_t smem_bytes = sizeof(float) * (d + 1);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_online_kernel<float, block_size, max_local_out><<<grid_size, block_size, smem_bytes>>>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_online_kernel<half, block_size, max_local_out><<<grid_size, block_size, smem_bytes>>>(
            reinterpret_cast<half *>(attn_val),
            reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(k),
            reinterpret_cast<const half *>(v),
            seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_online_kernel<__nv_bfloat16, block_size, max_local_out><<<grid_size, block_size, smem_bytes>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
