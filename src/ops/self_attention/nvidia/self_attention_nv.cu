#include "../op.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../core/context/context.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

namespace llaisys::ops::nvidia {

constexpr int Br = 16;
constexpr int Bc = 16;

// shared memory layout (per block):
//   s_Q[Br * head_dim]
//   s_K[Bc * head_dim]
//   s_V[Bc * head_dim]
//   s_O[Br * head_dim]
//   s_m[Br]   (running max)
//   s_l[Br]   (running sum)

template<typename T>
__global__ void self_attention_kernel(
    T* output,
    const T* q_in,
    const T* k_in,
    const T* v_in,
    int q_len,
    int kv_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale)
{
    int q_block_idx = blockIdx.x;
    int head_idx    = blockIdx.y;
    int batch_idx   = blockIdx.z;
    int tid         = threadIdx.x;  // 0..Br-1

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int start_pos   = kv_len - q_len;

    int q_global_row = q_block_idx * Br + tid;

    int q_batch_offset  = batch_idx * q_len  * num_heads    * head_dim;
    int kv_batch_offset = batch_idx * kv_len * num_kv_heads * head_dim;

    // shared memory layout
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;                          // [Br * head_dim]
    float* s_K = s_Q + Br * head_dim;            // [Bc * head_dim]
    float* s_V = s_K + Bc * head_dim;            // [Bc * head_dim]
    float* s_O = s_V + Bc * head_dim;            // [Br * head_dim]
    float* s_m = s_O + Br * head_dim;            // [Br]
    float* s_l = s_m + Br;                       // [Br]

    // init running stats and output
    if (tid < Br) {
        s_m[tid] = -FLT_MAX;
        s_l[tid] = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            s_O[tid * head_dim + i] = 0.0f;
    }

    // load Q tile
    if (q_global_row < q_len) {
        int base = q_batch_offset + q_global_row * num_heads * head_dim + head_idx * head_dim;
        for (int i = 0; i < head_dim; ++i)
            s_Q[tid * head_dim + i] = static_cast<float>(q_in[base + i]);
    }
    __syncthreads();

    int num_k_blocks = (kv_len + Bc - 1) / Bc;
    for (int k_idx = 0; k_idx < num_k_blocks; ++k_idx) {

        int k_global_row_start = k_idx * Bc;

        // load K/V tile (tid indexes into Bc rows)
        if (tid < Bc) {
            int k_global_row = k_global_row_start + tid;
            if (k_global_row < kv_len) {
                int base = kv_batch_offset + k_global_row * num_kv_heads * head_dim + kv_head_idx * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    s_K[tid * head_dim + i] = static_cast<float>(k_in[base + i]);
                    s_V[tid * head_dim + i] = static_cast<float>(v_in[base + i]);
                }
            } else {
                for (int i = 0; i < head_dim; ++i) {
                    s_K[tid * head_dim + i] = 0.0f;
                    s_V[tid * head_dim + i] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_global_row < q_len) {
            int causal_limit = start_pos + q_global_row;

            // compute S = Q * K^T for this tile
            float r_S[Bc];
            float m_ij = -FLT_MAX;

            for (int j = 0; j < Bc; ++j) {
                int kj = k_global_row_start + j;
                if (kj >= kv_len || kj > causal_limit) {
                    r_S[j] = -FLT_MAX;
                    continue;
                }
                float sum = 0.0f;
                for (int i = 0; i < head_dim; ++i)
                    sum += s_Q[tid * head_dim + i] * s_K[j * head_dim + i];
                r_S[j] = sum * sm_scale;
                m_ij = fmaxf(m_ij, r_S[j]);
            }

            if (m_ij == -FLT_MAX) {
                __syncthreads();
                continue;
            }

            float m_i_new  = fmaxf(s_m[tid], m_ij);
            float exp_diff = expf(s_m[tid] - m_i_new);

            float l_ij = 0.0f;
            for (int j = 0; j < Bc; ++j) {
                r_S[j] = (r_S[j] == -FLT_MAX) ? 0.0f : expf(r_S[j] - m_i_new);
                l_ij += r_S[j];
            }

            for (int i = 0; i < head_dim; ++i) {
                float pv = 0.0f;
                for (int j = 0; j < Bc; ++j)
                    pv += r_S[j] * s_V[j * head_dim + i];
                s_O[tid * head_dim + i] = exp_diff * s_O[tid * head_dim + i] + pv;
            }

            s_m[tid] = m_i_new;
            s_l[tid] = exp_diff * s_l[tid] + l_ij;
        }
        __syncthreads();
    }

    // write output
    if (q_global_row < q_len) {
        float inv_l = 1.0f / s_l[tid];
        int base = q_batch_offset + q_global_row * num_heads * head_dim + head_idx * head_dim;
        for (int i = 0; i < head_dim; ++i)
            output[base + i] = static_cast<T>(s_O[tid * head_dim + i] * inv_l);
    }
}

template<typename T>
void launch_self_attention_kernel(tensor_t output, tensor_t q, tensor_t k, tensor_t v, float scale)
{
    int batch_size, q_len, kv_len, num_heads, num_kv_heads, head_dim;

    if (q->ndim() == 3) {
        batch_size   = 1;
        q_len        = q->shape()[0];
        num_heads    = q->shape()[1];
        head_dim     = q->shape()[2];
        kv_len       = k->shape()[0];
        num_kv_heads = k->shape()[1];
    } else {
        batch_size   = q->shape()[0];
        q_len        = q->shape()[1];
        num_heads    = q->shape()[2];
        head_dim     = q->shape()[3];
        kv_len       = k->shape()[1];
        num_kv_heads = k->shape()[2];
    }

    dim3 grid((q_len + Br - 1) / Br, num_heads, batch_size);
    dim3 block(Br);
    // s_Q + s_K + s_V + s_O + s_m + s_l
    size_t smem_bytes = (2 * Br + 2 * Bc) * head_dim * sizeof(float)
                      + 2 * Br * sizeof(float);

    self_attention_kernel<T><<<grid, block, smem_bytes>>>(
        reinterpret_cast<T*>(output->data()),
        reinterpret_cast<const T*>(q->data()),
        reinterpret_cast<const T*>(k->data()),
        reinterpret_cast<const T*>(v->data()),
        q_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
}

void self_attention(tensor_t output, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:  launch_self_attention_kernel<float>(output, q, k, v, scale); break;
        case LLAISYS_DTYPE_F16:  launch_self_attention_kernel<__half>(output, q, k, v, scale); break;
        case LLAISYS_DTYPE_BF16: launch_self_attention_kernel<__nv_bfloat16>(output, q, k, v, scale); break;
        default: abort();
    }
}

} // namespace llaisys::ops::nvidia
