#include "selfattn_cu.cuh"

#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace selfattnops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// One block per (seq, head) pair.
// Dynamic shared memory layout:
//   float scores[kvlen]       -- QK dot products
//   float reduce[blockDim.x]  -- block-reduction scratch
//
// NOTE: requires (kvlen + blockDim.x) * sizeof(float) <= shared memory limit
//       (~49 KB on Ampere), i.e. kvlen up to ~12 000 tokens at blockDim.x=128.
template <typename T>
__global__ void
self_attn_kernel(T *attn_val,
                 const T *q, const T *k, const T *v,
                 size_t seqlen, size_t num_head, size_t head_dim,
                 size_t kvlen, size_t num_kv_head, size_t vdim,
                 float scale) {
    extern __shared__ float smem[];
    float *scores     = smem;
    float *reduce_buf = smem + kvlen;   // [blockDim.x] for reductions

    size_t block_id   = blockIdx.x;
    size_t s          = block_id / num_head;
    size_t h          = block_id % num_head;
    unsigned tid      = threadIdx.x;
    unsigned nthreads = blockDim.x;

    size_t num_groups = num_head / num_kv_head;
    size_t head_k     = h / num_groups;

    // causal: attend to positions [0, L)
    size_t L = kvlen - seqlen + s + 1;
    if (L > kvlen) L = kvlen;

    size_t qbase = s * num_head * head_dim + h * head_dim;

    // ---- Phase 1: compute QK^T scores ----
    for (size_t t = tid; t < L; t += nthreads) {
        size_t kbase = t * num_kv_head * head_dim + head_k * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d)
            dot += to_float(q[qbase + d]) * to_float(k[kbase + d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ---- Phase 2: find global max (numerically stable softmax) ----
    float local_max = -FLT_MAX;
    for (size_t t = tid; t < L; t += nthreads)
        local_max = fmaxf(local_max, scores[t]);
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (unsigned s2 = nthreads / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s2]);
        __syncthreads();
    }
    float gmax = reduce_buf[0];
    __syncthreads();

    // ---- Phase 3: exp(score - max), compute sum ----
    float local_sum = 0.0f;
    for (size_t t = tid; t < L; t += nthreads) {
        scores[t] = expf(scores[t] - gmax);
        local_sum += scores[t];
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (unsigned s2 = nthreads / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) reduce_buf[tid] += reduce_buf[tid + s2];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_buf[0];
    __syncthreads();

    // ---- Phase 4: normalize scores in-place ----
    for (size_t t = tid; t < L; t += nthreads)
        scores[t] *= inv_sum;
    __syncthreads();

    // ---- Phase 5: weighted sum of V ----
    size_t obase = s * num_head * vdim + h * vdim;
    for (size_t d = tid; d < vdim; d += nthreads) {
        float acc = 0.0f;
        for (size_t t = 0; t < L; ++t) {
            size_t vbase = t * num_kv_head * vdim + head_k * vdim;
            acc += scores[t] * to_float(v[vbase + d]);
        }
        attn_val[obase + d] = from_float<T>(acc);
    }
}

} // namespace selfattnops::nvidia

namespace llaisys::ops::nvidia {

void self_attn(std::byte *attn_val,
               const std::byte *q,
               const std::byte *k,
               const std::byte *v,
               size_t seqlen,
               size_t num_head,
               size_t head_dim,
               size_t kvlen,
               size_t num_kv_head,
               size_t vdim,
               float scale,
               llaisysDataType_t dtype) {
    constexpr int threads = 128; // must be power of 2
    size_t smem_size = (kvlen + threads) * sizeof(float);
    unsigned int blocks = static_cast<unsigned int>(seqlen * num_head);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        selfattnops::nvidia::self_attn_kernel<float>
            <<<blocks, threads, smem_size>>>(
                reinterpret_cast<float *>(attn_val),
                reinterpret_cast<const float *>(q),
                reinterpret_cast<const float *>(k),
                reinterpret_cast<const float *>(v),
                seqlen, num_head, head_dim, kvlen, num_kv_head, vdim, scale);
        break;
    case LLAISYS_DTYPE_F16:
        selfattnops::nvidia::self_attn_kernel<__half>
            <<<blocks, threads, smem_size>>>(
                reinterpret_cast<__half *>(attn_val),
                reinterpret_cast<const __half *>(q),
                reinterpret_cast<const __half *>(k),
                reinterpret_cast<const __half *>(v),
                seqlen, num_head, head_dim, kvlen, num_kv_head, vdim, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        selfattnops::nvidia::self_attn_kernel<__nv_bfloat16>
            <<<blocks, threads, smem_size>>>(
                reinterpret_cast<__nv_bfloat16 *>(attn_val),
                reinterpret_cast<const __nv_bfloat16 *>(q),
                reinterpret_cast<const __nv_bfloat16 *>(k),
                reinterpret_cast<const __nv_bfloat16 *>(v),
                seqlen, num_head, head_dim, kvlen, num_kv_head, vdim, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
