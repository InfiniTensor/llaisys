#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void selfAttentionKernel(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t qlen, size_t kvlen,
                                    size_t nhead, size_t nkvhead, size_t head_dim) {
    extern __shared__ float shared_mem[];
    float *scores = shared_mem;
    float *reduce_buf = shared_mem + kvlen;

    const size_t qi = static_cast<size_t>(blockIdx.x);
    const size_t h = static_cast<size_t>(blockIdx.y);
    const size_t group_size = nhead / nkvhead;
    const size_t kv_h = h / group_size;
    const int tid = threadIdx.x;

    const size_t q_base = (qi * nhead + h) * head_dim;
    const size_t max_attend = kvlen - qlen + qi;

    float thread_max = -INFINITY;
    for (size_t ki = static_cast<size_t>(tid); ki < kvlen; ki += blockDim.x) {
        float score = -INFINITY;
        if (ki <= max_attend) {
            score = 0.0f;
            const size_t k_base = (ki * nkvhead + kv_h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                score += llaisys::device::nvidia::scalarToFloat<T>(q[q_base + d])
                       * llaisys::device::nvidia::scalarToFloat<T>(k[k_base + d]);
            }
            score *= scale;
        }
        scores[ki] = score;
        if (score > thread_max) {
            thread_max = score;
        }
    }

    reduce_buf[tid] = thread_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    float thread_sum = 0.0f;
    for (size_t ki = static_cast<size_t>(tid); ki < kvlen; ki += blockDim.x) {
        const float prob = expf(scores[ki] - max_score);
        scores[ki] = prob;
        thread_sum += prob;
    }

    reduce_buf[tid] = thread_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] += reduce_buf[tid + offset];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    for (size_t d = static_cast<size_t>(tid); d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (size_t ki = 0; ki < kvlen; ++ki) {
            const size_t v_base = (ki * nkvhead + kv_h) * head_dim;
            out_val += (scores[ki] * inv_sum) * llaisys::device::nvidia::scalarToFloat<T>(v[v_base + d]);
        }
        attn_val[q_base + d] = llaisys::device::nvidia::floatToScalar<T>(out_val);
    }
}

} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t head_dim,
                    llaisysStream_t stream) {
    if (qlen == 0 || kvlen == 0 || nhead == 0 || nkvhead == 0 || head_dim == 0) {
        return;
    }

    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned int>(qlen), static_cast<unsigned int>(nhead), 1);
    const size_t shared_bytes = (kvlen + threads) * sizeof(float);
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        selfAttentionKernel<<<grid, threads, shared_bytes, cuda_stream>>>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale,
            qlen,
            kvlen,
            nhead,
            nkvhead,
            head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        selfAttentionKernel<<<grid, threads, shared_bytes, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale,
            qlen,
            kvlen,
            nhead,
            nkvhead,
            head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        selfAttentionKernel<<<grid, threads, shared_bytes, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale,
            qlen,
            kvlen,
            nhead,
            nkvhead,
            head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
