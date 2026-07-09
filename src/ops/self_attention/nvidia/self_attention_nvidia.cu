#include "self_attention_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../nvidia_utils.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void self_attention_kernel(T *out, const T *q, const T *k, const T *v, float scale, size_t qlen, size_t kvlen,
                                      size_t nh, size_t nkvh, size_t hd, size_t dv) {
    size_t i = static_cast<size_t>(blockIdx.x);
    size_t h = static_cast<size_t>(blockIdx.y);
    if (i >= qlen || h >= nh) {
        return;
    }
    size_t kv_group = nh / nkvh;
    size_t kvh = h / kv_group;
    const T *q_ptr = q + (i * nh + h) * hd;
    extern __shared__ float shared[];
    float *smax = shared;
    float *ssum = shared + blockDim.x;
    int tid = threadIdx.x;
    float local_max = -INFINITY;
    int64_t limit = static_cast<int64_t>(i) + static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);
    for (size_t j = static_cast<size_t>(tid); j < kvlen; j += static_cast<size_t>(blockDim.x)) {
        if (static_cast<int64_t>(j) > limit) {
            continue;
        }
        const T *k_ptr = k + (j * nkvh + kvh) * hd;
        float acc = 0.0f;
        for (size_t d = 0; d < hd; ++d) {
            acc += detail::to_float(q_ptr[d]) * detail::to_float(k_ptr[d]);
        }
        acc *= scale;
        if (acc > local_max) {
            local_max = acc;
        }
    }
    smax[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<int>(s)) {
            float other = smax[tid + s];
            if (other > smax[tid]) {
                smax[tid] = other;
            }
        }
        __syncthreads();
    }
    float max_score = smax[0];
    float local_sum = 0.0f;
    for (size_t j = static_cast<size_t>(tid); j < kvlen; j += static_cast<size_t>(blockDim.x)) {
        if (static_cast<int64_t>(j) > limit) {
            continue;
        }
        const T *k_ptr = k + (j * nkvh + kvh) * hd;
        float acc = 0.0f;
        for (size_t d = 0; d < hd; ++d) {
            acc += detail::to_float(q_ptr[d]) * detail::to_float(k_ptr[d]);
        }
        acc *= scale;
        local_sum += expf(acc - max_score);
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<int>(s)) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    float sum_score = ssum[0];
    if (sum_score <= 0.0f) {
        sum_score = 1.0f;
    }
    for (size_t d = static_cast<size_t>(tid); d < dv; d += static_cast<size_t>(blockDim.x)) {
        float acc = 0.0f;
        for (size_t j = 0; j < kvlen; ++j) {
            if (static_cast<int64_t>(j) > limit) {
                continue;
            }
            const T *k_ptr = k + (j * nkvh + kvh) * hd;
            float score = 0.0f;
            for (size_t kd = 0; kd < hd; ++kd) {
                score += detail::to_float(q_ptr[kd]) * detail::to_float(k_ptr[kd]);
            }
            score *= scale;
            float w = expf(score - max_score) / sum_score;
            const T *v_ptr = v + (j * nkvh + kvh) * dv;
            acc += w * detail::to_float(v_ptr[d]);
        }
        out[(i * nh + h) * dv + d] = detail::from_float<T>(acc);
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t dv) {
    auto &runtime = llaisys::core::context().runtime();
    dim3 grid(static_cast<unsigned int>(qlen), static_cast<unsigned int>(nh), 1);
    int threads = 256;
    size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float) * 2;
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
