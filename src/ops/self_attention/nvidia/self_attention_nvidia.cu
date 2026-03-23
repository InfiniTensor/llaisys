#include "self_attention_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void self_attention_kernel(T *out, const T *q, const T *k, const T *v, size_t qlen, size_t kvlen,
                                      size_t nhead, size_t nkvh, size_t dim, size_t dv, float scale) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = qlen * nhead;
    if (idx >= total) {
        return;
    }
    size_t s = idx / nhead;
    size_t h = idx % nhead;
    size_t head_factor = nhead / nkvh;
    size_t kh = h / head_factor;

    const T *q_vec = q + (s * nhead + h) * dim;
    const T *k_base = k + kh * dim;
    const T *v_base = v + kh * dv;

    int allow_upto = static_cast<int>(s + kvlen - qlen);
    float max_logit = -1e20f;
    for (size_t t = 0; t < kvlen; ++t) {
        float logit;
        if (static_cast<int>(t) > allow_upto) {
            logit = -1e20f;
        } else {
            const T *k_vec = k_base + t * nkvh * dim;
            float dot = 0.f;
            for (size_t j = 0; j < dim; ++j) {
                float qv = llaisys::device::nvidia::ScalarOps<T>::load(q_vec + j);
                float kv = llaisys::device::nvidia::ScalarOps<T>::load(k_vec + j);
                dot += qv * kv;
            }
            logit = dot * scale;
        }
        max_logit = fmaxf(max_logit, logit);
    }

    float sum_exp = 0.f;
    for (size_t t = 0; t < kvlen; ++t) {
        float logit;
        if (static_cast<int>(t) > allow_upto) {
            logit = -1e20f;
        } else {
            const T *k_vec = k_base + t * nkvh * dim;
            float dot = 0.f;
            for (size_t j = 0; j < dim; ++j) {
                float qv = llaisys::device::nvidia::ScalarOps<T>::load(q_vec + j);
                float kv = llaisys::device::nvidia::ScalarOps<T>::load(k_vec + j);
                dot += qv * kv;
            }
            logit = dot * scale;
        }
        sum_exp += expf(logit - max_logit);
    }
    float inv_sum = 1.0f / sum_exp;

    T *y = out + (s * nhead + h) * dv;
    for (size_t d = 0; d < dv; ++d) {
        float acc = 0.f;
        for (size_t t = 0; t < kvlen; ++t) {
            float logit;
            if (static_cast<int>(t) > allow_upto) {
                logit = -1e20f;
            } else {
                const T *k_vec = k_base + t * nkvh * dim;
                float dot = 0.f;
                for (size_t j = 0; j < dim; ++j) {
                    float qv = llaisys::device::nvidia::ScalarOps<T>::load(q_vec + j);
                    float kv = llaisys::device::nvidia::ScalarOps<T>::load(k_vec + j);
                    dot += qv * kv;
                }
                logit = dot * scale;
            }
            float prob = expf(logit - max_logit) * inv_sum;
            const T *v_vec = v_base + t * nkvh * dv;
            float vv = llaisys::device::nvidia::ScalarOps<T>::load(v_vec + d);
            acc += prob * vv;
        }
        llaisys::device::nvidia::ScalarOps<T>::store(y + d, acc);
    }
}

template <typename T>
void launch_self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, size_t qlen,
                           size_t kvlen, size_t nhead, size_t nkvh, size_t dim, size_t dv, float scale) {
    size_t total = qlen * nhead;
    const int threads = 64;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    self_attention_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(q),
                                               reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v), qlen,
                                               kvlen, nhead, nkvh, dim, dv, scale);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvh, size_t dim,
                    size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention<float>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention<llaisys::bf16_t>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention<llaisys::fp16_t>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
