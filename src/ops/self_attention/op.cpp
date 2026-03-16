#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>
#include <limits>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

// 因果 mask：与 PyTorch tril(diagonal=kvlen-qlen) 一致，即 (i,j) 有效当且仅当 j <= i + (kvlen - qlen)
template <typename T>
void self_attention_impl(float *attn_val, const T *q, const T *k, const T *v, float scale,
                        size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    const ptrdiff_t causal_off = static_cast<ptrdiff_t>(kvlen) - static_cast<ptrdiff_t>(qlen);
    const float neg_inf = -std::numeric_limits<float>::infinity();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> scores(static_cast<size_t>(kvlen));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nh; h++) {
            const size_t kv_h = h * nkvh / nh;
            const T *q_row = q + i * nh * hd + h * hd;
            const T *k_base = k + kv_h * hd;
            const T *v_base = v + kv_h * hd;

            for (size_t j = 0; j < kvlen; j++) {
                float dot = 0.f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t d = 0; d < hd; d++)
                        dot += llaisys::utils::cast<float>(q_row[d]) *
                               llaisys::utils::cast<float>(k_base[j * nkvh * hd + d]);
                } else {
                    for (size_t d = 0; d < hd; d++)
                        dot += static_cast<float>(q_row[d]) * static_cast<float>(k_base[j * nkvh * hd + d]);
                }
                float s = scale * dot;
                if (static_cast<ptrdiff_t>(j) > static_cast<ptrdiff_t>(i) + causal_off)
                    s = neg_inf;
                scores[j] = s;
            }

            // softmax over kvlen (numerically stable)
            float max_s = scores[0];
            for (size_t j = 1; j < kvlen; j++)
                if (scores[j] > max_s) max_s = scores[j];
            float sum_exp = 0.f;
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] = std::exp(scores[j] - max_s);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j < kvlen; j++)
                scores[j] /= sum_exp;

            // attn_val[i,h,:] = sum_j scores[j] * v[j,kv_h,:]
            float *out_row = attn_val + (i * nh + h) * hd;
            for (size_t d = 0; d < hd; d++) {
                float sum_v = 0.f;
                for (size_t j = 0; j < kvlen; j++)
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
                        sum_v += scores[j] * llaisys::utils::cast<float>(v_base[j * nkvh * hd + d]);
                    else
                        sum_v += scores[j] * static_cast<float>(v_base[j * nkvh * hd + d]);
                out_row[d] = sum_v;
            }
        }
        }
    }
}

template <typename T>
void self_attention_cpu_typed(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                              float scale, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    const T *q_t = reinterpret_cast<const T *>(q);
    const T *k_t = reinterpret_cast<const T *>(k);
    const T *v_t = reinterpret_cast<const T *>(v);
    std::vector<float> out_f(qlen * nh * hd);
    self_attention_impl(out_f.data(), q_t, k_t, v_t, scale, qlen, kvlen, nh, nkvh, hd);
    T *out_t = reinterpret_cast<T *>(attn_val);
    const size_t total = qlen * nh * hd;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < total; idx++)
        out_t[idx] = llaisys::utils::cast<T>(out_f[idx]);
}

void self_attention_cpu(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                        llaisysDataType_t dtype, float scale, size_t qlen, size_t kvlen, size_t nh,
                        size_t nkvh, size_t hd) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                           reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale,
                           qlen, kvlen, nh, nkvh, hd);
        return;
    case LLAISYS_DTYPE_F16:
        self_attention_cpu_typed<llaisys::fp16_t>(attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd);
        return;
    case LLAISYS_DTYPE_BF16:
        self_attention_cpu_typed<llaisys::bf16_t>(attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->dtype() == q->dtype() && q->dtype() == k->dtype() && k->dtype() == v->dtype(),
           "self_attention: attn_val, q, k, v must have same dtype");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "self_attention: q,k,v,attn_val must be 3D");
    size_t qlen = q->shape()[0], nh = q->shape()[1], hd = q->shape()[2];
    size_t kvlen = k->shape()[0], nkvh = k->shape()[1];
    size_t dv = v->shape()[2];
    ASSERT(k->shape()[2] == hd, "self_attention: k head dim must match q");
    ASSERT(nh >= nkvh && nh % nkvh == 0, "self_attention: nhead must be multiple of nkvhead");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == dv,
           "self_attention: attn_val shape must be [qlen, nhead, dv]");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh, "self_attention: v shape must match k");

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return self_attention_cpu(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   scale, qlen, kvlen, nh, nkvh, hd);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
        break;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
