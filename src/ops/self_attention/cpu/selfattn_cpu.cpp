#include "selfattn_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

template <typename T>
static void self_attn_impl(T *attn_val,
                           const T *q,
                           const T *k,
                           const T *v,
                           size_t seqlen,
                           size_t num_head,
                           size_t head_dim,
                           size_t kvlen,
                           size_t num_kv_head,
                           size_t vdim,
                           float scale) {

#pragma omp parallel
    {
        std::vector<float> scores(kvlen);

#pragma omp for collapse(2) schedule(static)
        for (size_t h = 0; h < num_head; ++h) {
            size_t num_groups = num_head / num_kv_head;
            size_t head_k = h / num_groups;

            for (size_t s = 0; s < seqlen; ++s) {
                float *sc = scores.data();

                size_t bound = kvlen - seqlen + s;
                size_t L = std::min(kvlen, bound + 1);

                // QK^T + max
                float mx = -INFINITY;
                size_t qbase = s * num_head * head_dim + h * head_dim;

                for (size_t t = 0; t < L; ++t) {
                    size_t kbase
                        = t * num_kv_head * head_dim + head_k * head_dim;
                    long double dot = 0.0L;
#pragma omp simd reduction(+ : dot)
                    for (size_t d = 0; d < head_dim; ++d)
                        dot += casting(long double, q[qbase + d])
                             * casting(long double, k[kbase + d]);
                    float val = static_cast<float>(dot * scale);
                    sc[t] = val;
                    mx = (val > mx) ? val : mx;
                }
                for (size_t t = L; t < kvlen; ++t) sc[t] = 0.f;

                // softmax
                long double sum = 0.0L;
                for (size_t t = 0; t < L; ++t) {
                    long double e = std::exp(static_cast<long double>(sc[t])
                                             - static_cast<long double>(mx));
                    sc[t] = static_cast<float>(e);
                    sum += e;
                }
                float inv = 1.0f / static_cast<float>(sum);
#pragma omp simd
                for (size_t t = 0; t < L; ++t) sc[t] *= inv;

                // PV (cache-friendly order)
                size_t obase = s * num_head * vdim + h * vdim;
                std::vector<long double> out(vdim, 0.0L); // 可换成分块栈数组

                for (size_t i = 0; i < L; ++i) {
                    long double p = static_cast<long double>(sc[i]);
                    const T *vptr = v + i * num_kv_head * vdim + head_k * vdim;
#pragma omp simd
                    for (size_t t = 0; t < vdim; ++t)
                        out[t] += p * casting(long double, vptr[t]);
                }
                for (size_t t = 0; t < vdim; ++t)
                    attn_val[obase + t]
                        = casting(T, static_cast<float>(out[t]));
            }
        }
    }
}

template <typename T>
static void self_attn_impl_without_openmp(T *attn_val,
                                          const T *q,
                                          const T *k,
                                          const T *v,
                                          size_t seqlen,
                                          size_t num_head,
                                          size_t head_dim,
                                          size_t kvlen,
                                          size_t num_kv_head,
                                          size_t vdim,
                                          float scale) {

    using usize = std::size_t;

    usize ngroup = num_head / num_kv_head;
    for (usize s = 0; s < seqlen; s++) {
        for (usize h = 0; h < num_head; h++) {
            usize head_k = h / ngroup;

            std::vector<float> scores(kvlen, 0);
            for (usize kl = 0; kl < kvlen; kl++) {
                float sum = 0;
                for (usize d = 0; d < head_dim; d++) {
                    usize qbase = s * num_head * head_dim + h * head_dim + d;
                    usize kbase
                        = kl * num_kv_head * head_dim + head_k * head_dim + d;
                    sum += casting(float, q[qbase]) * casting(float, k[kbase]);
                }
                scores[kl] = sum * scale;
            }

            usize L = std::min(kvlen, kvlen - seqlen + s + 1);
            // softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (usize t = 0; t < L; t++)
                if (scores[t] > max_score)
                    max_score = scores[t];

            float sum_exp = 0.0f;
            for (usize t = 0; t < L; t++) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            for (usize t = 0; t < L; t++) scores[t] /= sum_exp;
            for (usize t = L; t < kvlen; t++) scores[t] = 0.0f;

            // PV
            for (usize t = 0; t < vdim; t++) {
                float acc = 0.0f;
                for (usize i = 0; i < L; i++) {
                    usize vbase = i * num_kv_head * vdim + head_k * vdim + t;
                    acc += scores[i] * casting(float, v[vbase]);
                }
                usize obase = s * num_head * vdim + h * vdim + t;
                attn_val[obase] = casting(T, acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {

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
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attn_impl(
            recast(float *, attn_val), recast(const float *, q),
            recast(const float *, k), recast(const float *, v), seqlen,
            num_head, head_dim, kvlen, num_kv_head, vdim, scale);

    case LLAISYS_DTYPE_F16:
        return self_attn_impl(recast(llaisys::fp16_t *, attn_val),
                              recast(const llaisys::fp16_t *, q),
                              recast(const llaisys::fp16_t *, k),
                              recast(const llaisys::fp16_t *, v), seqlen,
                              num_head, head_dim, kvlen, num_kv_head, vdim,
                              scale);

    case LLAISYS_DTYPE_BF16:
        return self_attn_impl(recast(llaisys::bf16_t *, attn_val),
                              recast(const llaisys::bf16_t *, q),
                              recast(const llaisys::bf16_t *, k),
                              recast(const llaisys::bf16_t *, v), seqlen,
                              num_head, head_dim, kvlen, num_kv_head, vdim,
                              scale);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu