#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
static inline float avx2_dot(const float *a, const float *b, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}
#endif

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t qlen, size_t kvlen,
                     size_t nh, size_t nkvh, size_t d) {
    size_t group_size = nh / nkvh;

    bool need_cast = !std::is_same<T, float>::value;

    std::vector<float> fq, fk, fv;
    if (need_cast) {
        fq.resize(qlen * nh * d);
        fk.resize(kvlen * nkvh * d);
        fv.resize(kvlen * nkvh * d);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < qlen * nh * d; i++)
            fq[i] = llaisys::utils::cast<float>(q[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < kvlen * nkvh * d; i++)
            fk[i] = llaisys::utils::cast<float>(k[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < kvlen * nkvh * d; i++)
            fv[i] = llaisys::utils::cast<float>(v[i]);
    }

    const float *qf = need_cast ? fq.data() : reinterpret_cast<const float *>(q);
    const float *kf = need_cast ? fk.data() : reinterpret_cast<const float *>(k);
    const float *vf = need_cast ? fv.data() : reinterpret_cast<const float *>(v);

    #pragma omp parallel for schedule(dynamic)
    for (size_t h = 0; h < nh; h++) {
        size_t kvh = h / group_size;

        std::vector<float> scores(qlen * kvlen);

        for (size_t qi = 0; qi < qlen; qi++) {
            const float *qrow = qf + (qi * nh + h) * d;
            for (size_t ki = 0; ki < kvlen; ki++) {
                const float *krow = kf + (ki * nkvh + kvh) * d;
#ifdef __AVX2__
                scores[qi * kvlen + ki] = avx2_dot(qrow, krow, d) * scale;
#else
                float dot = 0.0f;
                for (size_t di = 0; di < d; di++)
                    dot += qrow[di] * krow[di];
                scores[qi * kvlen + ki] = dot * scale;
#endif
            }
        }

        for (size_t qi = 0; qi < qlen; qi++) {
            size_t max_ki = qi + (kvlen - qlen);

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki <= max_ki && ki < kvlen; ki++)
                max_score = std::max(max_score, scores[qi * kvlen + ki]);

            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (ki <= max_ki) {
                    scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                    sum_exp += scores[qi * kvlen + ki];
                } else {
                    scores[qi * kvlen + ki] = 0.0f;
                }
            }

            float inv_sum = 1.0f / sum_exp;
            for (size_t ki = 0; ki < kvlen; ki++)
                scores[qi * kvlen + ki] *= inv_sum;
        }

        for (size_t qi = 0; qi < qlen; qi++) {
            for (size_t di = 0; di < d; di++) {
                float sum = 0.0f;
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                size_t ki = 0;
                for (; ki + 8 <= kvlen; ki += 8) {
                    __m256 vs = _mm256_loadu_ps(&scores[qi * kvlen + ki]);
                    // Gather v values: v[(ki+j)*nkvh+kvh]*d+di for j=0..7
                    // Manual gather since stride is non-trivial
                    float vvals[8];
                    for (size_t j = 0; j < 8; j++)
                        vvals[j] = vf[((ki + j) * nkvh + kvh) * d + di];
                    __m256 vv = _mm256_loadu_ps(vvals);
                    vsum = _mm256_fmadd_ps(vs, vv, vsum);
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, vsum);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                      tmp[4] + tmp[5] + tmp[6] + tmp[7];
                for (; ki < kvlen; ki++)
                    sum += scores[qi * kvlen + ki] * vf[(ki * nkvh + kvh) * d + di];
#else
                for (size_t ki = 0; ki < kvlen; ki++)
                    sum += scores[qi * kvlen + ki] * vf[(ki * nkvh + kvh) * d + di];
#endif
                if (need_cast)
                    attn_val[(qi * nh + h) * d + di] = llaisys::utils::cast<T>(sum);
                else
                    reinterpret_cast<float *>(attn_val)[(qi * nh + h) * d + di] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                                reinterpret_cast<const float *>(q),
                                reinterpret_cast<const float *>(k),
                                reinterpret_cast<const float *>(v),
                                scale, qlen, kvlen, nh, nkvh, d);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                reinterpret_cast<const llaisys::bf16_t *>(q),
                                reinterpret_cast<const llaisys::bf16_t *>(k),
                                reinterpret_cast<const llaisys::bf16_t *>(v),
                                scale, qlen, kvlen, nh, nkvh, d);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                reinterpret_cast<const llaisys::fp16_t *>(q),
                                reinterpret_cast<const llaisys::fp16_t *>(k),
                                reinterpret_cast<const llaisys::fp16_t *>(v),
                                scale, qlen, kvlen, nh, nkvh, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
