#include "self_attention_cpu.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#ifdef __C
#pragma push_macro("__C")
#undef __C
#define LLAISYS_RESTORE_C_MACRO
#endif
#include <immintrin.h>
#ifdef LLAISYS_RESTORE_C_MACRO
#pragma pop_macro("__C")
#undef LLAISYS_RESTORE_C_MACRO
#endif
#endif

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2_fma() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

__attribute__((target("avx2,fma,sse3")))
inline float hsum256_ps(__m256 v) {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
float dot_f32_avx2(const float *a, const float *b, std::ptrdiff_t numel) {
    const std::ptrdiff_t simd_numel = numel - (numel % 8);
    __m256 acc = _mm256_setzero_ps();
    for (std::ptrdiff_t i = 0; i < simd_numel; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }

    float sum = hsum256_ps(acc);
    for (std::ptrdiff_t i = simd_numel; i < numel; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__attribute__((target("avx2,fma")))
void fill_zero_f32_avx2(float *dst, std::ptrdiff_t numel) {
    const std::ptrdiff_t simd_numel = numel - (numel % 8);
    const __m256 zero = _mm256_setzero_ps();
    std::ptrdiff_t i = 0;
    for (; i < simd_numel; i += 8) {
        _mm256_storeu_ps(dst + i, zero);
    }
    for (; i < numel; ++i) {
        dst[i] = 0.0f;
    }
}

__attribute__((target("avx2,fma")))
void axpy_f32_avx2(float *dst, const float *src, float alpha, std::ptrdiff_t numel) {
    const std::ptrdiff_t simd_numel = numel - (numel % 8);
    const __m256 scale = _mm256_set1_ps(alpha);
    std::ptrdiff_t i = 0;
    for (; i < simd_numel; i += 8) {
        const __m256 cur = _mm256_loadu_ps(dst + i);
        const __m256 x = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(scale, x, cur));
    }
    for (; i < numel; ++i) {
        dst[i] += alpha * src[i];
    }
}

__attribute__((target("avx2,fma")))
void scale_f32_avx2(float *dst, float alpha, std::ptrdiff_t numel) {
    const std::ptrdiff_t simd_numel = numel - (numel % 8);
    const __m256 scale = _mm256_set1_ps(alpha);
    std::ptrdiff_t i = 0;
    for (; i < simd_numel; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), scale));
    }
    for (; i < numel; ++i) {
        dst[i] *= alpha;
    }
}

__attribute__((target("avx2,fma")))
void self_attention_decode_f32_avx2(float *attn_val, const float *q, const float *k, const float *v, float scale,
                                    const std::vector<size_t> &q_shape,
                                    const std::vector<size_t> &k_shape,
                                    const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_heads = nh * sk * d >= 32768;

#pragma omp parallel for schedule(static) if (parallel_heads)
    for (std::ptrdiff_t h = 0; h < nh; ++h) {
        const std::ptrdiff_t h_kv = h / n_rep;
        const float *q_vec = q + h * d;
        float *out_ptr = attn_val + h * dv;
        fill_zero_f32_avx2(out_ptr, dv);

        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;
        for (std::ptrdiff_t j = 0; j < sk; ++j) {
            const float *k_vec = k + (j * nh_kv + h_kv) * d;
            const float *v_vec = v + (j * nh_kv + h_kv) * dv;
            const float score = dot_f32_avx2(q_vec, k_vec, d) * scale;

            if (score > max_score) {
                const float rescale = std::exp(max_score - score);
                if (sum_exp > 0.0f) {
                    scale_f32_avx2(out_ptr, rescale, dv);
                }
                sum_exp = sum_exp * rescale + 1.0f;
                max_score = score;
                axpy_f32_avx2(out_ptr, v_vec, 1.0f, dv);
            } else {
                const float weight = std::exp(score - max_score);
                sum_exp += weight;
                axpy_f32_avx2(out_ptr, v_vec, weight, dv);
            }
        }

        if (sum_exp > 0.0f) {
            scale_f32_avx2(out_ptr, 1.0f / sum_exp, dv);
        }
    }
}

__attribute__((target("avx2,fma")))
void self_attention_f32_avx2(float *attn_val, const float *q, const float *k, const float *v, float scale,
                             const std::vector<size_t> &q_shape,
                             const std::vector<size_t> &k_shape,
                             const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t sq = static_cast<std::ptrdiff_t>(q_shape[0]);
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_work = sq * nh * sk * d >= 32768;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_work)
    for (std::ptrdiff_t i = 0; i < sq; ++i) {
        for (std::ptrdiff_t h = 0; h < nh; ++h) {
            const std::ptrdiff_t h_kv = h / n_rep;
            const std::ptrdiff_t q_abs_pos = sk - sq + i;
            const std::ptrdiff_t valid_len = std::min(sk, q_abs_pos + 1);
            const float *q_vec = q + (i * nh + h) * d;
            float *out_ptr = attn_val + (i * nh + h) * dv;

            if (valid_len == 1) {
                const float *v_vec = v + h_kv * dv;
                std::copy(v_vec, v_vec + dv, out_ptr);
                continue;
            }

            std::vector<float> scores(static_cast<size_t>(valid_len));
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const float *k_vec = k + (j * nh_kv + h_kv) * d;
                const float score = dot_f32_avx2(q_vec, k_vec, d) * scale;
                scores[static_cast<size_t>(j)] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;
            for (float &score : scores) {
                score = std::exp(score - max_score);
                sum_exp += score;
            }
            const float inv_sum = 1.0f / (sum_exp + 1e-10f);

            fill_zero_f32_avx2(out_ptr, dv);
            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const float weight = scores[static_cast<size_t>(j)] * inv_sum;
                if (weight < 1e-10f) {
                    continue;
                }
                const float *v_vec = v + (j * nh_kv + h_kv) * dv;
                axpy_f32_avx2(out_ptr, v_vec, weight, dv);
            }
        }
    }
}
#endif

void scale_f32(float *dst, float alpha, std::ptrdiff_t numel) {
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < numel; ++i) {
        dst[i] *= alpha;
    }
}

void self_attention_decode_f32(float *attn_val, const float *q, const float *k, const float *v, float scale,
                               const std::vector<size_t> &q_shape,
                               const std::vector<size_t> &k_shape,
                               const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_heads = nh * sk * d >= 32768;

#pragma omp parallel for schedule(static) if (parallel_heads)
    for (std::ptrdiff_t h = 0; h < nh; ++h) {
        const std::ptrdiff_t h_kv = h / n_rep;
        const float *q_vec = q + h * d;
        float *out_ptr = attn_val + h * dv;

#pragma omp simd
        for (std::ptrdiff_t l = 0; l < dv; ++l) {
            out_ptr[l] = 0.0f;
        }

        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;
        for (std::ptrdiff_t j = 0; j < sk; ++j) {
            const float *k_vec = k + (j * nh_kv + h_kv) * d;
            const float *v_vec = v + (j * nh_kv + h_kv) * dv;
            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (std::ptrdiff_t l = 0; l < d; ++l) {
                dot += q_vec[l] * k_vec[l];
            }
            const float score = dot * scale;

            if (score > max_score) {
                const float rescale = std::exp(max_score - score);
                if (sum_exp > 0.0f) {
                    scale_f32(out_ptr, rescale, dv);
                }
                sum_exp = sum_exp * rescale + 1.0f;
                max_score = score;
#pragma omp simd
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_ptr[l] += v_vec[l];
                }
            } else {
                const float weight = std::exp(score - max_score);
                sum_exp += weight;
#pragma omp simd
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_ptr[l] += weight * v_vec[l];
                }
            }
        }

        if (sum_exp > 0.0f) {
            scale_f32(out_ptr, 1.0f / sum_exp, dv);
        }
    }
}

void self_attention_f32(float *attn_val, const float *q, const float *k, const float *v, float scale,
                        const std::vector<size_t> &q_shape,
                        const std::vector<size_t> &k_shape,
                        const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t sq = static_cast<std::ptrdiff_t>(q_shape[0]);
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_work = sq * nh * sk * d >= 32768;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_work)
    for (std::ptrdiff_t i = 0; i < sq; ++i) {
        for (std::ptrdiff_t h = 0; h < nh; ++h) {
            const std::ptrdiff_t h_kv = h / n_rep;
            const std::ptrdiff_t q_abs_pos = sk - sq + i;
            const std::ptrdiff_t valid_len = std::min(sk, q_abs_pos + 1);
            const float *q_vec = q + (i * nh + h) * d;
            float *out_ptr = attn_val + (i * nh + h) * dv;

            if (valid_len == 1) {
                const float *v_vec = v + h_kv * dv;
                std::copy(v_vec, v_vec + dv, out_ptr);
                continue;
            }

            std::vector<float> scores(static_cast<size_t>(valid_len));
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const float *k_vec = k + (j * nh_kv + h_kv) * d;
                float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
                for (std::ptrdiff_t l = 0; l < d; ++l) {
                    dot += q_vec[l] * k_vec[l];
                }
                const float score = dot * scale;
                scores[static_cast<size_t>(j)] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;
            for (float &score : scores) {
                score = std::exp(score - max_score);
                sum_exp += score;
            }
            const float inv_sum = 1.0f / (sum_exp + 1e-10f);

#pragma omp simd
            for (std::ptrdiff_t l = 0; l < dv; ++l) {
                out_ptr[l] = 0.0f;
            }

            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const float weight = scores[static_cast<size_t>(j)] * inv_sum;
                if (weight < 1e-10f) {
                    continue;
                }
                const float *v_vec = v + (j * nh_kv + h_kv) * dv;
#pragma omp simd
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_ptr[l] += weight * v_vec[l];
                }
            }
        }
    }
}

template <typename T>
void self_attention_decode_generic(T *attn_val, const T *q, const T *k, const T *v, float scale,
                                   const std::vector<size_t> &q_shape,
                                   const std::vector<size_t> &k_shape,
                                   const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_heads = nh * sk * d >= 32768;

#pragma omp parallel for schedule(static) if (parallel_heads)
    for (std::ptrdiff_t h = 0; h < nh; ++h) {
        const std::ptrdiff_t h_kv = h / n_rep;
        const T *q_vec = q + h * d;
        std::vector<float> out_accum(static_cast<size_t>(dv), 0.0f);
        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        for (std::ptrdiff_t j = 0; j < sk; ++j) {
            const T *k_vec = k + (j * nh_kv + h_kv) * d;
            const T *v_vec = v + (j * nh_kv + h_kv) * dv;
            float dot = 0.0f;
            for (std::ptrdiff_t l = 0; l < d; ++l) {
                dot += llaisys::utils::cast<float>(q_vec[l]) * llaisys::utils::cast<float>(k_vec[l]);
            }
            const float score = dot * scale;

            if (score > max_score) {
                const float rescale = std::exp(max_score - score);
                if (sum_exp > 0.0f) {
                    for (std::ptrdiff_t l = 0; l < dv; ++l) {
                        out_accum[static_cast<size_t>(l)] *= rescale;
                    }
                }
                sum_exp = sum_exp * rescale + 1.0f;
                max_score = score;
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_accum[static_cast<size_t>(l)] += llaisys::utils::cast<float>(v_vec[l]);
                }
            } else {
                const float weight = std::exp(score - max_score);
                sum_exp += weight;
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_accum[static_cast<size_t>(l)] += weight * llaisys::utils::cast<float>(v_vec[l]);
                }
            }
        }

        T *out_ptr = attn_val + h * dv;
        const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (std::ptrdiff_t l = 0; l < dv; ++l) {
            out_ptr[l] = llaisys::utils::cast<T>(out_accum[static_cast<size_t>(l)] * inv_sum);
        }
    }
}

template <typename T>
void self_attention_generic(T *attn_val, const T *q, const T *k, const T *v, float scale,
                            const std::vector<size_t> &q_shape,
                            const std::vector<size_t> &k_shape,
                            const std::vector<size_t> &v_shape) {
    const std::ptrdiff_t sq = static_cast<std::ptrdiff_t>(q_shape[0]);
    const std::ptrdiff_t nh = static_cast<std::ptrdiff_t>(q_shape[1]);
    const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(q_shape[2]);
    const std::ptrdiff_t sk = static_cast<std::ptrdiff_t>(k_shape[0]);
    const std::ptrdiff_t nh_kv = static_cast<std::ptrdiff_t>(k_shape[1]);
    const std::ptrdiff_t dv = static_cast<std::ptrdiff_t>(v_shape[2]);
    const std::ptrdiff_t n_rep = nh / nh_kv;
    const bool parallel_work = sq * nh * sk * d >= 32768;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_work)
    for (std::ptrdiff_t i = 0; i < sq; ++i) {
        for (std::ptrdiff_t h = 0; h < nh; ++h) {
            const std::ptrdiff_t h_kv = h / n_rep;
            const std::ptrdiff_t q_abs_pos = sk - sq + i;
            const std::ptrdiff_t valid_len = std::min(sk, q_abs_pos + 1);
            const T *q_vec = q + (i * nh + h) * d;
            std::vector<float> scores(static_cast<size_t>(valid_len));
            float max_score = -std::numeric_limits<float>::infinity();

            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const T *k_vec = k + (j * nh_kv + h_kv) * d;
                float dot = 0.0f;
                for (std::ptrdiff_t l = 0; l < d; ++l) {
                    dot += llaisys::utils::cast<float>(q_vec[l]) * llaisys::utils::cast<float>(k_vec[l]);
                }
                const float score = dot * scale;
                scores[static_cast<size_t>(j)] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;
            for (float &score : scores) {
                score = std::exp(score - max_score);
                sum_exp += score;
            }
            const float inv_sum = 1.0f / (sum_exp + 1e-10f);

            std::vector<float> out_accum(static_cast<size_t>(dv), 0.0f);
            for (std::ptrdiff_t j = 0; j < valid_len; ++j) {
                const float weight = scores[static_cast<size_t>(j)] * inv_sum;
                if (weight < 1e-10f) {
                    continue;
                }

                const T *v_vec = v + (j * nh_kv + h_kv) * dv;
                for (std::ptrdiff_t l = 0; l < dv; ++l) {
                    out_accum[static_cast<size_t>(l)] += weight * llaisys::utils::cast<float>(v_vec[l]);
                }
            }

            T *out_ptr = attn_val + (i * nh + h) * dv;
            for (std::ptrdiff_t l = 0; l < dv; ++l) {
                out_ptr[l] = llaisys::utils::cast<T>(out_accum[static_cast<size_t>(l)]);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t dtype,
                    const std::vector<size_t> &q_shape,
                    const std::vector<size_t> &k_shape,
                    const std::vector<size_t> &v_shape) {
    const bool single_query = q_shape[0] == 1;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2_fma()) {
            if (single_query) {
                return self_attention_decode_f32_avx2(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, q_shape, k_shape, v_shape);
            }
            return self_attention_f32_avx2(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, q_shape, k_shape, v_shape);
        }
#endif
        if (single_query) {
            return self_attention_decode_f32(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, q_shape, k_shape, v_shape);
        }
        return self_attention_f32(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_BF16:
        if (single_query) {
            return self_attention_decode_generic(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, q_shape, k_shape, v_shape);
        }
        return self_attention_generic(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_F16:
        if (single_query) {
            return self_attention_decode_generic(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, q_shape, k_shape, v_shape);
        }
        return self_attention_generic(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, q_shape, k_shape, v_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
