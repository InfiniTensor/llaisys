#include "rope_cpu.hpp"

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
#include <cstddef>
#include <cmath>

#include <vector>

namespace {

std::vector<double> build_inv_freq(float theta, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    std::vector<double> inv_freq(static_cast<size_t>(half_dim));
    for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
        const double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(head_dim);
        inv_freq[static_cast<size_t>(j)] = std::pow(static_cast<double>(theta), -exponent);
    }
    return inv_freq;
}

const std::vector<double> &get_inv_freq(float theta, std::ptrdiff_t head_dim) {
    static thread_local std::vector<double> cached_inv_freq;
    static thread_local float cached_theta = 0.0f;
    static thread_local std::ptrdiff_t cached_head_dim = 0;
    if (cached_inv_freq.empty() || cached_theta != theta || cached_head_dim != head_dim) {
        cached_inv_freq = build_inv_freq(theta, head_dim);
        cached_theta = theta;
        cached_head_dim = head_dim;
    }
    return cached_inv_freq;
}

void build_trig_tables(std::vector<float> &sin_table, std::vector<float> &cos_table, const int64_t *pos_ids, const std::vector<double> &inv_freq, std::ptrdiff_t seq_len, std::ptrdiff_t half_dim) {
    const bool parallel_trig = seq_len * half_dim >= 4096;
#pragma omp parallel for schedule(static) if (parallel_trig)
    for (std::ptrdiff_t s = 0; s < seq_len; ++s) {
        const double pos = static_cast<double>(pos_ids[s]);
        float *sin_row = sin_table.data() + s * half_dim;
        float *cos_row = cos_table.data() + s * half_dim;
        for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
            const double angle = pos * inv_freq[static_cast<size_t>(j)];
            sin_row[j] = static_cast<float>(std::sin(angle));
            cos_row[j] = static_cast<float>(std::cos(angle));
        }
    }
}

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2_fma() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

__attribute__((target("avx2,fma")))
void apply_rope_f32_avx2(float *out, const float *in, const std::vector<float> &sin_table, const std::vector<float> &cos_table, std::ptrdiff_t seq_len, std::ptrdiff_t num_heads, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    const std::ptrdiff_t simd_half_dim = half_dim - (half_dim % 8);
    const bool parallel_apply = seq_len * num_heads * half_dim >= 4096;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_apply)
    for (std::ptrdiff_t s = 0; s < seq_len; ++s) {
        for (std::ptrdiff_t h = 0; h < num_heads; ++h) {
            const std::ptrdiff_t offset = s * num_heads * head_dim + h * head_dim;
            const float *src_vec = in + offset;
            float *dst_vec = out + offset;
            const float *sin_row = sin_table.data() + s * half_dim;
            const float *cos_row = cos_table.data() + s * half_dim;

            std::ptrdiff_t j = 0;
            for (; j < simd_half_dim; j += 8) {
                const __m256 a = _mm256_loadu_ps(src_vec + j);
                const __m256 b = _mm256_loadu_ps(src_vec + half_dim + j);
                const __m256 sin_v = _mm256_loadu_ps(sin_row + j);
                const __m256 cos_v = _mm256_loadu_ps(cos_row + j);
                const __m256 a_rot = _mm256_fmsub_ps(a, cos_v, _mm256_mul_ps(b, sin_v));
                const __m256 b_rot = _mm256_fmadd_ps(a, sin_v, _mm256_mul_ps(b, cos_v));
                _mm256_storeu_ps(dst_vec + j, a_rot);
                _mm256_storeu_ps(dst_vec + half_dim + j, b_rot);
            }

            for (; j < half_dim; ++j) {
                const float sin_v = sin_row[j];
                const float cos_v = cos_row[j];
                const float a = src_vec[j];
                const float b = src_vec[half_dim + j];
                dst_vec[j] = a * cos_v - b * sin_v;
                dst_vec[half_dim + j] = b * cos_v + a * sin_v;
            }
        }
    }
}
#endif

void rope_f32(float *out, const float *in, const int64_t *pos_ids, float theta, std::ptrdiff_t seq_len, std::ptrdiff_t num_heads, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    std::vector<float> sin_table(static_cast<size_t>(seq_len * half_dim));
    std::vector<float> cos_table(static_cast<size_t>(seq_len * half_dim));
    const auto &inv_freq = get_inv_freq(theta, head_dim);
    build_trig_tables(sin_table, cos_table, pos_ids, inv_freq, seq_len, half_dim);
    const bool parallel_apply = seq_len * num_heads * half_dim >= 4096;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_apply)
    for (std::ptrdiff_t s = 0; s < seq_len; ++s) {
        for (std::ptrdiff_t h = 0; h < num_heads; ++h) {
            const std::ptrdiff_t offset = s * num_heads * head_dim + h * head_dim;
            const float *src_vec = in + offset;
            float *dst_vec = out + offset;
            const float *sin_row = sin_table.data() + s * half_dim;
            const float *cos_row = cos_table.data() + s * half_dim;

#pragma omp simd
            for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
                const float a = src_vec[j];
                const float b = src_vec[half_dim + j];
                dst_vec[j] = a * cos_row[j] - b * sin_row[j];
                dst_vec[half_dim + j] = b * cos_row[j] + a * sin_row[j];
            }
        }
    }
}

void rope_single_f32(float *out, const float *in, const int64_t *pos_ids, float theta, std::ptrdiff_t num_heads, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    const auto &inv_freq = get_inv_freq(theta, head_dim);
    const double pos = static_cast<double>(pos_ids[0]);
    const bool parallel_heads = num_heads * half_dim >= 4096;

#pragma omp parallel for schedule(static) if (parallel_heads)
    for (std::ptrdiff_t h = 0; h < num_heads; ++h) {
        const std::ptrdiff_t offset = h * head_dim;
        const float *src_vec = in + offset;
        float *dst_vec = out + offset;
        for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
            const double angle = pos * inv_freq[static_cast<size_t>(j)];
            const float sin_v = static_cast<float>(std::sin(angle));
            const float cos_v = static_cast<float>(std::cos(angle));
            const float a = src_vec[j];
            const float b = src_vec[half_dim + j];
            dst_vec[j] = a * cos_v - b * sin_v;
            dst_vec[half_dim + j] = b * cos_v + a * sin_v;
        }
    }
}

template <typename T>
void rope_generic(T *out, const T *in, const int64_t *pos_ids, float theta, std::ptrdiff_t seq_len, std::ptrdiff_t num_heads, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    std::vector<float> sin_table(static_cast<size_t>(seq_len * half_dim));
    std::vector<float> cos_table(static_cast<size_t>(seq_len * half_dim));
    const auto &inv_freq = get_inv_freq(theta, head_dim);
    build_trig_tables(sin_table, cos_table, pos_ids, inv_freq, seq_len, half_dim);
    const bool parallel_apply = seq_len * num_heads * half_dim >= 4096;

#pragma omp parallel for collapse(2) schedule(static) if (parallel_apply)
    for (std::ptrdiff_t s = 0; s < seq_len; ++s) {
        for (std::ptrdiff_t h = 0; h < num_heads; ++h) {
            const std::ptrdiff_t offset = s * num_heads * head_dim + h * head_dim;
            const T *src_vec = in + offset;
            T *dst_vec = out + offset;
            const float *sin_row = sin_table.data() + s * half_dim;
            const float *cos_row = cos_table.data() + s * half_dim;

            for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
                const float a = llaisys::utils::cast<float>(src_vec[j]);
                const float b = llaisys::utils::cast<float>(src_vec[half_dim + j]);
                dst_vec[j] = llaisys::utils::cast<T>(a * cos_row[j] - b * sin_row[j]);
                dst_vec[half_dim + j] = llaisys::utils::cast<T>(b * cos_row[j] + a * sin_row[j]);
            }
        }
    }
}

template <typename T>
void rope_single_generic(T *out, const T *in, const int64_t *pos_ids, float theta, std::ptrdiff_t num_heads, std::ptrdiff_t head_dim) {
    const std::ptrdiff_t half_dim = head_dim / 2;
    const auto &inv_freq = get_inv_freq(theta, head_dim);
    const double pos = static_cast<double>(pos_ids[0]);
    const bool parallel_heads = num_heads * half_dim >= 4096;

#pragma omp parallel for schedule(static) if (parallel_heads)
    for (std::ptrdiff_t h = 0; h < num_heads; ++h) {
        const std::ptrdiff_t offset = h * head_dim;
        const T *src_vec = in + offset;
        T *dst_vec = out + offset;
        for (std::ptrdiff_t j = 0; j < half_dim; ++j) {
            const double angle = pos * inv_freq[static_cast<size_t>(j)];
            const float sin_v = static_cast<float>(std::sin(angle));
            const float cos_v = static_cast<float>(std::cos(angle));
            const float a = llaisys::utils::cast<float>(src_vec[j]);
            const float b = llaisys::utils::cast<float>(src_vec[half_dim + j]);
            dst_vec[j] = llaisys::utils::cast<T>(a * cos_v - b * sin_v);
            dst_vec[half_dim + j] = llaisys::utils::cast<T>(b * cos_v + a * sin_v);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t dtype, const std::vector<size_t> &shape) {
    const std::ptrdiff_t seq_len = static_cast<std::ptrdiff_t>(shape[0]);
    const std::ptrdiff_t num_heads = static_cast<std::ptrdiff_t>(shape[1]);
    const std::ptrdiff_t head_dim = static_cast<std::ptrdiff_t>(shape[2]);

    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        if (seq_len == 1) {
            return rope_single_f32(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ptr, theta, num_heads, head_dim);
        }
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2_fma()) {
            std::vector<float> sin_table(static_cast<size_t>(seq_len * (head_dim / 2)));
            std::vector<float> cos_table(static_cast<size_t>(seq_len * (head_dim / 2)));
            const auto &inv_freq = get_inv_freq(theta, head_dim);
            build_trig_tables(sin_table, cos_table, pos_ptr, inv_freq, seq_len, head_dim / 2);
            return apply_rope_f32_avx2(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), sin_table, cos_table, seq_len, num_heads, head_dim);
        }
#endif
        rope_f32(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        if (seq_len == 1) {
            return rope_single_generic(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos_ptr, theta, num_heads, head_dim);
        }
        rope_generic(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        if (seq_len == 1) {
            return rope_single_generic(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos_ptr, theta, num_heads, head_dim);
        }
        rope_generic(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
