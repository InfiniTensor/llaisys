#include "argmax_cpu.hpp"

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
#include <type_traits>

namespace {

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2")))
void argmax_f32_avx2(size_t *max_idx, float *max_val, const float *vals, std::ptrdiff_t numel) {
    if (numel <= 0) {
        *max_idx = 0;
        *max_val = 0.0f;
        return;
    }

    std::ptrdiff_t i = 0;
    float best_val = vals[0];
    size_t best_idx = 0;

    if (numel >= 8) {
        __m256 max_vals = _mm256_loadu_ps(vals);
        __m256i max_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i cur_indices = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        const __m256i step = _mm256_set1_epi32(8);
        i = 8;

        for (; i + 8 <= numel; i += 8) {
            const __m256 cur_vals = _mm256_loadu_ps(vals + i);
            const __m256 mask = _mm256_cmp_ps(cur_vals, max_vals, _CMP_GT_OQ);
            max_vals = _mm256_blendv_ps(max_vals, cur_vals, mask);
            max_indices = _mm256_blendv_epi8(max_indices, cur_indices, _mm256_castps_si256(mask));
            cur_indices = _mm256_add_epi32(cur_indices, step);
        }

        alignas(32) float lane_vals[8];
        alignas(32) int lane_indices[8];
        _mm256_store_ps(lane_vals, max_vals);
        _mm256_store_si256(reinterpret_cast<__m256i *>(lane_indices), max_indices);
        for (int lane = 0; lane < 8; ++lane) {
            if (lane_vals[lane] > best_val) {
                best_val = lane_vals[lane];
                best_idx = static_cast<size_t>(lane_indices[lane]);
            }
        }
    }

    for (; i < numel; ++i) {
        if (vals[i] > best_val) {
            best_val = vals[i];
            best_idx = static_cast<size_t>(i);
        }
    }

    *max_idx = best_idx;
    *max_val = best_val;
}
#endif

template <typename T>
void argmax_generic(size_t *max_idx, T *max_val, const T *vals, std::ptrdiff_t numel){
    size_t max_index = 0;
    float max_value = llaisys::utils::cast<float>(vals[0]);

    std::ptrdiff_t i = 1;
    for (; i + 3 < numel; i += 4) {
        const float v0 = llaisys::utils::cast<float>(vals[i + 0]);
        const float v1 = llaisys::utils::cast<float>(vals[i + 1]);
        const float v2 = llaisys::utils::cast<float>(vals[i + 2]);
        const float v3 = llaisys::utils::cast<float>(vals[i + 3]);
        if (v0 > max_value) {
            max_value = v0;
            max_index = static_cast<size_t>(i + 0);
        }
        if (v1 > max_value) {
            max_value = v1;
            max_index = static_cast<size_t>(i + 1);
        }
        if (v2 > max_value) {
            max_value = v2;
            max_index = static_cast<size_t>(i + 2);
        }
        if (v3 > max_value) {
            max_value = v3;
            max_index = static_cast<size_t>(i + 3);
        }
    }

    for (; i < numel; ++i) {
        const float current_value = llaisys::utils::cast<float>(vals[i]);
        if (current_value > max_value) {
            max_value = current_value;
            max_index = static_cast<size_t>(i);
        }
    }

    *max_idx = max_index;
    *max_val = llaisys::utils::cast<T>(max_value);
}

} // namespace

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel){
    const auto elem_count = static_cast<std::ptrdiff_t>(numel);
    switch (type) {
    case LLAISYS_DTYPE_F32:
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2()) {
            return argmax_f32_avx2(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), elem_count);
        }
#endif
        return argmax_generic(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), elem_count);
    case LLAISYS_DTYPE_BF16:
        return argmax_generic(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), elem_count);
    case LLAISYS_DTYPE_F16:
        return argmax_generic(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), elem_count);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
