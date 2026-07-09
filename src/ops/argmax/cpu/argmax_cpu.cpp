#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    float best = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        if (numel >= 8) {
            __m256 vbest = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            __m256i vidx = _mm256_setzero_si256();
            __m256i vcur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i vinc = _mm256_set1_epi32(8);

            size_t i = 0;
            for (; i + 8 <= numel; i += 8) {
                __m256 vv = _mm256_loadu_ps(vals + i);
                __m256 mask = _mm256_cmp_ps(vv, vbest, _CMP_GT_OQ);
                vbest = _mm256_blendv_ps(vbest, vv, mask);
                vidx = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(vidx), _mm256_castsi256_ps(vcur), mask));
                vcur = _mm256_add_epi32(vcur, vinc);
            }

            float bests[8];
            int32_t idxs[8];
            _mm256_storeu_ps(bests, vbest);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(idxs), vidx);

            for (int j = 0; j < 8; j++) {
                if (bests[j] > best) {
                    best = bests[j];
                    best_idx = idxs[j];
                }
            }

            for (; i < numel; i++) {
                if (vals[i] > best) {
                    best = vals[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }

            *max_idx = best_idx;
            *max_val = static_cast<T>(best);
            return;
        }
    }
#endif

    for (size_t i = 0; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best) {
            best = v;
            best_idx = static_cast<int64_t>(i);
        }
    }
    *max_idx = best_idx;
    *max_val = llaisys::utils::cast<T>(best);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    auto *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
