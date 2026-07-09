#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel - (numel % 8); i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
        }
        for (size_t i = numel - (numel % 8); i < numel; i++) {
            c[i] = a[i] + b[i];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
