#include "add_cpu.hpp"

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

#include <cstddef>
#include <type_traits>

namespace {

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2")))
void add_f32_avx2(float *c, const float *a, const float *b, std::ptrdiff_t numel) {
    const std::ptrdiff_t simd_numel = numel - (numel % 8);

#pragma omp parallel for schedule(static) if (numel >= 4096)
    for (std::ptrdiff_t i = 0; i < simd_numel; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }

    for (std::ptrdiff_t i = simd_numel; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }
}
#endif

void add_f32(float *c, const float *a, const float *b, std::ptrdiff_t numel) {
#pragma omp parallel for schedule(static) if (numel >= 4096)
    for (std::ptrdiff_t i = 0; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void add_generic(T *c, const T *a, const T *b, std::ptrdiff_t numel) {
#pragma omp parallel for schedule(static) if (numel >= 4096)
    for (std::ptrdiff_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    const auto elem_count = static_cast<std::ptrdiff_t>(numel);
    switch (type) {
    case LLAISYS_DTYPE_F32:
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2()) {
            return add_f32_avx2(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), elem_count);
        }
#endif
        return add_f32(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), elem_count);
    case LLAISYS_DTYPE_BF16:
        return add_generic(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                           reinterpret_cast<const llaisys::bf16_t *>(b), elem_count);
    case LLAISYS_DTYPE_F16:
        return add_generic(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                           reinterpret_cast<const llaisys::fp16_t *>(b), elem_count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
