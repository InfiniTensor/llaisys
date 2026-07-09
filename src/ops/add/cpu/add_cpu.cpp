#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef __C
    #define LLAISYS_EXTERN_C 1
    #pragma push_macro("__C")
    #undef __C
#endif

#include <immintrin.h>   // AVX2 for vectorized conversions

#ifdef LLAISYS_EXTERN_C
    #undef LLAISYS_EXTERN_C
    #pragma pop_macro("__C")
#endif


template <typename T>
void add_(std::byte *c, const std::byte *a, const std::byte *b, size_t numel);

template<>
void add_<float>(std::byte *c, const std::byte *a, const std::byte *b, size_t numel)
{
    float* c_typed = reinterpret_cast<float*>(c);
    const float* a_typed = reinterpret_cast<const float*>(a);
    const float* b_typed = reinterpret_cast<const float*>(b);

#ifdef __AVX2__

    size_t last = numel - (numel % 8);

    for (size_t i = 0; i < last; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a_typed + i);
        __m256 vb = _mm256_loadu_ps(b_typed + i);

        __m256 vc = _mm256_add_ps(va, vb);

        _mm256_storeu_ps(c_typed + i, vc);
    }

    for (size_t i = last; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#else

    for (size_t i = 0; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#endif
}

template<>
void add_<llaisys::fp16_t>(std::byte *c, const std::byte *a, const std::byte *b, size_t numel)
{
    uint16_t* c16 = reinterpret_cast<uint16_t*>(c);
    const uint16_t* a16 = reinterpret_cast<const uint16_t*>(a);
    const uint16_t* b16 = reinterpret_cast<const uint16_t*>(b);

#if defined(__AVX2__) && defined(__F16C__)

    size_t last = numel - (numel % 16);

    for (size_t i = 0; i < last; i += 16)
    {
        __m256i va = _mm256_loadu_si256((__m256i*)(a16 + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b16 + i));

        __m128i alo = _mm256_extracti128_si256(va,0);
        __m128i ahi = _mm256_extracti128_si256(va,1);

        __m128i blo = _mm256_extracti128_si256(vb,0);
        __m128i bhi = _mm256_extracti128_si256(vb,1);

        __m256 fa0 = _mm256_cvtph_ps(alo);
        __m256 fa1 = _mm256_cvtph_ps(ahi);

        __m256 fb0 = _mm256_cvtph_ps(blo);
        __m256 fb1 = _mm256_cvtph_ps(bhi);

        __m256 fc0 = _mm256_add_ps(fa0,fb0);
        __m256 fc1 = _mm256_add_ps(fa1,fb1);

        __m128i lo = _mm256_cvtps_ph(fc0,0);
        __m128i hi = _mm256_cvtps_ph(fc1,0);

        __m256i packed = _mm256_set_m128i(hi,lo);

        _mm256_storeu_si256((__m256i*)(c16 + i), packed);
    }

    for (size_t i = last; i < numel; i++) {
        reinterpret_cast<llaisys::fp16_t*>(c)[i] = llaisys::utils::cast<llaisys::fp16_t>(
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t*>(a)[i]) +
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t*>(b)[i]));
    }

#else

    for (size_t i = 0; i < numel; i++) {
        reinterpret_cast<llaisys::fp16_t*>(c)[i] = llaisys::utils::cast<llaisys::fp16_t>(
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t*>(a)[i]) +
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t*>(b)[i]));
    }

#endif
}

template<>
void add_<llaisys::bf16_t>(std::byte *c, const std::byte *a, const std::byte *b, size_t numel)
{
    const uint16_t* a16 = reinterpret_cast<const uint16_t*>(a);
    const uint16_t* b16 = reinterpret_cast<const uint16_t*>(b);
    uint16_t* c16 = reinterpret_cast<uint16_t*>(c);

#ifdef __AVX2__

    size_t last = numel - (numel % 16);

#ifdef _OPENMP
#pragma omp parallel for if(numel > 16384)
#endif
    for (size_t i = 0; i < last; i += 16)
    {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a16 + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b16 + i));

        __m128i alo = _mm256_extracti128_si256(va, 0);
        __m128i ahi = _mm256_extracti128_si256(va, 1);

        __m128i blo = _mm256_extracti128_si256(vb, 0);
        __m128i bhi = _mm256_extracti128_si256(vb, 1);

        __m256i alo32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(alo), 16);
        __m256i ahi32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(ahi), 16);

        __m256i blo32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(blo), 16);
        __m256i bhi32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bhi), 16);

        __m256 fa_lo = _mm256_castsi256_ps(alo32);
        __m256 fa_hi = _mm256_castsi256_ps(ahi32);

        __m256 fb_lo = _mm256_castsi256_ps(blo32);
        __m256 fb_hi = _mm256_castsi256_ps(bhi32);

        __m256 fc_lo = _mm256_add_ps(fa_lo, fb_lo);
        __m256 fc_hi = _mm256_add_ps(fa_hi, fb_hi);

        __m256i lo_i = _mm256_castps_si256(fc_lo);
        __m256i hi_i = _mm256_castps_si256(fc_hi);

        __m256i bias = _mm256_set1_epi32(0x7FFF);

        __m256i lsb_lo = _mm256_and_si256(_mm256_srli_epi32(lo_i,16), _mm256_set1_epi32(1));
        __m256i lsb_hi = _mm256_and_si256(_mm256_srli_epi32(hi_i,16), _mm256_set1_epi32(1));

        __m256i round_lo = _mm256_add_epi32(lo_i,_mm256_add_epi32(bias,lsb_lo));
        __m256i round_hi = _mm256_add_epi32(hi_i,_mm256_add_epi32(bias,lsb_hi));

        __m256i shr_lo = _mm256_srli_epi32(round_lo,16);
        __m256i shr_hi = _mm256_srli_epi32(round_hi,16);

        __m128i lo = _mm_packus_epi32(_mm256_extracti128_si256(shr_lo,0),
                                     _mm256_extracti128_si256(shr_lo,1));

        __m128i hi = _mm_packus_epi32(_mm256_extracti128_si256(shr_hi,0),
                                     _mm256_extracti128_si256(shr_hi,1));

        __m256i packed = _mm256_set_m128i(hi,lo);

        _mm256_storeu_si256((__m256i*)(c16 + i), packed);
    }

    for (size_t i = last; i < numel; i++) {
        reinterpret_cast<llaisys::bf16_t*>(c)[i] = llaisys::utils::cast<llaisys::bf16_t>(
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t*>(a)[i]) +
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t*>(b)[i]));
    }

#else
    for (size_t i = 0; i < numel; i++) {
        reinterpret_cast<llaisys::bf16_t*>(c)[i] = llaisys::utils::cast<llaisys::bf16_t>(
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t*>(a)[i]) +
            llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t*>(b)[i]));
    }
#endif
}

template<>
void add_<int32_t>(std::byte *c, const std::byte *a, const std::byte *b, size_t numel)
{
    int32_t* c_typed = reinterpret_cast<int32_t*>(c);
    const int32_t* a_typed = reinterpret_cast<const int32_t*>(a);
    const int32_t* b_typed = reinterpret_cast<const int32_t*>(b);

#ifdef __AVX2__

    size_t last = numel - (numel % 8);

    for (size_t i = 0; i < last; i += 8)
    {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a_typed + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b_typed + i));

        __m256i vc = _mm256_add_epi32(va, vb);

        _mm256_storeu_si256((__m256i*)(c_typed + i), vc);
    }

    for (size_t i = last; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#else

    for (size_t i = 0; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#endif
}

template<>
void add_<double>(std::byte *c, const std::byte *a, const std::byte *b, size_t numel)
{
    double* c_typed = reinterpret_cast<double*>(c);
    const double* a_typed = reinterpret_cast<const double*>(a);
    const double* b_typed = reinterpret_cast<const double*>(b);

#ifdef __AVX2__

    size_t last = numel - (numel % 4);

    for (size_t i = 0; i < last; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a_typed + i);
        __m256d vb = _mm256_loadu_pd(b_typed + i);

        __m256d vc = _mm256_add_pd(va, vb);

        _mm256_storeu_pd(c_typed + i, vc);
    }

    for (size_t i = last; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#else

    for (size_t i = 0; i < numel; i++)
        c_typed[i] = a_typed[i] + b_typed[i];

#endif
}

template <typename T>
void add_(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    T *c_typed = reinterpret_cast<T*>(c);
    const T *a_typed = reinterpret_cast<const T*>(a);
    const T *b_typed = reinterpret_cast<const T*>(b);
    for (size_t i = 0; i < numel; i++) {
        c_typed[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a_typed[i]) + llaisys::utils::cast<float>(b_typed[i]));
    }
}

#define DISPATCH_SWIGLU(dtype, ctype) case dtype: add_<ctype>(c, a, b, numel); break;

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F32, float)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
