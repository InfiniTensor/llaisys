#include "types.hpp"

#include <cstring>
#include <stdexcept>

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

#include <omp.h>

namespace llaisys::utils {

size_t dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return sizeof(char);
    case LLAISYS_DTYPE_BOOL:
        return sizeof(char);
    case LLAISYS_DTYPE_I8:
        return sizeof(int8_t);
    case LLAISYS_DTYPE_I16:
        return sizeof(int16_t);
    case LLAISYS_DTYPE_I32:
        return sizeof(int32_t);
    case LLAISYS_DTYPE_I64:
        return sizeof(int64_t);
    case LLAISYS_DTYPE_U8:
        return sizeof(uint8_t);
    case LLAISYS_DTYPE_U16:
        return sizeof(uint16_t);
    case LLAISYS_DTYPE_U32:
        return sizeof(uint32_t);
    case LLAISYS_DTYPE_U64:
        return sizeof(uint64_t);
    case LLAISYS_DTYPE_F8:
        return sizeof(f8_t);
    case LLAISYS_DTYPE_F16:
        return sizeof(fp16_t);
    case LLAISYS_DTYPE_BF16:
        return sizeof(bf16_t);
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F64:
        return sizeof(double);
    case LLAISYS_DTYPE_C16:
        return sizeof(cp16_t);
    case LLAISYS_DTYPE_C32:
        return sizeof(cp32_t);
    case LLAISYS_DTYPE_C64:
        return sizeof(cp64_t);
    case LLAISYS_DTYPE_C128:
        return sizeof(cp128_t);
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

const char *dtype_to_str(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return "byte";
    case LLAISYS_DTYPE_BOOL:
        return "bool";
    case LLAISYS_DTYPE_I8:
        return "int8";
    case LLAISYS_DTYPE_I16:
        return "int16";
    case LLAISYS_DTYPE_I32:
        return "int32";
    case LLAISYS_DTYPE_I64:
        return "int64";
    case LLAISYS_DTYPE_U8:
        return "uint8";
    case LLAISYS_DTYPE_U16:
        return "uint16";
    case LLAISYS_DTYPE_U32:
        return "uint32";
    case LLAISYS_DTYPE_U64:
        return "uint64";
    case LLAISYS_DTYPE_F8:
        return "float8";
    case LLAISYS_DTYPE_F16:
        return "float16";
    case LLAISYS_DTYPE_BF16:
        return "bfloat16";
    case LLAISYS_DTYPE_F32:
        return "float32";
    case LLAISYS_DTYPE_F64:
        return "float64";
    case LLAISYS_DTYPE_C16:
        return "complex16";
    case LLAISYS_DTYPE_C32:
        return "complex32";
    case LLAISYS_DTYPE_C64:
        return "complex64";
    case LLAISYS_DTYPE_C128:
        return "complex128";
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

void fp16_to_fp32_vec(const uint16_t* src, float* dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    const size_t last_block_start = n - (n % 16);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 16384)
#endif
    for (size_t i = 0; i < last_block_start; i += 16) {
        __m256i vfp16 = _mm256_loadu_si256((const __m256i*)(src + i));
        __m128i lo = _mm256_extracti128_si256(vfp16, 0);
        __m128i hi = _mm256_extracti128_si256(vfp16, 1);
        _mm256_store_ps(dst + i, _mm256_cvtph_ps(lo));
        _mm256_store_ps(dst + i + 8, _mm256_cvtph_ps(hi));
    }
    for (size_t i = last_block_start; i < n; i++) {
        dst[i] = _f16_to_f32(reinterpret_cast<const fp16_t*>(src)[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = _f16_to_f32(reinterpret_cast<const fp16_t*>(src)[i]);
    }
#endif
}

void bf16_to_fp32_vec(const uint16_t* src, float* dst, size_t n) {
#ifdef __AVX2__
    const size_t last_block_start = n - (n % 16);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 16384)
#endif
    for (size_t i = 0; i < last_block_start; i += 16) {
        __m256i vbf16 = _mm256_loadu_si256((const __m256i*)(src + i));
        __m128i lo = _mm256_extracti128_si256(vbf16, 0);
        __m128i hi = _mm256_extracti128_si256(vbf16, 1);
        __m256i lo_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16);
        __m256i hi_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16);
        _mm256_store_ps(dst + i, _mm256_castsi256_ps(lo_32));
        _mm256_store_ps(dst + i + 8, _mm256_castsi256_ps(hi_32));
    }
    for (size_t i = last_block_start; i < n; i++) {
        dst[i] = _bf16_to_f32(reinterpret_cast<const bf16_t*>(src)[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = _bf16_to_f32(reinterpret_cast<const bf16_t*>(src)[i]);
    }
#endif
}

void fp32_to_fp16_vec(const float* src, uint16_t* dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    const size_t last_block_start = n - (n % 16);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 16384)
#endif
    for (size_t i = 0; i < last_block_start; i += 16) {
        __m256 lo = _mm256_load_ps(src + i);
        __m256 hi = _mm256_load_ps(src + i + 8);
        __m128i lo16 = _mm256_cvtps_ph(lo, 0);
        __m128i hi16 = _mm256_cvtps_ph(hi, 0);
        __m256i combined = _mm256_set_m128i(hi16, lo16);
        _mm256_storeu_si256((__m256i*)(dst + i), combined);
    }
    for (size_t i = last_block_start; i < n; i++) {
        dst[i] = _f32_to_f16(src[i])._v;
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = _f32_to_f16(src[i])._v;
    }
#endif
}

void fp32_to_bf16_vec(const float* src, uint16_t* dst, size_t n) {
#ifdef __AVX__
    const size_t last_block_start = n - (n % 8);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 16384)
#endif
    for (size_t i = 0; i < last_block_start; i += 8) {
        __m256 f = _mm256_load_ps(src + i);
        __m256i as_int = _mm256_castps_si256(f);
        __m256i bias = _mm256_set1_epi32(0x7FFF);
        __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(as_int, 16), _mm256_set1_epi32(1));
        __m256i rounding = _mm256_add_epi32(bias, lsb);
        __m256i rounded = _mm256_add_epi32(as_int, rounding);
        __m256i shifted = _mm256_srli_epi32(rounded, 16);

        __m128i lo = _mm256_extracti128_si256(shifted, 0); // 4 x uint32
        __m128i hi = _mm256_extracti128_si256(shifted, 1); // 4 x uint32

        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i*)(dst + i), packed);
    }
    for (size_t i = last_block_start; i < n; i++) {
        dst[i] = _f32_to_bf16(src[i])._v;
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = _f32_to_bf16(src[i])._v;
    }
#endif
}

float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;          // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;            // Extract the mantissa (fraction part)

    if (exponent >= 16) { // Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        // Infinity
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) { // Normalized case
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000; // Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else {
        // Too small for subnormal: return signed zero
        return fp16_t{(uint16_t)sign};
    }
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
} // namespace llaisys::utils
