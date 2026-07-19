#pragma once
#include <immintrin.h>
#include <cstdint>

namespace llaisys::utils {

// ============================================================
// AVX2 SIMD 辅助函数
// 供各个 op 的 AVX2 特化共用
// ============================================================

// AVX2 水平求和: 8 个 float → 1 个 float
inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float r;
    _mm_store_ss(&r, sum4);
    return r;
}

// AVX2 float32 双累加器点积
inline float avx2_dot(const float* __restrict__ a,
                      const float* __restrict__ b,
                      size_t len)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t k = 0;
    for (; k + 15 < len; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),     _mm256_loadu_ps(b + k),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
    }
    for (; k + 7 < len; k += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    float dot = hsum256(acc0);
    for (; k < len; ++k) {
        dot += a[k] * b[k];
    }
    return dot;
}

// ============================================================
// 批量类型转换 (8 元素 SIMD)
// ============================================================

// bf16 x8 → f32 x8
// bf16 与 f32 共享指数格式, 仅需左移 16 位
inline __m256 bf16x8_to_f32x8(const uint16_t* p) {
    __m128i h8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i i32 = _mm256_cvtepu16_epi32(h8);
    i32 = _mm256_slli_epi32(i32, 16);
    return _mm256_castsi256_ps(i32);
}

// fp16 x8 → f32 x8
// 提取 sign/exp/mantissa, 重偏指数 (+112), 处理零值
inline __m256 fp16x8_to_f32x8(const uint16_t* p) {
    __m128i h8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i i32 = _mm256_cvtepu16_epi32(h8);

    __m256i sign = _mm256_slli_epi32(
        _mm256_and_si256(i32, _mm256_set1_epi32(0x8000)), 16);
    __m256i exp16 = _mm256_and_si256(
        _mm256_srli_epi32(i32, 10), _mm256_set1_epi32(0x1F));
    __m256i mant = _mm256_and_si256(i32, _mm256_set1_epi32(0x3FF));

    __m256i exp32 = _mm256_slli_epi32(
        _mm256_add_epi32(exp16, _mm256_set1_epi32(112)), 23);
    __m256i mant32 = _mm256_slli_epi32(mant, 13);

    __m256i result = _mm256_or_si256(sign, _mm256_or_si256(exp32, mant32));

    // 处理零 (exp==0 且 mant==0)
    __m256i is_zero = _mm256_cmpeq_epi32(
        _mm256_and_si256(i32, _mm256_set1_epi32(0x7FFF)),
        _mm256_setzero_si256());
    result = _mm256_andnot_si256(is_zero, result);
    result = _mm256_or_si256(result, _mm256_and_si256(is_zero, sign));

    return _mm256_castsi256_ps(result);
}

// ============================================================
// AVX-512 SIMD 辅助函数 (仅在支持 AVX-512 的编译器/目标上可用)
// ============================================================
#ifdef __AVX512F__

// AVX-512 水平求和: 16 个 float → 1 个 float
inline float hsum512(__m512 v) {
    // 256-bit 上下两半相加
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum8 = _mm256_add_ps(lo, hi);
    return hsum256(sum8);
}

// bf16 x16 → f32 x16 (AVX-512)
// 从 16 个 uint16_t 生成 16 个 f32
inline __m512 bf16x16_to_f32x16(const uint16_t* p) {
    __m256i h16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i i32 = _mm512_cvtepu16_epi32(h16);
    i32 = _mm512_slli_epi32(i32, 16);
    return _mm512_castsi512_ps(i32);
}

// fp16 x16 → f32 x16 (AVX-512)
inline __m512 fp16x16_to_f32x16(const uint16_t* p) {
    __m256i h16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i i32 = _mm512_cvtepu16_epi32(h16);

    __m512i sign = _mm512_slli_epi32(
        _mm512_and_si512(i32, _mm512_set1_epi32(0x8000)), 16);
    __m512i exp16 = _mm512_and_si512(
        _mm512_srli_epi32(i32, 10), _mm512_set1_epi32(0x1F));
    __m512i mant = _mm512_and_si512(i32, _mm512_set1_epi32(0x3FF));

    __m512i exp32 = _mm512_slli_epi32(
        _mm512_add_epi32(exp16, _mm512_set1_epi32(112)), 23);
    __m512i mant32 = _mm512_slli_epi32(mant, 13);

    __m512i result = _mm512_or_si512(sign, _mm512_or_si512(exp32, mant32));

    // 处理零 (exp==0 且 mant==0): 使用 mask 操作
    __mmask16 is_zero = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(i32, _mm512_set1_epi32(0x7FFF)),
        _mm512_setzero_si512());
    result = _mm512_mask_mov_epi32(result, is_zero, sign);

    return _mm512_castsi512_ps(result);
}

#endif // __AVX512F__

} // namespace llaisys::utils
