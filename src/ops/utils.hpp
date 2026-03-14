#include "llaisys.h"

#include <cstring>
#include <cstdlib>

#include <type_traits>
#include <new>
#include <iostream>
#include <stdexcept>

#ifdef __C
    #pragma push_macro("__C")
    #undef __C
#endif

#include <immintrin.h>   // 只在这个 cpp 内部使用
#ifdef __C
    #pragma pop_macro("__C")
#endif

#include <omp.h>

namespace llaisys::ops::utils {
// AVX2 + F16C vectorized conversions
template <typename T>
class OpenBlasCapableArray {
public:
    explicit OpenBlasCapableArray(size_t n, const T *data, llaisysDataType_t dtype) : numel_(n) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float *data_float = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&data_float), 32, n * sizeof(float)) != 0) {
                throw std::bad_alloc();
            }
            
            data_ = reinterpret_cast<std::byte*>(data_float);
            dtype_ = LLAISYS_DTYPE_F32;
            
            if constexpr (std::is_same_v<T, float>) {
                std::memcpy(data_float, data, n * sizeof(float));
            } else if constexpr (std::is_same_v<T, fp16_t>) {
                const size_t last_block_start = n - (n % 16);
                const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(static_cast<const void*>(data));
#ifdef _OPENMP
                #pragma omp parallel for if(n > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 16) {
                    __m256i vfp16 = _mm256_loadu_si256((const __m256i*)(raw_data + i));
                    
                    __m128i lo = _mm256_extracti128_si256(vfp16, 0);
                    __m128i hi = _mm256_extracti128_si256(vfp16, 1);
                    
                    _mm256_store_ps(data_float + i, _mm256_cvtph_ps(lo));
                    _mm256_store_ps(data_float + i + 8, _mm256_cvtph_ps(hi));
                }

                for (size_t i = last_block_start; i < n; i++) {
                    data_float[i] = llaisys::utils::cast<float>(data[i]);
                }
            } else if constexpr (std::is_same_v<T, bf16_t>){
                const size_t last_block_start = n - (n % 16);
                const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(static_cast<const void*>(data));
#ifdef _OPENMP
                #pragma omp parallel for if(n > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 16) {
                    __m256i vbf16 = _mm256_loadu_si256((const __m256i*)(raw_data + i));

                    __m128i lo = _mm256_extracti128_si256(vbf16, 0);
                    __m128i hi = _mm256_extracti128_si256(vbf16, 1);

                    __m256i lo_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16);
                    __m256i hi_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16);

                    _mm256_store_ps(data_float + i, _mm256_castsi256_ps(lo_32));
                    _mm256_store_ps(data_float + i + 8, _mm256_castsi256_ps(hi_32));
                }
                for (size_t i = last_block_start; i < n; i++) {
                    data_float[i] = llaisys::utils::cast<float>(data[i]);
                }
            }
        } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int32_t>) {
            double *data_double = nullptr;
            
            if (posix_memalign(reinterpret_cast<void**>(&data_double), 32, n * sizeof(double)) != 0) {
                throw std::bad_alloc();
            }
            data_ = reinterpret_cast<std::byte*>(data_double);
            dtype_ = LLAISYS_DTYPE_F64;

            if constexpr (std::is_same_v<T, double>) {
                std::memcpy(data_double, data, n * sizeof(double));
            } else if constexpr (std::is_same_v<T, int32_t>) {
                const size_t last_block_start = n - (n % 8);
#ifdef _OPENMP
                #pragma omp parallel for if(n > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 8) {
                    __m256i vi = _mm256_loadu_si256((__m256i*)(data + i));

                    __m128i lo = _mm256_extracti128_si256(vi, 0);
                    __m128i hi = _mm256_extracti128_si256(vi, 1);

                    __m256d dlo = _mm256_cvtepi32_pd(lo);
                    __m256d dhi = _mm256_cvtepi32_pd(hi);

                    _mm256_storeu_pd(data_double + i, dlo);
                    _mm256_storeu_pd(data_double + i + 4, dhi);
                }
                for (size_t i = last_block_start; i < n; i++) {
                    data_double[i] = llaisys::utils::cast<double>(data[i]);
                }
            }
        } else {
            std::cerr << "OpenBlasCapableArray only support bf16 fp16 float double int32.\n";
        }
    }

    OpenBlasCapableArray(const OpenBlasCapableArray&) = delete;
    OpenBlasCapableArray& operator=(const OpenBlasCapableArray&) = delete;
    OpenBlasCapableArray(OpenBlasCapableArray&& other) = delete;

    ~OpenBlasCapableArray() {
        if (data_) {
            std::free(data_);
        }
    }

    // Getters
    size_t numel() const noexcept { return numel_; }
    std::byte* data() const noexcept { return data_; }
    llaisysDataType_t dtype() const noexcept { return dtype_; }

private:
    size_t numel_;
    std::byte *data_;
    llaisysDataType_t dtype_;

public:
    void cast_back(T* dst) const {
        if constexpr (std::is_same_v<T, fp16_t>) {
            const float* data_float = reinterpret_cast<const float*>(data_);
            uint16_t* dst_raw = reinterpret_cast<uint16_t*>(static_cast<void*>(dst));
            size_t last_block_start = numel_ - (numel_ % 16);

#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 16) {
                __m256 lo = _mm256_loadu_ps(data_float + i);
                __m256 hi = _mm256_loadu_ps(data_float + i + 8);
                __m128i lo16 = _mm256_cvtps_ph(lo, 0);
                __m128i hi16 = _mm256_cvtps_ph(hi, 0);
                __m256i combined = _mm256_set_m128i(hi16, lo16);
                _mm256_storeu_si256((__m256i*)(dst_raw + i), combined);
            }
            for (size_t i = last_block_start; i < numel_; i++) {
                dst_raw[i] = llaisys::utils::cast<fp16_t>(data_float[i])._v;
            }

        } else if constexpr (std::is_same_v<T, bf16_t>) {
            const float* data_float = reinterpret_cast<const float*>(data_);
            uint16_t* dst_raw = reinterpret_cast<uint16_t*>(static_cast<void*>(dst));
            size_t last_block_start = numel_ - (numel_ % 16);

#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < last_block_start; i += 8) { // 8 floats -> 8 bf16
                __m256 f = _mm256_loadu_ps(data_float + i);
                __m256i as_int = _mm256_castps_si256(f);
                __m256i bias = _mm256_set1_epi32(0x7FFF);
                __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(as_int, 16), _mm256_set1_epi32(1));
                __m256i rounding = _mm256_add_epi32(bias, lsb);
                __m256i rounded = _mm256_add_epi32(as_int, rounding);
                __m256i shifted = _mm256_srli_epi32(rounded, 16);

                __m128i lo = _mm256_extracti128_si256(shifted, 0); // 4 x uint32
                __m128i hi = _mm256_extracti128_si256(shifted, 1); // 4 x uint32

                __m128i packed = _mm_packus_epi32(lo, hi);
                _mm_storeu_si128((__m128i*)(dst_raw + i), packed);
            }
            for (size_t i = last_block_start; i < numel_; i++) {
                dst_raw[i] = llaisys::utils::cast<bf16_t>(data_float[i])._v;
            }

        } else if constexpr (std::is_same_v<T, int32_t>) {
            const double* data_double = reinterpret_cast<const double*>(data_);
            size_t last_block_start = numel_ - (numel_ % 8);

#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 8) {
                __m256d d1 = _mm256_loadu_pd(data_double + i);
                __m256d d2 = _mm256_loadu_pd(data_double + i + 4);
                __m128i i1 = _mm256_cvttpd_epi32(d1);
                __m128i i2 = _mm256_cvttpd_epi32(d2);
                __m256i combined = _mm256_set_m128i(i2, i1);
                _mm256_storeu_si256((__m256i*)(dst + i), combined);
            }
            for (size_t i = last_block_start; i < numel_; i++) {
                dst[i] = static_cast<int32_t>(data_double[i]);
            }

        } else { // float, double
            std::memcpy(dst, data_, numel_ * sizeof(T));
        }
    }
};
} // namespace llaisys::ops::utils
