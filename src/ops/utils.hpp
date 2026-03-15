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
class OpenBlasCapableArray {
public:
    // Allocate an intermediate buffer suitable for OpenBLAS (F32/F64).
    explicit OpenBlasCapableArray(size_t n, llaisysDataType_t dtype)
        : numel_(n), data_(nullptr), dtype_(dtype), owns_data_(true) {
        size_t elem_size = (dtype_ == LLAISYS_DTYPE_F64 ? sizeof(double) : sizeof(float));
        if (posix_memalign(&data_, 32, n * elem_size) != 0) {
            throw std::bad_alloc();
        }
    }

    // When the source is already float/double, we can use the caller's pointer directly
    // (no allocation, no own/free). This avoids extra copies when OpenBLAS can work on
    // the original buffer.
    template <typename T>
    OpenBlasCapableArray(const T* src, size_t n, llaisysDataType_t dtype)
        : numel_(n), data_(const_cast<void*>(static_cast<const void*>(src))), dtype_(dtype), owns_data_(false) {
        if constexpr (std::is_same_v<T, float>) {
            if (dtype_ != LLAISYS_DTYPE_F32) {
                throw std::invalid_argument("dtype mismatch for float source");
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (dtype_ != LLAISYS_DTYPE_F64) {
                throw std::invalid_argument("dtype mismatch for double source");
            }
        } else {
            // For other types, we still need to allocate and cast.
            // We do this by falling back to the “normal” construction path.
            size_t elem_size = (dtype_ == LLAISYS_DTYPE_F64 ? sizeof(double) : sizeof(float));
            if (posix_memalign(&data_, 32, n * elem_size) != 0) {
                throw std::bad_alloc();
            }
            owns_data_ = true;
            cast_from(src);
        }
    }

    // No copies / moves for simplicity
    OpenBlasCapableArray(const OpenBlasCapableArray&) = delete;
    OpenBlasCapableArray& operator=(const OpenBlasCapableArray&) = delete;
    OpenBlasCapableArray(OpenBlasCapableArray&& other) = delete;

    ~OpenBlasCapableArray() {
        if (owns_data_) {
            std::free(data_);
        }
    }

    // Fill the internal buffer from user-provided data
    template <typename T>
    void cast_from(const T* src) {
        if (dtype_ == LLAISYS_DTYPE_F32) {
            cast_helper(src, reinterpret_cast<float*>(data_), numel_);
        } else if (dtype_ == LLAISYS_DTYPE_F64) {
            cast_helper(src, reinterpret_cast<double*>(data_), numel_);
        } else {
            throw std::runtime_error("OpenBlasCapableArray only supports F32/F64 storage");
        }
    }

    // Broadcast a single row `src` (length cols) into a `rows x cols` block.
    template <typename T>
    void broadcast_row(const T* src, size_t rows, size_t cols) {
        if (rows == 0 || cols == 0) return;
        if (rows * cols != numel_) {
            throw std::out_of_range("broadcast_row is not equal to buffer size");
        }

        if (dtype_ == LLAISYS_DTYPE_F32) {
            float* base = reinterpret_cast<float*>(data_);
            cast_helper(src, base, cols);
#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t r = 1; r < rows; ++r) {
                std::memcpy(base + r * cols, base, cols * sizeof(float));
            }
        } else {
            double* base = reinterpret_cast<double*>(data_);
            cast_helper(src, base, cols);
#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t r = 1; r < rows; ++r) {
                std::memcpy(base + r * cols, base, cols * sizeof(double));
            }
        }
    }

    // Fill the internal buffer with zeros.
    // Uses AVX stores + OpenMP for large buffers to maximize throughput.
    void zeros() {
        if (numel_ == 0) return;

        if (dtype_ == LLAISYS_DTYPE_F32) {
            float* fdata = reinterpret_cast<float*>(data_);
            const size_t last_block_start = (numel_ / 8) * 8;
            __m256 zero = _mm256_setzero_ps();

#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 8) {
                _mm256_store_ps(fdata + i, zero);
            }
            for (size_t i = last_block_start; i < numel_; ++i) {
                fdata[i] = 0.0f;
            }

        } else if (dtype_ == LLAISYS_DTYPE_F64) {
            double* ddata = reinterpret_cast<double*>(data_);
            const size_t last_block_start = (numel_ / 4) * 4;
            __m256d zero = _mm256_setzero_pd();

#ifdef _OPENMP
            #pragma omp parallel for if(numel_ > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 4) {
                _mm256_store_pd(ddata + i, zero);
            }
            for (size_t i = last_block_start; i < numel_; ++i) {
                ddata[i] = 0.0;
            }

        } else {
            // Fallback: memset works for all bitwise-zero types and is often optimized.
            const size_t elem_size = (dtype_ == LLAISYS_DTYPE_F64 ? sizeof(double) : sizeof(float));
            std::memset(data_, 0, numel_ * elem_size);
        }
    }

    // Convert the internal buffer back to `T`.
    template <typename T>
    void cast_back(T* dst) const {
        if (dtype_ == LLAISYS_DTYPE_F32) {
            const float* data_float = reinterpret_cast<const float*>(data_);
            if constexpr (std::is_same_v<T, fp16_t>) {
                uint16_t* dst_raw = reinterpret_cast<uint16_t*>(static_cast<void*>(dst));
                size_t last_block_start = numel_ - (numel_ % 16);

#ifdef _OPENMP
                #pragma omp parallel for if(numel_ > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 16) {
                    __m256 lo = _mm256_load_ps(data_float + i);
                    __m256 hi = _mm256_load_ps(data_float + i + 8);
                    __m128i lo16 = _mm256_cvtps_ph(lo, 0);
                    __m128i hi16 = _mm256_cvtps_ph(hi, 0);
                    __m256i combined = _mm256_set_m128i(hi16, lo16);
                    _mm256_storeu_si256((__m256i*)(dst_raw + i), combined);
                }
                for (size_t i = last_block_start; i < numel_; i++) {
                    dst_raw[i] = llaisys::utils::cast<fp16_t>(data_float[i])._v;
                }

            } else if constexpr (std::is_same_v<T, bf16_t>) {
                uint16_t* dst_raw = reinterpret_cast<uint16_t*>(static_cast<void*>(dst));
                size_t last_block_start = numel_ - (numel_ % 16);

#ifdef _OPENMP
                #pragma omp parallel for if(numel_ > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 8) { // 8 floats -> 8 bf16
                    __m256 f = _mm256_load_ps(data_float + i);
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

            } else {
                // float or other types: fallback to scalar cast
                for (size_t i = 0; i < numel_; i++) {
                    dst[i] = llaisys::utils::cast<T>(data_float[i]);
                }
            }

        } else {
            const double* data_double = reinterpret_cast<const double*>(data_);
            if constexpr (std::is_same_v<T, int32_t>) {
                size_t last_block_start = numel_ - (numel_ % 8);

#ifdef _OPENMP
                #pragma omp parallel for if(numel_ > 16384)
#endif
                for (size_t i = 0; i < last_block_start; i += 8) {
                    __m256d d1 = _mm256_load_pd(data_double + i);
                    __m256d d2 = _mm256_load_pd(data_double + i + 4);
                    __m128i i1 = _mm256_cvttpd_epi32(d1);
                    __m128i i2 = _mm256_cvttpd_epi32(d2);
                    __m256i combined = _mm256_set_m128i(i2, i1);
                    _mm256_storeu_si256((__m256i*)(dst + i), combined);
                }
                for (size_t i = last_block_start; i < numel_; i++) {
                    dst[i] = static_cast<int32_t>(data_double[i]);
                }
            } else {
                for (size_t i = 0; i < numel_; i++) {
                    dst[i] = llaisys::utils::cast<T>(data_double[i]);
                }
            }
        }
    }

    // Getters
    size_t numel() const noexcept { return numel_; }
    void* data() const noexcept { return data_; }
    llaisysDataType_t dtype() const noexcept { return dtype_; }
    bool owns_data() const noexcept { return owns_data_; }

private:
    template <typename T>
    static void cast_helper(const T* src, float* dst, size_t n) {
        if constexpr (std::is_same_v<T, float>) {
            std::memcpy(dst, src, n * sizeof(float));
        } else if constexpr (std::is_same_v<T, fp16_t>) {
            const size_t last_block_start = n - (n % 16);
            const uint16_t* raw_src = reinterpret_cast<const uint16_t*>(static_cast<const void*>(src));

#ifdef _OPENMP
            #pragma omp parallel for if(n > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 16) {
                __m256i vfp16 = _mm256_loadu_si256((const __m256i*)(raw_src + i));
                __m128i lo = _mm256_extracti128_si256(vfp16, 0);
                __m128i hi = _mm256_extracti128_si256(vfp16, 1);
                _mm256_store_ps(dst + i, _mm256_cvtph_ps(lo));
                _mm256_store_ps(dst + i + 8, _mm256_cvtph_ps(hi));
            }
            for (size_t i = last_block_start; i < n; i++) {
                dst[i] = llaisys::utils::cast<float>(src[i]);
            }

        } else if constexpr (std::is_same_v<T, bf16_t>) {
            const size_t last_block_start = n - (n % 16);
            const uint16_t* raw_src = reinterpret_cast<const uint16_t*>(static_cast<const void*>(src));

#ifdef _OPENMP
            #pragma omp parallel for if(n > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 16) {
                __m256i vbf16 = _mm256_loadu_si256((const __m256i*)(raw_src + i));
                __m128i lo = _mm256_extracti128_si256(vbf16, 0);
                __m128i hi = _mm256_extracti128_si256(vbf16, 1);
                __m256i lo_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16);
                __m256i hi_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16);
                _mm256_store_ps(dst + i, _mm256_castsi256_ps(lo_32));
                _mm256_store_ps(dst + i + 8, _mm256_castsi256_ps(hi_32));
            }
            for (size_t i = last_block_start; i < n; i++) {
                dst[i] = llaisys::utils::cast<float>(src[i]);
            }

        } else {
            for (size_t i = 0; i < n; i++) {
                dst[i] = llaisys::utils::cast<float>(src[i]);
            }
        }
    }

    template <typename T>
    static void cast_helper(const T* src, double* dst, size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            std::memcpy(dst, src, n * sizeof(double));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            const size_t last_block_start = n - (n % 8);

#ifdef _OPENMP
            #pragma omp parallel for if(n > 16384)
#endif
            for (size_t i = 0; i < last_block_start; i += 8) {
                __m256i vi = _mm256_loadu_si256((__m256i*)(src + i));
                __m128i lo = _mm256_extracti128_si256(vi, 0);
                __m128i hi = _mm256_extracti128_si256(vi, 1);
                __m256d dlo = _mm256_cvtepi32_pd(lo);
                __m256d dhi = _mm256_cvtepi32_pd(hi);
                _mm256_store_pd(dst + i, dlo);
                _mm256_store_pd(dst + i + 4, dhi);
            }
            for (size_t i = last_block_start; i < n; i++) {
                dst[i] = llaisys::utils::cast<double>(src[i]);
            }

        } else {
            for (size_t i = 0; i < n; i++) {
                dst[i] = llaisys::utils::cast<double>(src[i]);
            }
        }
    }

private:
    size_t numel_;
    void* data_;
    llaisysDataType_t dtype_;
    bool owns_data_ = true;
};
} // namespace llaisys::ops::utils
