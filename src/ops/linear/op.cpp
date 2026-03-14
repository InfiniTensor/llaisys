// simd.hpp MUST be included before op.hpp because op.hpp transitively pulls
// in llaisys.h which defines `#define __C extern "C"`.  That macro corrupts
// parameter names inside <immintrin.h> (e.g. ia32intrin.h:63 `__crc32b(...__C...)`).
// By including simd.hpp first, <immintrin.h> is parsed before the __C macro exists.
#include "../../utils/simd.hpp"
#include "op.hpp"
#include "../../utils.hpp"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaisys::ops {

// ============================================================
// 泛型版本 (fallback): 纯标量, 无 OpenMP, 无 SIMD
// ============================================================
template<typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias){
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* weight_ptr = reinterpret_cast<const T*>(weight->data());
    const T* bias_ptr = nullptr;
    if(bias && bias->numel() > 0) bias_ptr = reinterpret_cast<const T*>(bias->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    size_t K = in->shape().back();
    size_t N = weight->shape()[0];

    size_t M;
    if (in->shape().size() == 2) {
        M = in->shape()[0];
    } else {
        M = in->shape()[0] * in->shape()[1];
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float x_val = utils::cast<float>(in_ptr[k + i * K]);
                float w_val = utils::cast<float>(weight_ptr[k + j * K]);
                sum += x_val * w_val;
            }
            if (bias_ptr) {
                sum += utils::cast<float>(bias_ptr[j]);
            }
            out_ptr[j + i * N] = utils::cast<T>(sum);
        }
    }
}

// 提取维度信息: M, K, N
static inline void get_dims(tensor_t in, tensor_t weight,
                            size_t &M, size_t &K, size_t &N)
{
    const auto& xs = in->shape();
    const auto& ws = weight->shape();
    if (xs.size() == 2) {
        M = xs[0]; K = xs[1];
    } else if (xs.size() == 3) {
        M = xs[0] * xs[1]; K = xs[2];
    } else {
        K = xs.back(); M = in->numel() / K;
    }
    N = ws[0];
}

// ============================================================
// float32 AVX2+FMA 特化 (OpenMP 并行化)
// 保持原始 k0→j0→i 分块顺序以复用 W 块缓存。
// i 维度行间完全独立，#pragma omp parallel for 在最内层 i 循环上。
// ============================================================
template<>
void linear_cpu_kernel<float>(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias){
    const float* X = reinterpret_cast<const float*>(in->data());
    const float* W = reinterpret_cast<const float*>(weight->data());
    const float* b = (bias && bias->numel() > 0)
                     ? reinterpret_cast<const float*>(bias->data()) : nullptr;
    float* Y = reinterpret_cast<float*>(out->data());

    size_t M, K, N;
    get_dims(in, weight, M, K, N);

    if (b) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < M; ++i) {
            float* yrow = Y + i * N;
            size_t j = 0;
            for (; j + 7 < N; j += 8)
                _mm256_storeu_ps(yrow + j, _mm256_loadu_ps(b + j));
            for (; j < N; ++j)
                yrow[j] = b[j];
        }
    } else {
        std::memset(Y, 0, M * N * sizeof(float));
    }

    static constexpr size_t BLOCK_K = 512;
    static constexpr size_t BLOCK_N = 4;

    for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
        const size_t kc = std::min(BLOCK_K, K - k0);

        for (size_t j0 = 0; j0 < N; j0 += BLOCK_N) {
            const size_t jn = std::min(BLOCK_N, N - j0);

            if (jn == 4) {
                const float* w0 = W + (j0 + 0) * K + k0;
                const float* w1 = W + (j0 + 1) * K + k0;
                const float* w2 = W + (j0 + 2) * K + k0;
                const float* w3 = W + (j0 + 3) * K + k0;

#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < M; ++i) {
                    const float* xi = X + i * K + k0;
                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 15 < kc; k += 16) {
                        __m256 x0 = _mm256_loadu_ps(xi + k);
                        __m256 x1 = _mm256_loadu_ps(xi + k + 8);
                        a00 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w0 + k),     a00);
                        a01 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w0 + k + 8), a01);
                        a10 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w1 + k),     a10);
                        a11 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w1 + k + 8), a11);
                        a20 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w2 + k),     a20);
                        a21 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w2 + k + 8), a21);
                        a30 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w3 + k),     a30);
                        a31 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w3 + k + 8), a31);
                    }
                    for (; k + 7 < kc; k += 8) {
                        __m256 x0 = _mm256_loadu_ps(xi + k);
                        a00 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w0 + k), a00);
                        a10 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w1 + k), a10);
                        a20 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w2 + k), a20);
                        a30 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w3 + k), a30);
                    }

                    float d0 = utils::hsum256(_mm256_add_ps(a00, a01));
                    float d1 = utils::hsum256(_mm256_add_ps(a10, a11));
                    float d2 = utils::hsum256(_mm256_add_ps(a20, a21));
                    float d3 = utils::hsum256(_mm256_add_ps(a30, a31));

                    for (; k < kc; ++k) {
                        float xk = xi[k];
                        d0 += xk * w0[k]; d1 += xk * w1[k];
                        d2 += xk * w2[k]; d3 += xk * w3[k];
                    }

                    Y[i*N + j0]   += d0;
                    Y[i*N + j0+1] += d1;
                    Y[i*N + j0+2] += d2;
                    Y[i*N + j0+3] += d3;
                }
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const float* wj = W + (j0 + jj) * K + k0;
#ifdef _OPENMP
                    #pragma omp parallel for schedule(static)
#endif
                    for (size_t i = 0; i < M; ++i)
                        Y[i*N + j0 + jj] += utils::avx2_dot(X + i*K + k0, wj, kc);
                }
            }
        }
    }
}

// ============================================================
// bfloat16 AVX2 特化 (OpenMP 并行化)
// ybuf 作为整体预分配，i 循环写入各自行 ybuf[i*N..i*N+N-1]，行间无竞争。
// ============================================================
template<>
void linear_cpu_kernel<llaisys::bf16_t>(tensor_t out, tensor_t in,
                                         tensor_t weight, tensor_t bias)
{
    const uint16_t* X = reinterpret_cast<const uint16_t*>(in->data());
    const uint16_t* W = reinterpret_cast<const uint16_t*>(weight->data());
    const uint16_t* B = nullptr;
    if (bias && bias->numel() > 0)
        B = reinterpret_cast<const uint16_t*>(bias->data());
    uint16_t* Y = reinterpret_cast<uint16_t*>(out->data());

    size_t M, K, N;
    get_dims(in, weight, M, K, N);

    std::vector<float> ybuf(M * N);

    if (B) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < M; ++i) {
            float* yrow = ybuf.data() + i * N;
            for (size_t j = 0; j < N; ++j)
                yrow[j] = utils::cast<float>(llaisys::bf16_t{B[j]});
        }
    } else {
        std::memset(ybuf.data(), 0, M * N * sizeof(float));
    }

    static constexpr size_t BLOCK_K = 512;
    static constexpr size_t BLOCK_N = 4;

    for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
        const size_t kc = std::min(BLOCK_K, K - k0);

        for (size_t j0 = 0; j0 < N; j0 += BLOCK_N) {
            const size_t jn = std::min(BLOCK_N, N - j0);

            if (jn == 4) {
                const uint16_t* w0 = W + (j0 + 0) * K + k0;
                const uint16_t* w1 = W + (j0 + 1) * K + k0;
                const uint16_t* w2 = W + (j0 + 2) * K + k0;
                const uint16_t* w3 = W + (j0 + 3) * K + k0;

#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < M; ++i) {
                    const uint16_t* xi = X + i * K + k0;

                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 15 < kc; k += 16) {
                        __m256 x0 = utils::bf16x8_to_f32x8(xi + k);
                        __m256 x1 = utils::bf16x8_to_f32x8(xi + k + 8);
                        a00 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w0 + k),     a00);
                        a01 = _mm256_fmadd_ps(x1, utils::bf16x8_to_f32x8(w0 + k + 8), a01);
                        a10 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w1 + k),     a10);
                        a11 = _mm256_fmadd_ps(x1, utils::bf16x8_to_f32x8(w1 + k + 8), a11);
                        a20 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w2 + k),     a20);
                        a21 = _mm256_fmadd_ps(x1, utils::bf16x8_to_f32x8(w2 + k + 8), a21);
                        a30 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w3 + k),     a30);
                        a31 = _mm256_fmadd_ps(x1, utils::bf16x8_to_f32x8(w3 + k + 8), a31);
                    }
                    for (; k + 7 < kc; k += 8) {
                        __m256 x0 = utils::bf16x8_to_f32x8(xi + k);
                        a00 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w0 + k), a00);
                        a10 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w1 + k), a10);
                        a20 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w2 + k), a20);
                        a30 = _mm256_fmadd_ps(x0, utils::bf16x8_to_f32x8(w3 + k), a30);
                    }

                    float d0 = utils::hsum256(_mm256_add_ps(a00, a01));
                    float d1 = utils::hsum256(_mm256_add_ps(a10, a11));
                    float d2 = utils::hsum256(_mm256_add_ps(a20, a21));
                    float d3 = utils::hsum256(_mm256_add_ps(a30, a31));

                    for (; k < kc; ++k) {
                        float xk = utils::cast<float>(llaisys::bf16_t{xi[k]});
                        d0 += xk * utils::cast<float>(llaisys::bf16_t{w0[k]});
                        d1 += xk * utils::cast<float>(llaisys::bf16_t{w1[k]});
                        d2 += xk * utils::cast<float>(llaisys::bf16_t{w2[k]});
                        d3 += xk * utils::cast<float>(llaisys::bf16_t{w3[k]});
                    }

                    ybuf[i*N + j0]     += d0;
                    ybuf[i*N + j0 + 1] += d1;
                    ybuf[i*N + j0 + 2] += d2;
                    ybuf[i*N + j0 + 3] += d3;
                }
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const uint16_t* wj = W + (j0 + jj) * K + k0;
#ifdef _OPENMP
                    #pragma omp parallel for schedule(static)
#endif
                    for (size_t i = 0; i < M; ++i) {
                        const uint16_t* xi = X + i * K + k0;
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        size_t k = 0;
                        for (; k + 15 < kc; k += 16) {
                            acc0 = _mm256_fmadd_ps(utils::bf16x8_to_f32x8(xi + k),
                                                    utils::bf16x8_to_f32x8(wj + k), acc0);
                            acc1 = _mm256_fmadd_ps(utils::bf16x8_to_f32x8(xi + k + 8),
                                                    utils::bf16x8_to_f32x8(wj + k + 8), acc1);
                        }
                        for (; k + 7 < kc; k += 8)
                            acc0 = _mm256_fmadd_ps(utils::bf16x8_to_f32x8(xi + k),
                                                    utils::bf16x8_to_f32x8(wj + k), acc0);
                        float sum = utils::hsum256(_mm256_add_ps(acc0, acc1));
                        for (; k < kc; ++k)
                            sum += utils::cast<float>(llaisys::bf16_t{xi[k]})
                                 * utils::cast<float>(llaisys::bf16_t{wj[k]});
                        ybuf[i*N + j0 + jj] += sum;
                    }
                }
            }
        }
    }

    // f32 → bf16 写回
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < M * N; ++idx) {
        llaisys::bf16_t v = utils::cast<llaisys::bf16_t>(ybuf[idx]);
        Y[idx] = v._v;
    }
}

// ============================================================
// float16 AVX2 特化 (OpenMP 并行化)
// ybuf 作为整体预分配，i 循环写入各自行 ybuf[i*N..i*N+N-1]，行间无竞争。
// ============================================================
template<>
void linear_cpu_kernel<llaisys::fp16_t>(tensor_t out, tensor_t in,
                                         tensor_t weight, tensor_t bias)
{
    const uint16_t* X = reinterpret_cast<const uint16_t*>(in->data());
    const uint16_t* W = reinterpret_cast<const uint16_t*>(weight->data());
    const uint16_t* B = nullptr;
    if (bias && bias->numel() > 0)
        B = reinterpret_cast<const uint16_t*>(bias->data());
    uint16_t* Y_out = reinterpret_cast<uint16_t*>(out->data());

    size_t M, K, N;
    get_dims(in, weight, M, K, N);

    std::vector<float> ybuf(M * N);

    if (B) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < M; ++i) {
            float* yrow = ybuf.data() + i * N;
            for (size_t j = 0; j < N; ++j)
                yrow[j] = utils::cast<float>(llaisys::fp16_t{B[j]});
        }
    } else {
        std::memset(ybuf.data(), 0, M * N * sizeof(float));
    }

    static constexpr size_t BLOCK_K = 512;
    static constexpr size_t BLOCK_N = 4;

    for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
        const size_t kc = std::min(BLOCK_K, K - k0);

        for (size_t j0 = 0; j0 < N; j0 += BLOCK_N) {
            const size_t jn = std::min(BLOCK_N, N - j0);

            if (jn == 4) {
                const uint16_t* w0 = W + (j0 + 0) * K + k0;
                const uint16_t* w1 = W + (j0 + 1) * K + k0;
                const uint16_t* w2 = W + (j0 + 2) * K + k0;
                const uint16_t* w3 = W + (j0 + 3) * K + k0;

#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < M; ++i) {
                    const uint16_t* xi = X + i * K + k0;

                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 15 < kc; k += 16) {
                        __m256 x0 = utils::fp16x8_to_f32x8(xi + k);
                        __m256 x1 = utils::fp16x8_to_f32x8(xi + k + 8);
                        a00 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w0 + k),     a00);
                        a01 = _mm256_fmadd_ps(x1, utils::fp16x8_to_f32x8(w0 + k + 8), a01);
                        a10 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w1 + k),     a10);
                        a11 = _mm256_fmadd_ps(x1, utils::fp16x8_to_f32x8(w1 + k + 8), a11);
                        a20 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w2 + k),     a20);
                        a21 = _mm256_fmadd_ps(x1, utils::fp16x8_to_f32x8(w2 + k + 8), a21);
                        a30 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w3 + k),     a30);
                        a31 = _mm256_fmadd_ps(x1, utils::fp16x8_to_f32x8(w3 + k + 8), a31);
                    }
                    for (; k + 7 < kc; k += 8) {
                        __m256 x0 = utils::fp16x8_to_f32x8(xi + k);
                        a00 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w0 + k), a00);
                        a10 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w1 + k), a10);
                        a20 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w2 + k), a20);
                        a30 = _mm256_fmadd_ps(x0, utils::fp16x8_to_f32x8(w3 + k), a30);
                    }

                    float d0 = utils::hsum256(_mm256_add_ps(a00, a01));
                    float d1 = utils::hsum256(_mm256_add_ps(a10, a11));
                    float d2 = utils::hsum256(_mm256_add_ps(a20, a21));
                    float d3 = utils::hsum256(_mm256_add_ps(a30, a31));

                    for (; k < kc; ++k) {
                        float xk = utils::cast<float>(llaisys::fp16_t{xi[k]});
                        d0 += xk * utils::cast<float>(llaisys::fp16_t{w0[k]});
                        d1 += xk * utils::cast<float>(llaisys::fp16_t{w1[k]});
                        d2 += xk * utils::cast<float>(llaisys::fp16_t{w2[k]});
                        d3 += xk * utils::cast<float>(llaisys::fp16_t{w3[k]});
                    }

                    ybuf[i*N + j0]     += d0;
                    ybuf[i*N + j0 + 1] += d1;
                    ybuf[i*N + j0 + 2] += d2;
                    ybuf[i*N + j0 + 3] += d3;
                }
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const uint16_t* wj = W + (j0 + jj) * K + k0;
#ifdef _OPENMP
                    #pragma omp parallel for schedule(static)
#endif
                    for (size_t i = 0; i < M; ++i) {
                        const uint16_t* xi = X + i * K + k0;
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        size_t k = 0;
                        for (; k + 15 < kc; k += 16) {
                            acc0 = _mm256_fmadd_ps(utils::fp16x8_to_f32x8(xi + k),
                                                    utils::fp16x8_to_f32x8(wj + k), acc0);
                            acc1 = _mm256_fmadd_ps(utils::fp16x8_to_f32x8(xi + k + 8),
                                                    utils::fp16x8_to_f32x8(wj + k + 8), acc1);
                        }
                        for (; k + 7 < kc; k += 8)
                            acc0 = _mm256_fmadd_ps(utils::fp16x8_to_f32x8(xi + k),
                                                    utils::fp16x8_to_f32x8(wj + k), acc0);
                        float sum = utils::hsum256(_mm256_add_ps(acc0, acc1));
                        for (; k < kc; ++k)
                            sum += utils::cast<float>(llaisys::fp16_t{xi[k]})
                                 * utils::cast<float>(llaisys::fp16_t{wj[k]});
                        ybuf[i*N + j0 + jj] += sum;
                    }
                }
            }
        }
    }

    // f32 → fp16 写回
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < M * N; ++idx) {
        llaisys::fp16_t v = utils::cast<llaisys::fp16_t>(ybuf[idx]);
        Y_out[idx] = v._v;
    }
}

// ============================================================
// 入口函数
// ============================================================
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    in = in->isContiguous() ? in : in->contiguous();
    weight = weight->isContiguous() ? weight : weight->contiguous();
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            linear_cpu_kernel<llaisys::fp16_t>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu_kernel<llaisys::bf16_t>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_F32:
            linear_cpu_kernel<float>(out, in, weight, bias);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
