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
#ifdef USE_OPENBLAS
#include <cblas.h>
#include <unordered_map>
#include <mutex>
#endif

namespace llaisys::ops {

#ifdef USE_OPENBLAS
// 权重缓存: bf16/fp16 权重在首次使用时转为 f32 并缓存, 后续调用直接复用。
// key = (原始权重数据指针, 元素个数, 类型标签)。
// 类型标签区分同一地址不同 dtype 的数据 (测试场景下内存可能被复用)。
// 推理场景中权重地址唯一且不释放, 缓存命中率 100%。
struct WeightCacheKey {
    const void* ptr;
    size_t count;
    int dtype_tag;  // 0=bf16, 1=fp16
    bool operator==(const WeightCacheKey& o) const {
        return ptr == o.ptr && count == o.count && dtype_tag == o.dtype_tag;
    }
};
struct WeightCacheKeyHash {
    size_t operator()(const WeightCacheKey& k) const {
        size_t h = std::hash<const void*>()(k.ptr);
        h ^= std::hash<size_t>()(k.count) << 1;
        h ^= std::hash<int>()(k.dtype_tag) << 2;
        return h;
    }
};
static std::unordered_map<WeightCacheKey, std::vector<float>, WeightCacheKeyHash> g_weight_cache;
static std::mutex g_weight_cache_mutex;

// 查找或创建 f32 权重缓存 (线程安全)
template<typename ConvertFn>
static const float* get_cached_f32_weights(const void* key, size_t count,
                                            int dtype_tag, ConvertFn convert_fn)
{
    WeightCacheKey cache_key{key, count, dtype_tag};
    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
    auto it = g_weight_cache.find(cache_key);
    if (it != g_weight_cache.end())
        return it->second.data();
    auto& buf = g_weight_cache[cache_key];
    buf.resize(count);
    convert_fn(buf.data());
    return buf.data();
}
#endif

// M 维度低于此阈值时跳过 OpenMP，避免线程创建/同步开销。
// 模型推理解码阶段 M=1，若不跳过，每次 linear 调用产生数千次无效 barrier。
static constexpr size_t OMP_M_THRESHOLD = 32;

// 条件并行 for：当 n >= threshold 时使用 OpenMP 多线程，否则纯串行。
// 使用 C 级 if 而非 OpenMP if() 子句，后者仍会进入 GOMP 运行时产生开销。
template<typename F>
static inline void parallel_for(size_t n, size_t threshold, F&& func) {
#ifdef _OPENMP
    if (n >= threshold) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) func(i);
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) func(i);
}

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
// float32 特化
// USE_OPENBLAS 时调用 cblas_sgemm / cblas_sgemv;
// 否则使用手写 AVX2+FMA + OpenMP 并行化版本。
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

#ifdef USE_OPENBLAS
    // Y = X[M,K] * W[N,K]^T  =>  sgemm(NoTrans, Trans, M, N, K, 1, X, K, W, K, 0, Y, N)
    // 若有 bias，先将 bias 广播填入 Y，然后用 beta=1 累加 GEMM 结果。
    if (b) {
        for (size_t i = 0; i < M; ++i)
            std::memcpy(Y + i * N, b, N * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (blasint)M, (blasint)N, (blasint)K,
                    1.0f, X, (blasint)K, W, (blasint)K,
                    1.0f, Y, (blasint)N);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (blasint)M, (blasint)N, (blasint)K,
                    1.0f, X, (blasint)K, W, (blasint)K,
                    0.0f, Y, (blasint)N);
    }
#else
    // ---------- 手写 AVX2+FMA fallback ----------
    if (b) {
        parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
            float* yrow = Y + i * N;
            size_t j = 0;
            for (; j + 7 < N; j += 8)
                _mm256_storeu_ps(yrow + j, _mm256_loadu_ps(b + j));
            for (; j < N; ++j)
                yrow[j] = b[j];
        });
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

                parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
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
                });
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const float* wj = W + (j0 + jj) * K + k0;
                    parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
                        Y[i*N + j0 + jj] += utils::avx2_dot(X + i*K + k0, wj, kc);
                    });
                }
            }
        }
    }
#endif
}

// ============================================================
// bfloat16 特化
// USE_OPENBLAS 时: M >= 阈值 → bf16→f32 + cblas_sgemm (权重缓存);
//                  M < 阈值 → 手写 AVX2 (内存带宽受限, 无需转换)。
// 否则始终使用手写 AVX2 + OpenMP 并行化版本。
// ============================================================

// M 维度 >= 此阈值时使用 OpenBLAS sgemm, 否则使用手写 AVX2。
// 解码阶段 M=1 是内存带宽瓶颈, bf16 直接读取比 f32 少一半带宽。
static constexpr size_t BLAS_M_THRESHOLD = 32;

// 手写 AVX2 bf16 linear 实现 (始终可用)
static void bf16_linear_avx2(uint16_t* Y, const uint16_t* X,
                              const uint16_t* W, const uint16_t* B,
                              size_t M, size_t K, size_t N)
{
    std::vector<float> ybuf(M * N);

    if (B) {
        parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
            float* yrow = ybuf.data() + i * N;
            for (size_t j = 0; j < N; ++j)
                yrow[j] = utils::cast<float>(llaisys::bf16_t{B[j]});
        });
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

                parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
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
                });
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const uint16_t* wj = W + (j0 + jj) * K + k0;
                    parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
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
                    });
                }
            }
        }
    }

    // f32 → bf16 写回
    parallel_for(M * N, OMP_M_THRESHOLD, [&](size_t idx) {
        llaisys::bf16_t v = utils::cast<llaisys::bf16_t>(ybuf[idx]);
        Y[idx] = v._v;
    });
}

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

#ifdef USE_OPENBLAS
    if (M >= BLAS_M_THRESHOLD) {
        // 大 M: bf16 → f32 → cblas_sgemm → f32 → bf16
        // cblas_sbgemm 在大 K 维度下结果不正确 (已知 OpenBLAS bug), 故用 sgemm。
        // 权重 W 在首次调用时转为 f32 并缓存, 后续直接复用。

        auto bf16_to_f32_bulk = [](const uint16_t* src, float* dst, size_t count) {
            size_t i = 0;
            for (; i + 7 < count; i += 8) {
                __m256 v = utils::bf16x8_to_f32x8(src + i);
                _mm256_storeu_ps(dst + i, v);
            }
            for (; i < count; ++i)
                dst[i] = utils::cast<float>(llaisys::bf16_t{src[i]});
        };

        // 缓存 f32 权重 (仅首次转换, 后续复用)
        const float* Wf = get_cached_f32_weights(W, N * K, /*dtype_tag=*/0,
            [&](float* dst) { bf16_to_f32_bulk(W, dst, N * K); });

        // 输入每次都转换
        std::vector<float> Xf(M * K);
        bf16_to_f32_bulk(X, Xf.data(), M * K);

        std::vector<float> Cf(M * N);

        if (B) {
            const float* bias_f = get_cached_f32_weights(B, N, /*dtype_tag=*/0,
                [&](float* dst) { bf16_to_f32_bulk(B, dst, N); });
            for (size_t i = 0; i < M; ++i)
                std::memcpy(Cf.data() + i * N, bias_f, N * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (blasint)M, (blasint)N, (blasint)K,
                        1.0f, Xf.data(), (blasint)K, Wf, (blasint)K,
                        1.0f, Cf.data(), (blasint)N);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (blasint)M, (blasint)N, (blasint)K,
                        1.0f, Xf.data(), (blasint)K, Wf, (blasint)K,
                        0.0f, Cf.data(), (blasint)N);
        }

        // f32 → bf16 写回
        for (size_t i = 0; i < M * N; ++i) {
            llaisys::bf16_t v = utils::cast<llaisys::bf16_t>(Cf[i]);
            Y[i] = v._v;
        }
        return;
    }
#endif
    // 小 M (含 M=1 decode): 手写 AVX2, 直接读 bf16, 节省内存带宽
    bf16_linear_avx2(Y, X, W, B, M, K, N);
}

// ============================================================
// float16 特化
// USE_OPENBLAS 时将 fp16 转为 f32 再调用 cblas_sgemm;
// 否则使用手写 AVX2 + OpenMP 并行化版本。
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

#ifdef USE_OPENBLAS
    // fp16 → f32 转换后调用 sgemm (OpenBLAS 无原生 fp16 GEMM)
    // 权重 W 在首次调用时转为 f32 并缓存, 后续直接复用。

    // fp16 → f32 (利用已有的 SIMD fp16x8_to_f32x8)
    auto fp16_to_f32_bulk = [](const uint16_t* src, float* dst, size_t count) {
        size_t i = 0;
        for (; i + 7 < count; i += 8) {
            __m256 v = utils::fp16x8_to_f32x8(src + i);
            _mm256_storeu_ps(dst + i, v);
        }
        for (; i < count; ++i)
            dst[i] = utils::cast<float>(llaisys::fp16_t{src[i]});
    };

    // 缓存 f32 权重 (仅首次转换, 后续复用)
    const float* Wf = get_cached_f32_weights(W, N * K, /*dtype_tag=*/1,
        [&](float* dst) { fp16_to_f32_bulk(W, dst, N * K); });

    // 输入每次都转换
    std::vector<float> Xf(M * K);
    fp16_to_f32_bulk(X, Xf.data(), M * K);

    std::vector<float> Cf(M * N);

    if (B) {
        const float* bias_f = get_cached_f32_weights(B, N, /*dtype_tag=*/1,
            [&](float* dst) { fp16_to_f32_bulk(B, dst, N); });
        for (size_t i = 0; i < M; ++i)
            std::memcpy(Cf.data() + i * N, bias_f, N * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (blasint)M, (blasint)N, (blasint)K,
                    1.0f, Xf.data(), (blasint)K, Wf, (blasint)K,
                    1.0f, Cf.data(), (blasint)N);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (blasint)M, (blasint)N, (blasint)K,
                    1.0f, Xf.data(), (blasint)K, Wf, (blasint)K,
                    0.0f, Cf.data(), (blasint)N);
    }

    // f32 → fp16 写回
    for (size_t i = 0; i < M * N; ++i) {
        llaisys::fp16_t v = utils::cast<llaisys::fp16_t>(Cf[i]);
        Y_out[i] = v._v;
    }
#else

    std::vector<float> ybuf(M * N);

    if (B) {
        parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
            float* yrow = ybuf.data() + i * N;
            for (size_t j = 0; j < N; ++j)
                yrow[j] = utils::cast<float>(llaisys::fp16_t{B[j]});
        });
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

                parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
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
                });
            } else {
                for (size_t jj = 0; jj < jn; ++jj) {
                    const uint16_t* wj = W + (j0 + jj) * K + k0;
                    parallel_for(M, OMP_M_THRESHOLD, [&](size_t i) {
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
                    });
                }
            }
        }
    }

    // f32 → fp16 写回
    parallel_for(M * N, OMP_M_THRESHOLD, [&](size_t idx) {
        llaisys::fp16_t v = utils::cast<llaisys::fp16_t>(ybuf[idx]);
        Y_out[idx] = v._v;
    });
#endif
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
