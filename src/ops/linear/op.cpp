#include "op.hpp"

#include "../../utils.hpp"

#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(ENABLE_OPENBLAS)
#include <cblas.h>
#endif

namespace {
// 匿名命名空间：本文件私有实现，避免与其他翻译单元符号冲突。
#if defined(__AVX2__)
// AVX2 点积内核：每次处理 8 个 float，剩余元素走标量尾处理。
inline float dot_f32_avx2(const float *a, const float *b, size_t k) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= k; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];

    for (; i < k; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

inline float dot_f32(const float *a, const float *b, size_t k) {
#if defined(__AVX2__)
    // 编译期选择：支持 AVX2 时走 SIMD 快路径。
    return dot_f32_avx2(a, b, k);
#else
    float sum = 0.0f;
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd reduction(+ : sum)
#endif
    for (size_t i = 0; i < k; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void linear_impl_f32(float *out_ptr, const float *in_ptr, const float *weight_ptr, const float *bias_ptr,
                     size_t M, size_t K, size_t N) {
#if defined(ENABLE_OPENBLAS)
    // out = in * weight^T，weight 当前布局是 [N, K]，因此 GEMM 里用 TransB。
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(M),
                static_cast<int>(N),
                static_cast<int>(K),
                1.0f,
                in_ptr,
                static_cast<int>(K),
                weight_ptr,
                static_cast<int>(K),
                0.0f,
                out_ptr,
                static_cast<int>(N));

    if (bias_ptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
            float *out_row = out_ptr + static_cast<size_t>(m) * N;
            for (size_t n = 0; n < N; ++n) {
                out_row[n] += bias_ptr[n];
            }
        }
    }
    return;
#endif

#if defined(_OPENMP)
// 线程按行切分（m 维）：每个线程写不同 out_row，无写冲突。
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
        const float *in_row = in_ptr + static_cast<size_t>(m) * K;
        float *out_row = out_ptr + static_cast<size_t>(m) * N;
        for (size_t n = 0; n < N; ++n) {
            const float *w_row = weight_ptr + n * K;
            float sum = dot_f32(in_row, w_row, K);
            if (bias_ptr) {
                sum += bias_ptr[n];
            }
            out_row[n] = sum;
        }
    }
}

template <typename T>
void linear_impl(T *out_ptr, const T *in_ptr, const T *weight_ptr, const T *bias_ptr,
                 size_t M, size_t K, size_t N) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd reduction(+ : sum)
#endif
            for (size_t k = 0; k < K; ++k) {
                // 转换为float进行计算，避免精度损失
                sum += llaisys::utils::cast<float>(in_ptr[static_cast<size_t>(m) * K + k]) * llaisys::utils::cast<float>(weight_ptr[n * K + k]);
            }
            if (bias_ptr) {
                sum += llaisys::utils::cast<float>(bias_ptr[n]);
            }
            // 转换回目标类型
            out_ptr[static_cast<size_t>(m) * N + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

template <typename LowpT>
void linear_impl_lowp_fast(LowpT *out_ptr, const LowpT *in_ptr, const LowpT *weight_ptr, const LowpT *bias_ptr,
                           size_t M, size_t K, size_t N) {
    // 低精度专用路径：将低精度张量批量转换为 float 后计算，避免在最内层循环重复 cast。
    std::vector<float> weight_f(N * K);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(N * K); ++idx) {
        weight_f[static_cast<size_t>(idx)] = llaisys::utils::cast<float>(weight_ptr[static_cast<size_t>(idx)]);
    }

    std::vector<float> bias_f;
    if (bias_ptr) {
        bias_f.resize(N);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t n = 0; n < static_cast<ptrdiff_t>(N); ++n) {
            bias_f[static_cast<size_t>(n)] = llaisys::utils::cast<float>(bias_ptr[static_cast<size_t>(n)]);
        }
    }

#if defined(ENABLE_OPENBLAS)
    std::vector<float> in_f(M * K);
    std::vector<float> out_f(M * N);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(M * K); ++idx) {
        in_f[static_cast<size_t>(idx)] = llaisys::utils::cast<float>(in_ptr[static_cast<size_t>(idx)]);
    }

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(M),
                static_cast<int>(N),
                static_cast<int>(K),
                1.0f,
                in_f.data(),
                static_cast<int>(K),
                weight_f.data(),
                static_cast<int>(K),
                0.0f,
                out_f.data(),
                static_cast<int>(N));

    if (bias_ptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
            float *out_row = out_f.data() + static_cast<size_t>(m) * N;
            for (size_t n = 0; n < N; ++n) {
                out_row[n] += bias_f[n];
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(M * N); ++idx) {
        out_ptr[static_cast<size_t>(idx)] = llaisys::utils::cast<LowpT>(out_f[static_cast<size_t>(idx)]);
    }
    return;
#endif

    // 无 OpenBLAS 时，复用 SIMD/标量点积内核，仍保持 float 累加。
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<float> in_row_f(K);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
            const size_t m_u = static_cast<size_t>(m);
            const LowpT *in_row_lowp = in_ptr + m_u * K;
            LowpT *out_row = out_ptr + m_u * N;

            for (size_t k = 0; k < K; ++k) {
                in_row_f[k] = llaisys::utils::cast<float>(in_row_lowp[k]);
            }

            for (size_t n = 0; n < N; ++n) {
                const float *w_row = weight_f.data() + n * K;
                float sum = dot_f32(in_row_f.data(), w_row, K);
                if (bias_ptr) {
                    sum += bias_f[n];
                }
                out_row[n] = llaisys::utils::cast<LowpT>(sum);
            }
        }
    }
}

void validate_linear_args(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t weight, llaisys::tensor_t bias) {
    // 检查输入输出张量是否在同一设备上
    CHECK_SAME_DEVICE(out, in, weight, bias);
    // 检查数据类型是否匹配
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    // 检查张量是否是连续存储的
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
           "Linear: all tensors must be contiguous.");
    // 检查形状是否符合要求
    CHECK_ARGUMENT(out->ndim() == 2, "Linear: out must be 2D.");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: in must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D.");
    CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D.");
    // 此时weight还没转置，故in的第二维度应等于weight的第二维度
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must equal weight.shape[1].");
    // 输出张量的第一维度应等于输入张量的第一维度
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear: out.shape[0] must equal in.shape[0].");
    // 输出张量的第二维度应等于weight的第一维度
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: out.shape[1] must equal weight.shape[0].");
    // bias可为空
    // 若不为空，bias的大小应等于输出张量的第二维度
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1], "Linear: bias.shape[0] must equal out.shape[1].");
    }
    // 目前仅支持CPU设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void dispatch_linear_kernel(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t weight, llaisys::tensor_t bias) {
    const size_t M = in->shape()[0];     // in行数
    const size_t K = in->shape()[1];     // in列数
    const size_t N = weight->shape()[0]; // weight行数
    const auto type = in->dtype();

    // 运行时 dtype 分发：f32 用特化快路径，f16/bf16 走模板通用路径。
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_impl_f32(reinterpret_cast<float *>(out->data()),
                               reinterpret_cast<const float *>(in->data()),
                               reinterpret_cast<const float *>(weight->data()),
                               bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                               M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_impl_lowp_fast(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                     reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                                     reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                                     bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr,
                                     M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_impl_lowp_fast(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                     reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                                     reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                                     bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr,
                                     M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 接口层：先校验参数，再分发内核实现。
    validate_linear_args(out, in, weight, bias);
    dispatch_linear_kernel(out, in, weight, bias);
}
} // namespace llaisys::ops
