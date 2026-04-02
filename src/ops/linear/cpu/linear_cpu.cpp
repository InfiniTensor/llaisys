#include "linear_cpu.hpp"
#include <cblas.h>
#include <type_traits>
#include <vector>
#include "../../../utils.hpp"
#include <omp.h>

template <typename T>
void linear_kernel(const T *input, const T *weight, const T *bias, T *out,
                   size_t M, size_t K, size_t N) {
    if constexpr (std::is_same_v<T, float>) {
        const float* f_input = (const float*)input;
        const float* f_weight = (const float*)weight;
        float* f_out = (float*)out;

        if (M == 1) {
            // 🚀【新增】：Decode 阶段的绝对王者！
            // 当 M=1 时，输入其实是一个向量。我们调用专门的 cblas_sgemv！
            // Weight 是 N 行 K 列，Input 是长度为 K 的向量，Output 是长度为 N 的向量。
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        (int)N, (int)K, 1.0f, 
                        f_weight, (int)K, 
                        f_input, 1, 
                        0.0f, 
                        f_out, 1);
        } else {
            // 🚀 Prefill 阶段：保持原样，大矩阵对撞
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)M, (int)N, (int)K, 1.0f,
                        f_input, (int)K, f_weight, (int)K,
                        0.0f, f_out, (int)N);
        }

        // 加上偏置
        if (bias != nullptr) {
            const float* f_bias = (const float*)bias;
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    f_out[m * N + n] += f_bias[n];
                }
            }
        }
    } 
    // 🚨 绝对安全版：剥离 OpenMP，彻底杜绝线程死锁！
    else if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
        
        // ==========================================
        // 🚀【核心优化】：引入静态线程缓存 (Workspace)
        // thread_local 保证了这三块内存只会在第一次调用时被申请，
        // 之后成千上万次的循环都会直接复用这块物理内存！绝对的零开销！
        // ==========================================
        thread_local std::vector<float> f32_x;
        thread_local std::vector<float> f32_w;
        thread_local std::vector<float> f32_out;

        // resize 在预留空间足够大时，几乎不消耗任何时间
        f32_x.resize(M * K);
        f32_w.resize(N * K);
        f32_out.resize(M * N);

        // 1. 单线程安全强转（CPU 跑这点 for 循环极快）
        for (size_t i = 0; i < M * K; i++) {
            f32_x[i] = llaisys::utils::cast<float>(input[i]);
        }
        
        for (size_t i = 0; i < N * K; i++) {
            f32_w[i] = llaisys::utils::cast<float>(weight[i]);
        }

        // 2. 将所有并发算力全部留给 OpenBLAS！
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)N, (int)K, 1.0f,
                    f32_x.data(), (int)K,
                    f32_w.data(), (int)K,
                    0.0f,
                    f32_out.data(), (int)N);

        // 3. 算完后，单线程安全转回
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float val = f32_out[m * N + n];
                if (bias != nullptr) {
                    val += llaisys::utils::cast<float>(bias[n]);
                }
                out[m * N + n] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void linear(const std::byte *input_data, const std::byte *weight_data, const std::byte *bias_data,
            std::byte *out_data, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<float>(
            reinterpret_cast<const float *>(input_data),
            reinterpret_cast<const float *>(weight_data),
            reinterpret_cast<const float *>(bias_data),
            reinterpret_cast<float *>(out_data),
            M, K, N);
        break;

    case LLAISYS_DTYPE_F16:
        linear_kernel<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(input_data),
            reinterpret_cast<const llaisys::fp16_t *>(weight_data),
            reinterpret_cast<const llaisys::fp16_t *>(bias_data),
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            M, K, N);
        break;

    case LLAISYS_DTYPE_BF16:
        linear_kernel<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(input_data),
            reinterpret_cast<const llaisys::bf16_t *>(weight_data),
            reinterpret_cast<const llaisys::bf16_t *>(bias_data),
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            M, K, N);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu