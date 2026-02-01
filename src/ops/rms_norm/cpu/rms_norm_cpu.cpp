#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_kernel(T *out, const T *in, const T *weight, size_t M, size_t d, float eps) {
    // 对行进行并行化处理
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        float sum_sq = 0.0f; //平方和累加器-用于存储当前处理行（第 i 行）中所有元素的平方累加值。
        const T *row_in = in + i * d; // 当前输入行指针-指向输入张量 in 中第 i 行的起始地址-由于张量在内存中是连续存储的，每一行有 d 个元素。
        T *row_out = out + i * d;// 当前输出行指针-指向输出张量 out 中第 i 行的起始地址。

        // 1. 计算平方和
        #pragma omp simd reduction(+:sum_sq)
        for (size_t j = 0; j < d; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]); // row_in[j] = *(row_in + j)
            sum_sq += val * val;
        }

        // 2. 计算均方根的倒数
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 3. 应用权重并归一化
        #pragma omp simd
        for (size_t j = 0; j < d; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            row_out[j] = llaisys::utils::cast<T>(val * w * inv_rms);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t M, size_t d, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                               reinterpret_cast<const float *>(weight), M, d, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, d, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, d, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu