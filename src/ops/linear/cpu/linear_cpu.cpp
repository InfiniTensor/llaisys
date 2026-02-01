#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                   size_t M, size_t N, size_t K) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) { // 外层 i 循环：遍历 Batch 维度 (M)。
        for (size_t j = 0; j < N; ++j) { // 中层 j 循环：遍历输出特征维度 (N)
            // 将 bias 的初始值直接赋给 sum，减少一次判断
            float sum = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
            
            // 显式提示编译器进行向量化 (SIMD)
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; ++k) {
                // 内层 k 循环：执行向量内积，累加K个元素的乘积
                // Y = X * W^T -> Y[i,j] = sum(X[i,k] * W[j,k])
                sum += llaisys::utils::cast<float>(in[i * K + k]) * 
                       llaisys::utils::cast<float>(weight[j * K + k]);
            }
            out[i * N + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t N, size_t K) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<float>(reinterpret_cast<float *>(out),
                             reinterpret_cast<const float *>(in),
                             reinterpret_cast<const float *>(weight),
                             reinterpret_cast<const float *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                       reinterpret_cast<const llaisys::fp16_t *>(in),
                                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                                       reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                       reinterpret_cast<const llaisys::bf16_t *>(in),
                                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                                       reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu