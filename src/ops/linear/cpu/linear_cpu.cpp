#include "linear_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {        // 遍历 X 的行
        for (size_t j = 0; j < N; ++j) {    // 遍历 W 的行 (即 Y 的列)
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) { // 内积计算
                // X[i, k] * W[j, k] (因为是 W^T，所以 W 也是取第 j 行第 k 列)
                float x_val = llaisys::utils::cast<float>(in[i * K + k]);
                float w_val = llaisys::utils::cast<float>(weight[j * K + k]);
                sum += x_val * w_val;
            }

            // 处理可选的 bias: b[j]
            if (bias) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }

            out[i * N + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
