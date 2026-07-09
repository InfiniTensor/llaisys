#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t input_rows, size_t input_cols, size_t weight_rows) {
    for (size_t i = 0; i < input_rows; i++) {
        for (size_t j = 0; j < weight_rows; j++) {

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // Accumulate in float to avoid unsupported operators on bf16/fp16.
                float acc = llaisys::utils::cast<float>(bias[j]);
                // Accumulate matrix multiplication: out = in @ weight.T
                for (size_t k = 0; k < input_cols; k++) {
                    acc += llaisys::utils::cast<float>(in[i * input_cols + k]) *
                           llaisys::utils::cast<float>(weight[j * input_cols + k]);
                }
                out[i * weight_rows + j] = llaisys::utils::cast<T>(acc);
            } else {
                out[i * weight_rows + j] = bias[j];
                for (size_t k = 0; k < input_cols; k++) {
                    out[i * weight_rows + j] += in[i * input_cols + k] * weight[j * input_cols + k];
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, size_t input_rows, size_t input_cols, size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), input_rows, input_cols, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
