#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::vector<size_t> out_shape, std::vector<size_t> in_shape, std::vector<size_t> weight_shape, std::vector<size_t> bias_shape) {
    for (size_t i = 0; i < out_shape[0]; i++) {
        for (size_t j = 0; j < out_shape[1]; j++) {
            float val = 0.0F; // 使用 float 累加，避免 f16/bf16 精度损失
            for (size_t k = 0; k < in_shape[1]; k++) {
                val += llaisys::utils::cast<float>(in[i * in_shape[1] + k])
                     * llaisys::utils::cast<float>(weight[j * weight_shape[1] + k]);
            }
            if (bias) {
                val += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_shape[1] + j] = llaisys::utils::cast<T>(val);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, std::vector<size_t> out_shape, std::vector<size_t> in_shape, std::vector<size_t> weight_shape, std::vector<size_t> bias_shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), out_shape, in_shape, weight_shape, bias_shape);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), out_shape, in_shape, weight_shape, bias_shape);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), out_shape, in_shape, weight_shape, bias_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
