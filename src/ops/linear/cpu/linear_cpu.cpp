#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void linear_t(std::byte* out_raw,
              const std::byte* in_raw,
              const std::byte* w_raw,
              const std::byte* b_raw,
              size_t batch,
              size_t in_features,
              size_t out_features,
              bool has_bias) {
    const T* x = reinterpret_cast<const T*>(in_raw);
    const T* w = reinterpret_cast<const T*>(w_raw);
    const T* b = reinterpret_cast<const T*>(b_raw);
    T* y = reinterpret_cast<T*>(out_raw);

    for(size_t i = 0; i < batch; ++i) {
        const T* x_row = x + i * in_features;
        T* y_row = y + i * out_features;
        for(size_t j = 0; j < out_features; ++j) {
            float acc = has_bias ? llaisys::utils::cast<float>(b[j]) : 0.0f;
            const T* w_col = w + j * in_features;
            for(size_t k = 0; k < in_features; ++k) {
                acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_col[k]);
            }
            y_row[j] = llaisys::utils::cast<T>(acc);
        }
    }
}
void linear(std::byte* out,
            const std::byte* in,
            const std::byte* weight,
            const std::byte* bias,
            llaisysDataType_t dtype,
            size_t batch,
            size_t in_features,
            size_t out_features,
            bool has_bias) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return linear_t<float>(out, in, weight, bias, batch, in_features, out_features, has_bias);
        case LLAISYS_DTYPE_F16:
            return linear_t<llaisys::fp16_t>(out, in, weight, bias, batch, in_features, out_features, has_bias);
        case LLAISYS_DTYPE_BF16:
            return linear_t<llaisys::bf16_t>(out, in, weight, bias, batch, in_features, out_features, has_bias);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}  // namespace llaisys::ops::cpu