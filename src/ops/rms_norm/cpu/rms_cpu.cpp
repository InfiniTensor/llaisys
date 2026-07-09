#include "rms_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(std::byte *out_raw, const std::byte *in_raw, const std::byte *weight_raw, size_t w_size, size_t numel, float eps) {
    T *out = reinterpret_cast<T*>(out_raw);
    const T *in = reinterpret_cast<const T*>(in_raw);
    const T *weight = reinterpret_cast<const T*>(weight_raw);

    /* 算子理解错了, 分母应该是行的平方和, 不是列的
    // 分母
    // naive 优化, 避免多次不连续列访问
    float *downs = (float*) calloc(w_size, sizeof(float));

    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            downs[i % w_size] += powf(llaisys::utils::cast<float>(in[numel]), 2);
        } else {
            downs[i % w_size] += powf(in[numel], 2);
        }
    }

    for (size_t i = 0; i < w_size; i++) {
        downs[i] = sqrt(downs[i] + eps);
    }

    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(weight[i % w_size])
                * llaisys::utils::cast<float>(in[i])
                / downs[i % w_size]
            );
        } else {
            out[i] = weight[i % w_size] * in[i] / downs[i % w_size];
        }
    }
    */

    // README 给的公式有问题!!!!!
    // $Y_i = \frac{W_i \times  X_i}{\sqrt{(\sum_{j=1}^n X_j^2) + \epsilon}}$ 不对
    // $Y_i = \frac{W_i \times  X_i}{\sqrt{(  \mathbf{\frac{1}{n}}  \sum_{j=1}^n X_j^2) + \epsilon}}$
    //                                         ^^^^^^^^^^^^^^^^^^
    size_t num_rows = numel / w_size;
    for (size_t row = 0; row < num_rows; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < w_size; ++col) {
            size_t idx = row * w_size + col;
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[idx]);
            } else {
                val = static_cast<float>(in[idx]);
            }
            sum_sq += val * val;
        }
        // 这里公式有问题
        float rms = sqrtf((sum_sq / static_cast<float>(w_size)) + eps);
        for (size_t col = 0; col < w_size; ++col) {
            size_t idx = row * w_size + col;
            float w_val;
            float in_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w_val = llaisys::utils::cast<float>(weight[col]);
                in_val = llaisys::utils::cast<float>(in[idx]);
            } else {
                w_val = static_cast<float>(weight[col]);
                in_val = static_cast<float>(in[idx]);
            }
            float out_val = w_val * in_val / rms;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[idx] = llaisys::utils::cast<T>(out_val);
            } else {
                out[idx] = static_cast<T>(out_val);
            }
        }
    }
}

#define DISPATCH_SELF_ATTENTION(dtype, ctype) case dtype: rms_norm_<ctype>(out, in, weight, w_size, numel, eps); break;

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t w_size, size_t numel, float eps) {
    switch (type) {
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F32, float)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
