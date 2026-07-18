#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <type_traits>
#include <cmath>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,float eps, size_t N, size_t D) {
    using acc_t = std::conditional_t<std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>, float, T>;

    for (size_t i = 0; i < N; ++i) {
        acc_t square_sum = 0;
        for (size_t j = 0; j < D; ++j) {
            if constexpr(std::is_same_v<T, acc_t>) {
                square_sum += in[i*D+j]*in[i*D+j];
            } else {
                auto x = llaisys::utils::cast<acc_t>(in[i*D + j]);
                square_sum += x*x;
            }
        }
        auto RMS = std::sqrt(square_sum/D+eps);
        for (size_t j = 0; j < D; ++j) {
            if constexpr (std::is_same_v<T, acc_t>) {
                out[i * D + j] = (in[i * D + j] / RMS) * weight[j];
            } else {
                auto x = llaisys::utils::cast<acc_t>(in[i * D + j]);
                auto w = llaisys::utils::cast<acc_t>(weight[j]);
                out[i * D + j] = llaisys::utils::cast<T>((x / RMS) * w);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,float eps, llaisysDataType_t dtype, size_t N, size_t D) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),eps, N, D);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight),eps, N, D);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight),eps, N, D);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu