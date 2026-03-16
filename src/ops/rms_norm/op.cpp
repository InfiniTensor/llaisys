#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

// Y_i = W_i * X_i / sqrt( (1/d) * sum_j(X_j^2) + eps ), 按行计算
template <typename T>
void rms_norm_impl(T *out, const T *in, const T *weight, size_t rows, size_t d, float eps) {
    const float inv_d = 1.0f / static_cast<float>(d);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < rows; i++) {
        const T *row_in = in + i * d;
        T *row_out = out + i * d;

        float sum_sq;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            sum_sq = 0;
            for (size_t j = 0; j < d; j++) {
                float x = llaisys::utils::cast<float>(row_in[j]);
                sum_sq += x * x;
            }
        } else {
            sum_sq = 0;
            for (size_t j = 0; j < d; j++) {
                float x = static_cast<float>(row_in[j]);
                sum_sq += x * x;
            }
        }
        float rms = std::sqrt(inv_d * sum_sq + eps);

        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float x = llaisys::utils::cast<float>(row_in[j]);
                float w = llaisys::utils::cast<float>(weight[j]);
                row_out[j] = llaisys::utils::cast<T>(w * x / rms);
            } else {
                row_out[j] = static_cast<T>(static_cast<float>(weight[j]) * static_cast<float>(row_in[j]) / rms);
            }
        }
    }
}

void rms_norm_cpu(std::byte *out, const std::byte *in, const std::byte *weight,
                  llaisysDataType_t dtype, size_t rows, size_t d, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                      reinterpret_cast<const float *>(weight), rows, d, eps);
        return;
    case LLAISYS_DTYPE_F16:
        rms_norm_impl(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                      reinterpret_cast<const llaisys::fp16_t *>(weight), rows, d, eps);
        return;
    case LLAISYS_DTYPE_BF16:
        rms_norm_impl(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                      reinterpret_cast<const llaisys::bf16_t *>(weight), rows, d, eps);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "rms_norm: out, in, weight must have same dtype");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: out, in, weight must be contiguous");
    ASSERT(out->ndim() == 2 && in->ndim() == 2, "rms_norm: out, in must be 2D");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be 1D");
    size_t rows = in->shape()[0], d = in->shape()[1];
    ASSERT(out->shape()[0] == rows && out->shape()[1] == d, "rms_norm: out shape must match in");
    ASSERT(weight->numel() == d, "rms_norm: weight size must equal last dim of in");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return rms_norm_cpu(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
