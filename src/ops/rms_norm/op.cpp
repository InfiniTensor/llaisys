#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template <typename T>
void rms_norm_cpu_(T *out, const T *in, const T *weight, float eps,
                   size_t batch, size_t d) {
    // For each row, compute: Y_i = (W_i * X_i) / sqrt((1/d) * sum(X_j^2) + eps)
    
    for (size_t b = 0; b < batch; b++) {
        // Compute RMS for this row
        float sum_sq = 0.0f;
        
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // For half precision, compute in float
            for (size_t i = 0; i < d; i++) {
                float val = llaisys::utils::cast<float>(in[b * d + i]);
                sum_sq += val * val;
            }
            
            // Compute RMS: sqrt((1/d) * sum_sq + eps)
            float rms = std::sqrt(sum_sq / d + eps);
            
            // Apply normalization and weight
            for (size_t i = 0; i < d; i++) {
                float in_val = llaisys::utils::cast<float>(in[b * d + i]);
                float w_val = llaisys::utils::cast<float>(weight[i]);
                float normalized = (in_val / rms) * w_val;
                out[b * d + i] = llaisys::utils::cast<T>(normalized);
            }
        } else {
            // For standard types
            for (size_t i = 0; i < d; i++) {
                T val = in[b * d + i];
                sum_sq += static_cast<float>(val) * static_cast<float>(val);
            }
            
            // Compute RMS: sqrt((1/d) * sum_sq + eps)
            float rms = std::sqrt(sum_sq / d + eps);
            
            // Apply normalization and weight
            for (size_t i = 0; i < d; i++) {
                float in_val = static_cast<float>(in[b * d + i]);
                float w_val = static_cast<float>(weight[i]);
                float normalized = (in_val / rms) * w_val;
                out[b * d + i] = static_cast<T>(normalized);
            }
        }
    }
}

void rms_norm_cpu(std::byte *out, const std::byte *in, const std::byte *weight,
                  float eps, llaisysDataType_t dtype, size_t batch, size_t d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_cpu_(reinterpret_cast<float *>(out),
                            reinterpret_cast<const float *>(in),
                            reinterpret_cast<const float *>(weight),
                            eps, batch, d);
    case LLAISYS_DTYPE_F16:
        return rms_norm_cpu_(reinterpret_cast<llaisys::fp16_t *>(out),
                            reinterpret_cast<const llaisys::fp16_t *>(in),
                            reinterpret_cast<const llaisys::fp16_t *>(weight),
                            eps, batch, d);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_cpu_(reinterpret_cast<llaisys::bf16_t *>(out),
                            reinterpret_cast<const llaisys::bf16_t *>(in),
                            reinterpret_cast<const llaisys::bf16_t *>(weight),
                            eps, batch, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Validate inputs
    CHECK_ARGUMENT(in->ndim() == 2, "rms_norm: in must be 2-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm: weight must be 1-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "rms_norm: out must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "rms_norm: out and in must have same dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "rms_norm: out and weight must have same dtype");
    CHECK_SAME_DEVICE(out, in, weight);
    
    size_t batch = in->shape()[0];
    size_t d = in->shape()[1];
    
    CHECK_ARGUMENT(weight->shape()[0] == d, "rms_norm: weight shape must match in shape[1]");
    CHECK_ARGUMENT(out->shape()[0] == batch, "rms_norm: out shape[0] must match in shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == d, "rms_norm: out shape[1] must match in shape[1]");
    
    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return rms_norm_cpu(out->data(), in->data(), weight->data(),
                           eps, out->dtype(), batch, d);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return rms_norm_cpu(out->data(), in->data(), weight->data(),
                           eps, out->dtype(), batch, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
