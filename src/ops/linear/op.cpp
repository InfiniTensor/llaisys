#include "op.hpp"

namespace llaisys::ops {
template <typename T>
void linear_cpu_(T *out, const T *in, const T *weight, const T *bias,
                 size_t batch, size_t in_features, size_t out_features) {
    // Compute Y = X @ W^T + b
    // out: [batch, out_features]
    // in: [batch, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features] (optional)
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            // Compute dot product of in[b, :] and weight[o, :]
            float sum = 0.0f;
            
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // For half precision, compute in float
                for (size_t i = 0; i < in_features; i++) {
                    float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    float weight_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                    sum += in_val * weight_val;
                }
                
                // Add bias if provided
                if (bias != nullptr) {
                    sum += llaisys::utils::cast<float>(bias[o]);
                }
                
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                // For standard types
                for (size_t i = 0; i < in_features; i++) {
                    sum += in[b * in_features + i] * weight[o * in_features + i];
                }
                
                // Add bias if provided
                if (bias != nullptr) {
                    sum += bias[o];
                }
                
                out[b * out_features + o] = static_cast<T>(sum);
            }
        }
    }
}

void linear_cpu(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                llaisysDataType_t dtype, size_t batch, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_cpu_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight),
                          bias ? reinterpret_cast<const float *>(bias) : nullptr,
                          batch, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_cpu_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                          batch, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_cpu_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                          batch, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Validate inputs
    CHECK_ARGUMENT(in->ndim() == 2, "linear: in must be 2-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "linear: weight must be 2-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "linear: out must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "linear: out and in must have same dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "linear: out and weight must have same dtype");
    
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "linear: bias must be 1-D tensor");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(), "linear: bias must have same dtype as out");
        CHECK_SAME_DEVICE(out, in, weight, bias);
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
    }
    
    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    CHECK_ARGUMENT(weight->shape()[1] == in_features, "linear: weight shape[1] must match in shape[1]");
    CHECK_ARGUMENT(out->shape()[0] == batch, "linear: out shape[0] must match in shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == out_features, "linear: out shape[1] must match weight shape[0]");
    
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == out_features, "linear: bias shape must match weight shape[0]");
    }
    
    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return linear_cpu(out->data(), in->data(), weight->data(),
                         bias ? bias->data() : nullptr,
                         out->dtype(), batch, in_features, out_features);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return linear_cpu(out->data(), in->data(), weight->data(),
                         bias ? bias->data() : nullptr,
                         out->dtype(), batch, in_features, out_features);
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
