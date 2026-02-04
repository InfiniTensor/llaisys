#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template <typename T>
void swiglu_cpu_(T *out, const T *gate, const T *up, size_t total_size) {
    // Compute: out[i] = up[i] * (gate[i] / (1 + exp(-gate[i])))
    // This is equivalent to: out[i] = up[i] * gate[i] * sigmoid(gate[i])
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, compute in float
        for (size_t i = 0; i < total_size; i++) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            
            // Compute gate / (1 + exp(-gate))
            // This is: gate * sigmoid(gate)
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // Multiply with up
            float result = up_val * silu;
            
            out[i] = llaisys::utils::cast<T>(result);
        }
    } else {
        // For standard types
        for (size_t i = 0; i < total_size; i++) {
            float gate_val = static_cast<float>(gate[i]);
            float up_val = static_cast<float>(up[i]);
            
            // Compute gate / (1 + exp(-gate))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // Multiply with up
            float result = up_val * silu;
            
            out[i] = static_cast<T>(result);
        }
    }
}

void swiglu_cpu(std::byte *out, const std::byte *gate, const std::byte *up,
                llaisysDataType_t dtype, size_t total_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_cpu_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(gate),
                          reinterpret_cast<const float *>(up),
                          total_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_cpu_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(gate),
                          reinterpret_cast<const llaisys::fp16_t *>(up),
                          total_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_cpu_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(gate),
                          reinterpret_cast<const llaisys::bf16_t *>(up),
                          total_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // Validate inputs
    CHECK_ARGUMENT(gate->ndim() == 2, "swiglu: gate must be 2-D tensor");
    CHECK_ARGUMENT(up->ndim() == 2, "swiglu: up must be 2-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "swiglu: out must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == gate->dtype(), "swiglu: out and gate must have same dtype");
    CHECK_ARGUMENT(out->dtype() == up->dtype(), "swiglu: out and up must have same dtype");
    CHECK_SAME_DEVICE(out, gate, up);
    
    CHECK_ARGUMENT(gate->shape()[0] == up->shape()[0], "swiglu: gate and up must have same shape[0]");
    CHECK_ARGUMENT(gate->shape()[1] == up->shape()[1], "swiglu: gate and up must have same shape[1]");
    CHECK_ARGUMENT(out->shape()[0] == gate->shape()[0], "swiglu: out must have same shape as gate");
    CHECK_ARGUMENT(out->shape()[1] == gate->shape()[1], "swiglu: out must have same shape as gate");
    
    size_t total_size = gate->shape()[0] * gate->shape()[1];
    
    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return swiglu_cpu(out->data(), gate->data(), up->data(),
                         out->dtype(), total_size);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return swiglu_cpu(out->data(), gate->data(), up->data(),
                         out->dtype(), total_size);
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
