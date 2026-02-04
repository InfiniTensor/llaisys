#include "op.hpp"

namespace llaisys::ops {
template <typename T>
void argmax_cpu_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;
    
    size_t idx = 0;
    T val;
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float max_val_f = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; i++) {
            float current = llaisys::utils::cast<float>(vals[i]);
            if (current > max_val_f) {
                max_val_f = current;
                idx = i;
            }
        }
        val = llaisys::utils::cast<T>(max_val_f);
    } else {
        val = vals[0];
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > val) {
                val = vals[i];
                idx = i;
            }
        }
    }
    
    max_idx[0] = static_cast<int64_t>(idx);
    max_val[0] = val;
}

void argmax_cpu(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
                llaisysDataType_t dtype, size_t numel) {
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_cpu_(idx_ptr, reinterpret_cast<float *>(max_val), 
                          reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_cpu_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), 
                          reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_cpu_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), 
                          reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // Validate inputs
    CHECK_ARGUMENT(max_idx->numel() == 1, "argmax: max_idx must have 1 element");
    CHECK_ARGUMENT(max_val->numel() == 1, "argmax: max_val must have 1 element");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be i64");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(), "argmax: max_val must have same dtype as vals");
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    // Always support CPU calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return argmax_cpu(max_idx->data(), max_val->data(), vals->data(), 
                         vals->dtype(), vals->numel());
    }
    
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return argmax_cpu(max_idx->data(), max_val->data(), vals->data(), 
                         vals->dtype(), vals->numel());
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
