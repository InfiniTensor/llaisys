#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstdint>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

template <typename T>
void argmax_impl(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;
    size_t idx = 0;
    T best = vals[0];
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float best_f = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; i++) {
            float v = llaisys::utils::cast<float>(vals[i]);
            if (v > best_f) {
                best_f = v;
                best = vals[i];
                idx = i;
            }
        }
    } else {
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > best) {
                best = vals[i];
                idx = i;
            }
        }
    }
    *max_idx = static_cast<int64_t>(idx);
    *max_val = best;
}

void argmax_cpu(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t vals_type, size_t numel) {
    int64_t *out_idx = reinterpret_cast<int64_t *>(max_idx);
    switch (vals_type) {
    case LLAISYS_DTYPE_F32:
        argmax_impl(out_idx, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
        return;
    case LLAISYS_DTYPE_F16:
        argmax_impl(out_idx, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        argmax_impl(out_idx, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals_type);
    }
}

} // namespace

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be int64");
    ASSERT(max_val->dtype() == vals->dtype(), "argmax: max_val dtype must match vals");
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "argmax: all tensors must be contiguous");
    ASSERT(vals->ndim() == 1, "argmax: vals must be 1D");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1, "argmax: max_idx and max_val must have one element");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return argmax_cpu(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return argmax_cpu(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
