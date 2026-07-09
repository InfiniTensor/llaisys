#include "argmax_cpu.hpp"
#include "../../../utils.hpp"

template <typename ValT, typename IdxT>
void argmax_(IdxT *max_idx, ValT *max_val, const ValT *vals, size_t numel) {
    // Find max value and its index
    ValT max_v = vals[0];
    IdxT max_i = 0;

    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<ValT, llaisys::bf16_t> || std::is_same_v<ValT, llaisys::fp16_t>) {
            float curr = llaisys::utils::cast<float>(vals[i]);
            float max_f = llaisys::utils::cast<float>(max_v);
            if (curr > max_f) {
                max_v = vals[i];
                max_i = static_cast<IdxT>(i);
            }
        } else {
            if (vals[i] > max_v) {
                max_v = vals[i];
                max_i = static_cast<IdxT>(i);
            }
        }
    }

    max_idx[0] = max_i;
    max_val[0] = max_v;
}

template <typename ValT>
void argmax_dispatch_idx(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                         llaisysDataType_t idx_type, size_t numel) {
    switch (idx_type) {
    case LLAISYS_DTYPE_I32:
        return argmax_<ValT, int32_t>(
            reinterpret_cast<int32_t *>(max_idx),
            reinterpret_cast<ValT *>(max_val),
            reinterpret_cast<const ValT *>(vals),
            numel);
    case LLAISYS_DTYPE_I64:
        return argmax_<ValT, int64_t>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<ValT *>(max_val),
            reinterpret_cast<const ValT *>(vals),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t idx_type, llaisysDataType_t val_type, size_t numel) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return argmax_dispatch_idx<float>(max_idx, max_val, vals, idx_type, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_dispatch_idx<llaisys::bf16_t>(max_idx, max_val, vals, idx_type, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_dispatch_idx<llaisys::fp16_t>(max_idx, max_val, vals, idx_type, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
