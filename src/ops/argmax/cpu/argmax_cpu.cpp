#include "argmax_cpu.hpp"
#include "../../../utils.hpp" 

namespace { 

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;

    int64_t best_idx = 0;
    T best_val_raw = vals[0];
    float best_val_f = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        float curr_val_f = llaisys::utils::cast<float>(vals[i]);
        if (curr_val_f > best_val_f) {
            best_val_f = curr_val_f;
            best_idx = i;
            best_val_raw = vals[i];
        }
    }

    *max_idx = best_idx;
    *max_val = best_val_raw;
}

} // namespace

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel) {
 
    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(max_idx_ptr, reinterpret_cast<float *>(max_val), 
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), 
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), 
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu