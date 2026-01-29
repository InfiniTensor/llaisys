#include "argmax_cpu.hpp"
#include "../../../utils.hpp"


namespace llaisys::ops::cpu {

template <typename T>
void argmax_t(const T* vals, size_t numel, int64_t* idx_out, std::byte* val_out) {
    size_t best = 0;
    float best_v = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best_v) {
            best = i;
            best_v = v;
        }
    }
    *idx_out = static_cast<int64_t>(best);
    reinterpret_cast<T*>(val_out)[0] = vals[best];
}

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t dtype, size_t numel) {
    auto* idx_ptr = reinterpret_cast<int64_t*>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_t(reinterpret_cast<const float*>(vals), numel, idx_ptr, max_val);
    case LLAISYS_DTYPE_F16:
        return argmax_t(reinterpret_cast<const llaisys::fp16_t*>(vals), numel, idx_ptr, max_val);
    case LLAISYS_DTYPE_BF16:
        return argmax_t(reinterpret_cast<const llaisys::bf16_t*>(vals), numel, idx_ptr, max_val);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}
