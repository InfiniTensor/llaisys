#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cstdint>

template <typename T>
static void argmax_impl(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T current_max = vals[0];
    int64_t current_max_idx = 0;
    for (size_t i = 1; i < numel; ++i) {
        if (casting(float, vals[i]) > casting(float, current_max)) {
            current_max = vals[i];
            current_max_idx = static_cast<int64_t>(i);
        }
    }
    *max_val = current_max;
    *max_idx = current_max_idx;
}

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_impl(reinterpret_cast<int64_t *>(max_idx),
                    reinterpret_cast<float *>(max_val),
                    reinterpret_cast<const float *>(vals),
                    numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl(reinterpret_cast<int64_t *>(max_idx),
                    reinterpret_cast<llaisys::fp16_t *>(max_val),
                    reinterpret_cast<const llaisys::fp16_t *>(vals),
                    numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl(reinterpret_cast<int64_t *>(max_idx),
                    reinterpret_cast<llaisys::bf16_t *>(max_val),
                    reinterpret_cast<const llaisys::bf16_t *>(vals),
                    numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu