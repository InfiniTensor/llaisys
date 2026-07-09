#include "argmax.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
static void argmax_(size_t *max_idx, T *max_val, const T *input, size_t numel) {
   if (numel == 0) {
       return;
   }
   *max_idx = static_cast<size_t>(0);
   *max_val = input[0];
   for(size_t i = 1; i < numel; ++i) {
       if (input[i] > *max_val) {
           *max_val = input[i];
           *max_idx = i;
       }
   }
}

namespace llaisys::ops::cpu {
void argmax(size_t *max_idx, std::byte *max_val, const std::byte *input, llaisysDataType_t val_type, size_t numel) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        argmax_(max_idx, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(input), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_(max_idx, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(input), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_(max_idx, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(input), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
