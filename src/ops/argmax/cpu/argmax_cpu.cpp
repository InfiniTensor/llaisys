#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <vector>

template <typename T>
void argmax_kernel(const T *vals, size_t numel, T *out_val, int64_t *out_idx) {
    if (numel == 0) {
        return;
    }

    T max_v_raw = vals[0];
    int64_t max_i = 0;

    float max_v_f32 = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; i++) {

        float current_val = llaisys::utils::cast<float>(vals[i]);

        if (current_val > max_v_f32) {
            max_v_f32 = current_val;
            max_v_raw = vals[i];
            max_i = i;
        }
    }

    *out_val = max_v_raw;
    *out_idx = max_i;
}

namespace llaisys::ops::cpu {

void argmax(const std::byte *vals, std::byte *max_val, std::byte *max_idx, llaisysDataType_t dtype, size_t numel) {

    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:

        return argmax_kernel<float>(reinterpret_cast<const float *>(vals), numel, reinterpret_cast<float *>(max_val), idx_ptr);

    case LLAISYS_DTYPE_F16:

        return argmax_kernel<llaisys::fp16_t>(reinterpret_cast<const llaisys::fp16_t *>(vals), numel, reinterpret_cast<llaisys::fp16_t *>(max_val), idx_ptr);

    case LLAISYS_DTYPE_BF16:

        return argmax_kernel<llaisys::bf16_t>(reinterpret_cast<const llaisys::bf16_t *>(vals), numel, reinterpret_cast<llaisys::bf16_t *>(max_val), idx_ptr);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu