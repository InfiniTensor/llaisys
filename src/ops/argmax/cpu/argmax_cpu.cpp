#include "argmax_cpu.hpp"
#include "../../../utils.hpp"

#include <cstdint>

namespace {

// T: element type of vals/max_val
template <typename T>
inline void argmax_impl(int64_t *out_idx,
                        T *out_val,
                        const T *vals,
                        size_t n) {
    // n > 0 is guaranteed by caller
    size_t best_i = 0;
    float best_f = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < n; ++i) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best_f) {
            best_f = v;
            best_i = i;
        }
    }

    *out_idx = static_cast<int64_t>(best_i);
    // Write back the original-typed value (not the float-casted one)
    *out_val = vals[best_i];
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void argmax(std::byte *out_idx,
            std::byte *out_val,
            const std::byte *vals,
            llaisysDataType_t type,
            size_t numel) {
    ASSERT(numel > 0, "argmax: input must have at least one element");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl(reinterpret_cast<int64_t *>(out_idx),
                           reinterpret_cast<float *>(out_val),
                           reinterpret_cast<const float *>(vals),
                           numel);
    case LLAISYS_DTYPE_F16:
        return argmax_impl(reinterpret_cast<int64_t *>(out_idx),
                           reinterpret_cast<llaisys::fp16_t *>(out_val),
                           reinterpret_cast<const llaisys::fp16_t *>(vals),
                           numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl(reinterpret_cast<int64_t *>(out_idx),
                           reinterpret_cast<llaisys::bf16_t *>(out_val),
                           reinterpret_cast<const llaisys::bf16_t *>(vals),
                           numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
