#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(std::byte *out_raw, const int64_t *index, const size_t i_size,
                const std::byte *weight_raw, const size_t w_rows, const size_t w_cols) {
    T * out = reinterpret_cast<T*>(out_raw);
    const T * weight = reinterpret_cast<const T*>(weight_raw);
    for (size_t i = 0; i < i_size; i++) {
        for (size_t j = 0; j < w_cols; j++) {
            out[i * w_cols + j] = weight[index[i] * w_cols + j];
        }
    }
}

#define DISPATCH_EMBEDDING(dtype, ctype) case dtype: embedding_<ctype>(out, reinterpret_cast<const int64_t *>(index), i_size, weight, w_rows, w_cols); break;

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const size_t i_size,
               const std::byte *weight, const size_t w_rows, const size_t w_cols,
               llaisysDataType_t type) {
    switch (type) {
        DISPATCH_EMBEDDING(LLAISYS_DTYPE_F32, float)
        DISPATCH_EMBEDDING(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_EMBEDDING(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_EMBEDDING(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_EMBEDDING(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
