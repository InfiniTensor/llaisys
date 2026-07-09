#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_indices, size_t weight_rows, size_t weight_cols) {
    for (size_t i = 0; i < num_indices; i++) {
        int64_t row = index[i];
        const T *src = weight + static_cast<size_t>(row) * weight_cols;
        T *dst = out + i * weight_cols;
        std::memcpy(dst, src, weight_cols * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t dtype, size_t num_indices,
               size_t weight_rows, size_t weight_cols) {
    auto *index_ptr = reinterpret_cast<const int64_t *>(index);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), index_ptr, reinterpret_cast<const float *>(weight),
                          num_indices, weight_rows, weight_cols);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), index_ptr,
                          reinterpret_cast<const llaisys::fp16_t *>(weight), num_indices, weight_rows, weight_cols);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), index_ptr,
                          reinterpret_cast<const llaisys::bf16_t *>(weight), num_indices, weight_rows, weight_cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
