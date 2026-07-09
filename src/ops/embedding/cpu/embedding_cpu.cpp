#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight,
                size_t seq_len, size_t hidden_size) {
    // Copy rows from weight matrix according to index
    for (size_t i = 0; i < seq_len; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * hidden_size;
        T *dst = out + i * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t hidden_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            seq_len, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            seq_len, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            seq_len, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
