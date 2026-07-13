#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

template <typename T, typename I>
void embedding_(T *out, const I *index, const T *weight, size_t num_indices, size_t hidden_size) {
    for (size_t i = 0; i < num_indices; i++) {
        memcpy(out + i * hidden_size, weight + index[i] * hidden_size, hidden_size * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices, size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), num_indices, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), num_indices, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), num_indices, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
