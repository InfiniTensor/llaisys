#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
static void embedding_(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_len, size_t row_len) {
    const auto idx = reinterpret_cast<const int64_t *>(index);
    const auto w = reinterpret_cast<const T *>(weight);
    auto o = reinterpret_cast<T *>(out);
    for (size_t i = 0; i < index_len; ++i) {
        const int64_t row = idx[i];
        const T *src = w + row * row_len;
        T *dst = o + i * row_len;
        for (size_t j = 0; j < row_len; ++j) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t index_len, size_t row_len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight, index_len, row_len);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(out, index, weight, index_len, row_len);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(out, index, weight, index_len, row_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}