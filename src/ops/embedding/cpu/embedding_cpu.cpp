#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void embedding_(std::byte *out_bytes, const std::byte *index_bytes, const std::byte *weight_bytes, size_t index_size, size_t embed_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto index = reinterpret_cast<const int64_t *>(index_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);

    for (size_t i = 0; i < index_size; ++i) {
        int64_t idx = index[i];
        const T *src = weight + idx * embed_dim;
        T *dst = out + i * embed_dim;
        std::memcpy(dst, src, embed_dim * sizeof(T));
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t index_size, size_t embed_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight, index_size, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(out, index, weight, index_size, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(out, index, weight, index_size, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu