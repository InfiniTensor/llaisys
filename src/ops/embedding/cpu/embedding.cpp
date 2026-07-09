#include "embedding.hpp"

#include "../../../utils.hpp"
#include "../../../core/llaisys_core.hpp"

template <typename T>
static void embedding_(
    T* out,
    const int64_t* index,
    const T* weight,
    std::size_t embedding_dim,
    std::size_t index_size
) {
    for (size_t i = 0; i < index_size; ++i) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            out + i * embedding_dim,
            weight + index[i] * embedding_dim,
            embedding_dim * sizeof(T),
            LLAISYS_MEMCPY_H2H
        );
    }
}

namespace llaisys::ops::cpu {
void embedding(
    void* out,
    const void* index,
    const void* weight,
    size_t index_size,
    size_t embedding_dim,
    llaisysDataType_t data_type
) {
    switch (data_type) {
        case LLAISYS_DTYPE_F32:
            embedding_(
                reinterpret_cast<float*>(out),
                reinterpret_cast<const int64_t*>(index),
                reinterpret_cast<const float*>(weight),
                embedding_dim,
                index_size
            );
            break;
        case LLAISYS_DTYPE_F16:
            embedding_(
                reinterpret_cast<llaisys::bf16_t*>(out),
                static_cast<const int64_t*>(index),
                reinterpret_cast<const llaisys::bf16_t*>(weight),
                embedding_dim,
                index_size
            );
            break;
        case LLAISYS_DTYPE_BF16:
            embedding_(
                reinterpret_cast<llaisys::fp16_t*>(out),
                static_cast<const int64_t*>(index),
                reinterpret_cast<const llaisys::fp16_t*>(weight),
                embedding_dim,
                index_size
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type in embedding operation.");
    }
}
} // namespace llaisys::ops::cpu