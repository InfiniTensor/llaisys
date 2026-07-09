#include "embedding_cpu.hpp"
#include <cstring>


template <typename T>
void embedding_(const T *weight_data, const int64_t *index_data, T *out_data,
                size_t embedding_dim, size_t num_embeddings, size_t num_indices) {

    for (size_t i = 0; i < num_indices; i++) {
        int64_t idx = index_data[i];

        // 越界保护
        if (idx < 0 || static_cast<size_t>(idx) >= num_embeddings) {
            continue;
        }

        const T *src = weight_data + idx * embedding_dim;
        T *dst = out_data + i * embedding_dim;

        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {

void embedding(const std::byte *weight_data, const std::byte *index_data, std::byte *out_data,
               llaisysDataType_t dtype,
               size_t embedding_dim, size_t num_embeddings, size_t num_indices) {

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index_data);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(
            reinterpret_cast<const float *>(weight_data),
            idx_ptr,
            reinterpret_cast<float *>(out_data),
            embedding_dim, num_embeddings, num_indices);

    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(weight_data),
            idx_ptr,
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            embedding_dim, num_embeddings, num_indices);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(weight_data),
            idx_ptr,
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            embedding_dim, num_embeddings, num_indices);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu