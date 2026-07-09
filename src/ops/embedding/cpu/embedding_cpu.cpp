#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>
#include <cstdint>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_indices, size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = index[i];
        // 将 weight 中的第 idx 行复制到 out 中的第 i 行
        const T *src = weight + (idx * embedding_dim);
        T *dst = out + (i * embedding_dim);
        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embedding_dim) {
    
    // index 固定为 Int64 (int64_t)
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx_ptr, 
                          reinterpret_cast<const float *>(weight), num_indices, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,
                          reinterpret_cast<const llaisys::bf16_t *>(weight), num_indices, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr,
                          reinterpret_cast<const llaisys::fp16_t *>(weight), num_indices, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu