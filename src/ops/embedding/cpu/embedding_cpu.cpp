#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t idx_len, size_t embed_dim) {
    for (size_t i = 0; i < idx_len; i++) {
        int64_t idx = index[i];
        const T *src_row = weight + idx * embed_dim;  // 源行指针
        T *dst_row = out + i * embed_dim;             // 目标行指针
        
        // 复制整行数据
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // 对于半精度类型，逐元素复制
            for (size_t j = 0; j < embed_dim; j++) {
                dst_row[j] = src_row[j];
            }
        } else {
            // 对于标准类型，使用内存复制
            std::memcpy(dst_row, src_row, embed_dim * sizeof(T));
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t idx_len, size_t embed_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            idx_len, embed_dim
        );
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            idx_len, embed_dim
        );
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            idx_len, embed_dim
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
