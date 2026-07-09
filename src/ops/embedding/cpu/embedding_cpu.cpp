#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(T *out, const size_t *index, const T *weight,
                const size_t idx_numel, const size_t dim0, const size_t dim1) {
    // 每行拷贝的字节数
    const size_t row_bytes = dim1 * sizeof(T);

    for (size_t i = 0; i < idx_numel; ++i) {
        size_t idx = index[i];

        // 边界检查：确保索引不越界
        ASSERT(idx < dim0, "Embedding: index out of vocabulary range");

        // 地址计算
        T *dst_ptr = out + (i * dim1);
        const T *src_ptr = weight + (idx * dim1);

        // 执行拷贝
        memcpy(dst_ptr, src_ptr, row_bytes);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, const size_t idx_numel, const size_t dim0, const size_t dim1) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const size_t *>(index),
                          reinterpret_cast<const float *>(weight), idx_numel, dim0, dim1);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const size_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), idx_numel, dim0, dim1);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const size_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), idx_numel, dim0, dim1);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu