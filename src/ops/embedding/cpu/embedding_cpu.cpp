#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <cstdint>

// CPU 侧实现：逐行从 weight 中按 index 拷贝到 out
// out[i, :] = weight[index[i], :]
template <typename T>
void embedding_(T *out,
                const int64_t *index,
                const T *weight,
                size_t index_numel,
                size_t embedding_dim) {
        for (size_t i = 0; i < index_numel; i++) {
            int64_t cur_idx = index[i];
            for (size_t j = 0; j < embedding_dim; j++) {
                out[i * embedding_dim + j] = weight[cur_idx * embedding_dim + j];
            }
        }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t index_numel,
               size_t embedding_dim) {
    // index 在 op 层已经保证是 I64，这里直接按 int64_t 解释
    const auto *index_i64 = reinterpret_cast<const int64_t *>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          index_i64,
                          reinterpret_cast<const float *>(weight),
                          index_numel,
                          embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          index_i64,
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          index_numel,
                          embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          index_i64,
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          index_numel,
                          embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
