#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t idx_size, size_t embd_dim) {
    for (size_t i = 0; i < idx_size; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * embd_dim;
        T *dst = out + i * embd_dim;
        for (size_t j = 0; j < embd_dim; j++) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t idx_size, size_t embd_dim) {
    auto idx = reinterpret_cast<const int64_t *>(index);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx, reinterpret_cast<const float *>(weight), idx_size, embd_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx, reinterpret_cast<const llaisys::bf16_t *>(weight), idx_size, embd_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx, reinterpret_cast<const llaisys::fp16_t *>(weight), idx_size, embd_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
