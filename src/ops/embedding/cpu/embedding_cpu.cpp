#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight,
                size_t N, size_t D) {
    for (size_t i = 0; i < N; ++i) {
        size_t idx = index[i];
        for (size_t j = 0; j < D; ++j) {
            out[i * D + j] = weight[idx * D + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t weight_dtype,
               size_t N, size_t D) {
    switch (weight_dtype) {
    case LLAISYS_DTYPE_F32: {
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), N, D);
    }
    case LLAISYS_DTYPE_BF16: {
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), N, D);
    }
    case LLAISYS_DTYPE_F16: {
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), N, D);
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(weight_dtype);
    }
}
} // namespace llaisys::ops::cpu