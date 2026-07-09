#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <limits>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num , size_t dim ) {
    for(size_t i = 0;i < num; i++)
        for(size_t j = 0;j < dim ; j++)
            out[i * dim + j] = weight[index[i] * dim + j];
}
namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num , size_t dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), num , dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), num , dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), num , dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
