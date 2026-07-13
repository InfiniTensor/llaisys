#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, 
                size_t num_indices, size_t vocab_size, size_t embedding_dim) {
    
    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = index[i];

        const T* src_row = weight + idx * embedding_dim;
        T* dst_row = out + i * embedding_dim;

        for (size_t j = 0; j < embedding_dim; ++j) {
            dst_row[j] = src_row[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t num_indices, size_t vocab_size, size_t embedding_dim) {
    
    const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx_ptr, 
                          reinterpret_cast<const float *>(weight), 
                          num_indices, vocab_size, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr, 
                          reinterpret_cast<const llaisys::bf16_t *>(weight), 
                          num_indices, vocab_size, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr, 
                          reinterpret_cast<const llaisys::fp16_t *>(weight), 
                          num_indices, vocab_size, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}