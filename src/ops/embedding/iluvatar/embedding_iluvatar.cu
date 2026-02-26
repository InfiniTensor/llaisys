#include "embedding_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::iluvatar {

template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight, 
                                size_t index_size, size_t embd_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= index_size * embd_dim) return;
    
    size_t row = idx / embd_dim;
    size_t col = idx % embd_dim;
    
    int64_t weight_row = index[row];
    out[idx] = weight[weight_row * embd_dim + col];
}

template <typename T>
void embedding_(std::byte *out_bytes, const std::byte *index_bytes, const std::byte *weight_bytes, 
                size_t index_size, size_t embd_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto index = reinterpret_cast<const int64_t *>(index_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);
    
    size_t total = index_size * embd_dim;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;
    
    embeddingKernel<T><<<numBlocks, blockSize>>>(out, index, weight, index_size, embd_dim);
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t index_size, size_t embd_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight, index_size, embd_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_<bf16_t_cuda>(out, index, weight, index_size, embd_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_<fp16_t_cuda>(out, index, weight, index_size, embd_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
