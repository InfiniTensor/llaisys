#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight,
                                size_t index_size, size_t embed_dim) {
    // Each thread handles one element of the embedding
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = index_size * embed_dim;

    if (idx < total_elements) {
        size_t i = idx / embed_dim;  // Which index
        size_t j = idx % embed_dim;  // Which dimension

        int64_t vocab_idx = index[i];
        out[idx] = weight[vocab_idx * embed_dim + j];
    }
}

template <typename T>
void embedding_(std::byte *out_bytes, const std::byte *index_bytes, const std::byte *weight_bytes,
                size_t index_size, size_t embed_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto index = reinterpret_cast<const int64_t *>(index_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);

    size_t total_elements = index_size * embed_dim;
    const int blockSize = 256;
    const int numBlocks = (total_elements + blockSize - 1) / blockSize;

    embeddingKernel<T><<<numBlocks, blockSize>>>(out, index, weight, index_size, embed_dim);
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_size, size_t embed_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight, index_size, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_<bf16_t_cuda>(out, index, weight, index_size, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_<fp16_t_cuda>(out, index, weight, index_size, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
