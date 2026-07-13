#include "embedding_cu.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace embeddingops::nvidia {

// Each block handles one output token; threads stride across the embedding dimension.
template <typename T>
__global__ void
embedding_kernel(T *output, const int64_t *indices, const T *weights,
                 size_t num_indices, size_t embedding_dim) {
    size_t token = blockIdx.x;
    if (token >= num_indices) return;
    int64_t w_idx = indices[token];
    for (size_t d = threadIdx.x; d < embedding_dim; d += blockDim.x)
        output[token * embedding_dim + d] = weights[w_idx * embedding_dim + d];
}

} // namespace embeddingops::nvidia

namespace llaisys::ops::nvidia {

void embedding(std::byte *output,
               const std::byte *indices,
               const std::byte *weights,
               size_t num_indices,
               size_t embedding_dim,
               llaisysDataType_t dtype) {
    int threads = static_cast<int>(embedding_dim < 1024u ? embedding_dim : 1024u);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embeddingops::nvidia::embedding_kernel<float>
            <<<static_cast<unsigned int>(num_indices), threads>>>(
                reinterpret_cast<float *>(output),
                reinterpret_cast<const int64_t *>(indices),
                reinterpret_cast<const float *>(weights),
                num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embeddingops::nvidia::embedding_kernel<__half>
            <<<static_cast<unsigned int>(num_indices), threads>>>(
                reinterpret_cast<__half *>(output),
                reinterpret_cast<const int64_t *>(indices),
                reinterpret_cast<const __half *>(weights),
                num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embeddingops::nvidia::embedding_kernel<__nv_bfloat16>
            <<<static_cast<unsigned int>(num_indices), threads>>>(
                reinterpret_cast<__nv_bfloat16 *>(output),
                reinterpret_cast<const int64_t *>(indices),
                reinterpret_cast<const __nv_bfloat16 *>(weights),
                num_indices, embedding_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
