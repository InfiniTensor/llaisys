#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"
#include <cstddef>

namespace {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t index_numel, size_t embedding_dim) {
    const size_t row = blockIdx.x;
    if (row >= index_numel) {
        return;
    }

    const int64_t idx = index[row];
    const size_t in_start = static_cast<size_t>(idx) * embedding_dim;
    const size_t out_start = row * embedding_dim;

    for (size_t col = threadIdx.x; col < embedding_dim; col += blockDim.x) {
        out[out_start + col] = weight[in_start + col];
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_numel,
               size_t embedding_dim) {

    const int block_size = 256;
    const int grid_size = index_numel;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight), index_numel, embedding_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<half><<<grid_size, block_size>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const half *>(weight), index_numel, embedding_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __nv_bfloat16 *>(weight), index_numel,
            embedding_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
