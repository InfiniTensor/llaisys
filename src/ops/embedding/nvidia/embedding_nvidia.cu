#include "embedding_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t embedding_dim) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = num_indices * embedding_dim;
    if (idx >= total) {
        return;
    }

    size_t row = idx / embedding_dim;
    size_t col = idx % embedding_dim;
    int64_t src_row = index[row];
    out[idx] = weight[static_cast<size_t>(src_row) * embedding_dim + col];
}

template <typename T>
void launch_embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t num_indices,
    size_t embedding_dim) {
    const size_t total = num_indices * embedding_dim;
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(total, threads);
    auto stream = current_stream();
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        num_indices,
        embedding_dim);
    check_cuda(cudaGetLastError(), "embedding_kernel launch");
}

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding<float>(out, index, weight, num_indices, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return launch_embedding<llaisys::fp16_t>(out, index, weight, num_indices, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding<llaisys::bf16_t>(out, index, weight, num_indices, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
