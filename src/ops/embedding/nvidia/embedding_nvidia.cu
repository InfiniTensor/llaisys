#include "embedding_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t numel, size_t embedding_dim) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t total = numel * embedding_dim;
    if (idx >= total) {
        return;
    }

    const size_t row = idx / embedding_dim;
    const size_t col = idx % embedding_dim;
    out[idx] = weight[static_cast<size_t>(index[row]) * embedding_dim + col];
}

template <typename T>
void embedding_impl(std::byte *out, const std::byte *index, const std::byte *weight, size_t numel, size_t embedding_dim) {
    constexpr int threads = 256;
    const size_t total = numel * embedding_dim;
    embedding_kernel<<<llaisys::ops::nvidia::blocks_for(total, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        numel,
        embedding_dim);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t numel, size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl<float>(out, index, weight, numel, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_impl<llaisys::fp16_t>(out, index, weight, numel, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl<llaisys::bf16_t>(out, index, weight, numel, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
