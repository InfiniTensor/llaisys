#include "embedding_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t index_numel, size_t dim,
                                 size_t vocab) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = index_numel * dim;
    if (idx >= total) {
        return;
    }
    size_t row = idx / dim;
    size_t col = idx % dim;
    int64_t token = index[row];
    if (token < 0 || static_cast<size_t>(token) >= vocab) {
        return;
    }
    size_t w_idx = static_cast<size_t>(token) * dim + col;
    float v = llaisys::device::nvidia::ScalarOps<T>::load(weight + w_idx);
    llaisys::device::nvidia::ScalarOps<T>::store(out + idx, v);
}

template <typename T>
void launch_embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_numel,
                      size_t dim, size_t vocab) {
    size_t total = index_numel * dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    embedding_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const int64_t *>(index),
                                          reinterpret_cast<const T *>(weight), index_numel, dim, vocab);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t index_numel, size_t embd_dim, size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding<float>(out, index, weight, index_numel, embd_dim, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding<llaisys::bf16_t>(out, index, weight, index_numel, embd_dim, weight_rows);
    case LLAISYS_DTYPE_F16:
        return launch_embedding<llaisys::fp16_t>(out, index, weight, index_numel, embd_dim, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
