#include "embedding_nvidia.hpp"

#include "../../nvidia_utils.cuh"
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t index_len, size_t row_len) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = index_len * row_len;
    if (idx >= total) {
        return;
    }
    size_t i = idx / row_len;
    size_t j = idx - i * row_len;
    int64_t row = index[i];
    out[idx] = weight[static_cast<size_t>(row) * row_len + j];
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t index_len, size_t row_len) {
    size_t total = index_len * row_len;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            index_len,
            row_len);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            index_len,
            row_len);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            index_len,
            row_len);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
