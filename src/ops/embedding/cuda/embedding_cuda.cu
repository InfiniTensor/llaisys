#include "embedding_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::ops::cuda {

__global__ void embedding_kernel(void *out, const int64_t *index, const void *weight, llaisysDataType_t dtype, size_t num,
                                 size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num * dim;
    if (idx >= total) {
        return;
    }

    size_t row = idx / dim;
    size_t col = idx % dim;
    int64_t token = index[row];
    size_t src_idx = static_cast<size_t>(token) * dim + col;
    float val = load_as_float(weight, src_idx, dtype);
    store_from_float(out, idx, val, dtype);
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t num,
               size_t dim) {
    ASSERT(dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16,
           "embedding cuda only supports F32/F16/BF16");
    size_t total = num * dim;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    embedding_kernel<<<grid, block>>>(out, reinterpret_cast<const int64_t *>(index), weight, dtype, num, dim);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "embedding cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
