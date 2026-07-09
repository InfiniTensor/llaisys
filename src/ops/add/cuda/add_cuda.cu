#include "add_cuda.cuh"
#include "../../cuda_utils.cuh"

__global__ void add_kernel(void *c, const void *a, const void *b,
                           llaisysDataType_t dtype, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float va = load_as_f32(a, idx, dtype);
    float vb = load_as_f32(b, idx, dtype);
    store_from_f32(c, idx, va + vb, dtype);
}

namespace llaisys::ops::cuda {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    add_kernel<<<cuda_grid_size(numel), CUDA_BLOCK_SIZE>>>(c, a, b, type, numel);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
