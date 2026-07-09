#include "swiglu_cuda.cuh"
#include "../../cuda_utils.cuh"

__global__ void swiglu_kernel(void *out, const void *gate, const void *up,
                              llaisysDataType_t dtype, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = load_as_f32(gate, idx, dtype);
    float u = load_as_f32(up, idx, dtype);
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    store_from_f32(out, idx, u * g * sigmoid_g, dtype);
}

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    swiglu_kernel<<<cuda_grid_size(numel), CUDA_BLOCK_SIZE>>>(out, gate, up, dtype, numel);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
