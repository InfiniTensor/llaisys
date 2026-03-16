#include "rms_norm_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <cmath>

// Each block handles one row. Block-level reduction for sum of squares.
__global__ void rms_norm_kernel(void *out, const void *in, const void *weight,
                                float eps, llaisysDataType_t dtype,
                                size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = load_as_f32(in, row * cols + c, dtype);
        local_sum += v * v;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / static_cast<float>(cols) + eps);

    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = load_as_f32(in, row * cols + c, dtype);
        float w = load_as_f32(weight, c, dtype);
        store_from_f32(out, row * cols + c, w * v * rms, dtype);
    }
}

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t rows, size_t cols) {
    int block_size = 256;
    if (cols > 256) block_size = 512;
    if (cols > 512) block_size = 1024;
    size_t shared_mem = block_size * sizeof(float);
    rms_norm_kernel<<<rows, block_size, shared_mem>>>(out, in, weight, eps, dtype, rows, cols);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
