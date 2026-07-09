#include "rms_norm_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::ops::cuda {

__global__ void rms_norm_kernel(void *out, const void *in, const void *weight, llaisysDataType_t type, size_t feature_dim,
                                float eps) {
    size_t row = blockIdx.x;
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (size_t col = threadIdx.x; col < feature_dim; col += blockDim.x) {
        float val = load_as_float(in, row * feature_dim + col, type);
        local_sum += val * val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / static_cast<float>(feature_dim) + eps);
    for (size_t col = threadIdx.x; col < feature_dim; col += blockDim.x) {
        float x = load_as_float(in, row * feature_dim + col, type);
        float w = load_as_float(weight, col, type);
        store_from_float(out, row * feature_dim + col, x * w * inv_rms, type);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t batch_size,
              size_t feature_dim, float eps) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "rms_norm cuda only supports F32/F16/BF16");
    dim3 block(256);
    dim3 grid(batch_size);
    size_t smem = block.x * sizeof(float);
    rms_norm_kernel<<<grid, block, smem>>>(out, in, weight, type, feature_dim, eps);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "rms_norm cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
