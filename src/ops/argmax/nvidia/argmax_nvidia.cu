#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace llaisys::ops::nvidia {

// Simpler approach: single block reduction for small sizes
template <typename T>
__global__ void argmaxSimpleKernel(const T *vals, T *max_val, int64_t *max_idx, size_t size) {
    extern __shared__ float svals[];
    int64_t *sidxs = (int64_t*)&svals[blockDim.x];

    unsigned int tid = threadIdx.x;

    // Initialize
    svals[tid] = -FLT_MAX;
    sidxs[tid] = 0;

    // Each thread processes multiple elements
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float v = to_float_cuda(vals[i]);
        if (v > svals[tid]) {
            svals[tid] = v;
            sidxs[tid] = i;
        }
    }
    __syncthreads();

    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (svals[tid + s] > svals[tid]) {
                svals[tid] = svals[tid + s];
                sidxs[tid] = sidxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = from_float_cuda<T>(svals[0]);
        *max_idx = sidxs[0];
    }
}

template <typename T>
void argmax_(std::byte *max_idx_bytes, std::byte *max_val_bytes, const std::byte *vals_bytes, size_t size) {
    auto max_idx = reinterpret_cast<int64_t *>(max_idx_bytes);
    auto max_val = reinterpret_cast<T *>(max_val_bytes);
    auto vals = reinterpret_cast<const T *>(vals_bytes);

    // Use single block with 256 threads for simplicity
    // This works well for typical LLM vocab sizes (up to ~100k)
    const int blockSize = 256;
    size_t sharedMemSize = blockSize * sizeof(float) + blockSize * sizeof(int64_t);
    argmaxSimpleKernel<T><<<1, blockSize, sharedMemSize>>>(vals, max_val, max_idx, size);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_BF16:
        return argmax_<bf16_t_cuda>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F16:
        return argmax_<fp16_t_cuda>(max_idx, max_val, vals, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
