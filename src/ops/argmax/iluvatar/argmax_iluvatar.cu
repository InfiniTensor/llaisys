#include "argmax_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace llaisys::ops::iluvatar {

template <typename T>
__global__ void argmaxKernel(const T *vals, float *max_val, int64_t *max_idx, size_t size) {
    // Simple implementation: use shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ int64_t shared_idx[256];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize
    float local_max = -FLT_MAX;
    int64_t local_idx = 0;
    
    // Each thread finds max in its portion
    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = to_float_cuda(vals[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        max_val[blockIdx.x] = shared_max[0];
        max_idx[blockIdx.x] = shared_idx[0];
    }
}

// Final reduction kernel
__global__ void argmaxFinalKernel(float *max_vals, int64_t *max_idxs, int num_blocks) {
    float global_max = -FLT_MAX;
    int64_t global_idx = 0;
    
    for (int i = 0; i < num_blocks; ++i) {
        if (max_vals[i] > global_max) {
            global_max = max_vals[i];
            global_idx = max_idxs[i];
        }
    }
    
    max_vals[0] = global_max;
    max_idxs[0] = global_idx;
}

template <typename T>
void argmax_(std::byte *max_idx_bytes, std::byte *max_val_bytes, const std::byte *vals_bytes, size_t size) {
    auto vals = reinterpret_cast<const T *>(vals_bytes);
    auto max_idx = reinterpret_cast<int64_t *>(max_idx_bytes);
    auto max_val = reinterpret_cast<float *>(max_val_bytes);
    
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Temporary storage for block-level results
    float *d_block_max;
    int64_t *d_block_idx;
    cudaMalloc(&d_block_max, numBlocks * sizeof(float));
    cudaMalloc(&d_block_idx, numBlocks * sizeof(int64_t));
    
    // First pass: find max per block
    argmaxKernel<T><<<numBlocks, blockSize>>>(vals, d_block_max, d_block_idx, size);
    
    // Second pass: find global max
    if (numBlocks > 1) {
        argmaxFinalKernel<<<1, 1>>>(d_block_max, d_block_idx, numBlocks);
    }
    
    // Copy results to output
    cudaMemcpy(max_val, d_block_max, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(max_idx, d_block_idx, sizeof(int64_t), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_block_max);
    cudaFree(d_block_idx);
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
