#include "argmax_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <cfloat>

// Parallel reduction for argmax
__global__ void argmax_kernel(int64_t *max_idx_out, void *max_val_out,
                              const void *vals, llaisysDataType_t dtype, size_t numel) {
    extern __shared__ char shared_mem[];
    float *svals = reinterpret_cast<float *>(shared_mem);
    int *sidxs = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    svals[tid] = -FLT_MAX;
    sidxs[tid] = 0;

    if (idx < numel) {
        svals[tid] = load_as_f32(vals, idx, dtype);
        sidxs[tid] = idx;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && svals[tid + s] > svals[tid]) {
            svals[tid] = svals[tid + s];
            sidxs[tid] = sidxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic compare: use atomicCAS on a global flag
        // For single-block case, just write directly
        // For multi-block, we need a second pass. Simplify: use single block for vocab-sized vectors.
        max_idx_out[blockIdx.x] = sidxs[0];
        store_from_f32(max_val_out, blockIdx.x, svals[0], dtype);
    }
}

// Second pass: reduce across blocks
__global__ void argmax_reduce_kernel(int64_t *final_idx, void *final_val,
                                     const int64_t *block_idx, const void *block_val,
                                     llaisysDataType_t dtype, int nblocks) {
    float best = -FLT_MAX;
    int64_t best_idx = 0;
    for (int i = 0; i < nblocks; i++) {
        float v = load_as_f32(block_val, i, dtype);
        if (v > best) {
            best = v;
            best_idx = block_idx[i];
        }
    }
    *final_idx = best_idx;
    store_from_f32(final_val, 0, best, dtype);
}

namespace llaisys::ops::cuda {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    int block_size = 1024;
    int nblocks = cuda_grid_size(numel, block_size);
    size_t shared_size = block_size * (sizeof(float) + sizeof(int));

    if (nblocks == 1) {
        argmax_kernel<<<1, block_size, shared_size>>>(
            reinterpret_cast<int64_t *>(max_idx), max_val, vals, type, numel);
        CUDA_KERNEL_CHECK();
    } else {
        int64_t *block_idx;
        std::byte *block_val;
        size_t val_size = cuda_dsize(type);
        cudaMalloc(&block_idx, nblocks * sizeof(int64_t));
        cudaMalloc(&block_val, nblocks * val_size);

        argmax_kernel<<<nblocks, block_size, shared_size>>>(
            block_idx, block_val, vals, type, numel);
        CUDA_KERNEL_CHECK();

        argmax_reduce_kernel<<<1, 1>>>(
            reinterpret_cast<int64_t *>(max_idx), max_val,
            block_idx, block_val, type, nblocks);
        CUDA_KERNEL_CHECK();

        cudaFree(block_idx);
        cudaFree(block_val);
    }
}
} // namespace llaisys::ops::cuda
