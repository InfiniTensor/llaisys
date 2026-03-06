#include "argmax_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat> // 使用标准的 FLT_MAX

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
__global__ void argmax_kernel_f32(int64_t* max_idx, float* max_val, const float* vals, size_t numel) {
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int64_t local_idx = -1;

    // 1. 每个线程在自己负责的跨度内找局部最大值
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = vals[i];
        if (val > local_max || local_idx == -1) {
            local_max = val;
            local_idx = i;
        }
    }

    __shared__ float shared_max[256];
    __shared__ int64_t shared_idx[256];

    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();

    // 2. 块内规约，找出全局最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_idx[tid + stride] != -1 && 
               (shared_idx[tid] == -1 || shared_max[tid + stride] > shared_max[tid])) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // 3. 0 号线程负责将最终结果写回全局内存
    if (tid == 0) {
        int64_t best_idx = shared_idx[0];
        if (best_idx != -1) {
            *max_idx = best_idx;
            *max_val = vals[best_idx]; // 直接从原数组取，保证精度无损
        }
    }
}

// --- F16 Kernel ---
__global__ void argmax_kernel_f16(int64_t* max_idx, void* max_val_ptr, const void* vals_ptr, size_t numel) {
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int64_t local_idx = -1;
    const __half* vals = reinterpret_cast<const __half*>(vals_ptr);

    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = __half2float(vals[i]);
        if (val > local_max || local_idx == -1) {
            local_max = val;
            local_idx = i;
        }
    }

    __shared__ float shared_max[256];
    __shared__ int64_t shared_idx[256];

    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_idx[tid + stride] != -1 && 
               (shared_idx[tid] == -1 || shared_max[tid + stride] > shared_max[tid])) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best_idx = shared_idx[0];
        if (best_idx != -1) {
            *max_idx = best_idx;
            reinterpret_cast<__half*>(max_val_ptr)[0] = vals[best_idx];
        }
    }
}

// --- BF16 Kernel ---
__global__ void argmax_kernel_bf16(int64_t* max_idx, void* max_val_ptr, const void* vals_ptr, size_t numel) {
#if __CUDACC_VER_MAJOR__ >= 11
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int64_t local_idx = -1;
    const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(vals_ptr);

    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = __bfloat162float(vals[i]);
        if (val > local_max || local_idx == -1) {
            local_max = val;
            local_idx = i;
        }
    }

    __shared__ float shared_max[256];
    __shared__ int64_t shared_idx[256];

    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_idx[tid + stride] != -1 && 
               (shared_idx[tid] == -1 || shared_max[tid + stride] > shared_max[tid])) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best_idx = shared_idx[0];
        if (best_idx != -1) {
            *max_idx = best_idx;
            reinterpret_cast<__nv_bfloat16*>(max_val_ptr)[0] = vals[best_idx];
        }
    }
#endif
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    if (numel == 0) return;

    // 因为是全局规约，只需要开 1 个 Block 即可处理千万级别的数据
    int threads_per_block = 256;
    int blocks_per_grid = 1; 
    
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel_f32<<<blocks_per_grid, threads_per_block>>>(idx_ptr, reinterpret_cast<float*>(max_val), reinterpret_cast<const float*>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel_f16<<<blocks_per_grid, threads_per_block>>>(idx_ptr, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(idx_ptr, max_val, vals, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia