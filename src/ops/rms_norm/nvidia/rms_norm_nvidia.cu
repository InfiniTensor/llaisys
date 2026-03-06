#include "rms_norm_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
__global__ void rms_norm_kernel_f32(float* c, const float* a, const float* w, int rows, int dim, float eps) {
    int row = blockIdx.x; // 当前处理的 Token 索引
    int tid = threadIdx.x; // 当前线程的索引
    if (row >= rows) return;

    const float* x_row = a + row * dim;
    float* y_row = c + row * dim;

    // 1. 每个线程计算自己负责元素的平方和
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }

    // 2. 使用共享内存进行块内规约 (Reduction) 求总和
    __shared__ float shared_sum[256]; 
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // 3. 由 0 号线程计算均方根的倒数 (rsqrtf 是 CUDA 原生硬件指令，极快)
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / dim + eps);
    }
    __syncthreads();

    // 4. 将归一化结果乘上权重
    for (int i = tid; i < dim; i += blockDim.x) {
        y_row[i] = x_row[i] * inv_rms * w[i];
    }
}

// --- F16 Kernel ---
__global__ void rms_norm_kernel_f16(void* c, const void* a, const void* w, int rows, int dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const __half* x_row = reinterpret_cast<const __half*>(a) + row * dim;
    const __half* w_row = reinterpret_cast<const __half*>(w);
    __half* y_row = reinterpret_cast<__half*>(c) + row * dim;

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        local_sum += val * val;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float weight = __half2float(w_row[i]);
        y_row[i] = __float2half(val * inv_rms * weight);
    }
}

// --- BF16 Kernel ---
__global__ void rms_norm_kernel_bf16(void* c, const void* a, const void* w, int rows, int dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

#if __CUDACC_VER_MAJOR__ >= 11
    const __nv_bfloat16* x_row = reinterpret_cast<const __nv_bfloat16*>(a) + row * dim;
    const __nv_bfloat16* w_row = reinterpret_cast<const __nv_bfloat16*>(w);
    __nv_bfloat16* y_row = reinterpret_cast<__nv_bfloat16*>(c) + row * dim;

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        local_sum += val * val;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        float weight = __bfloat162float(w_row[i]);
        y_row[i] = __float2bfloat16(val * inv_rms * weight);
    }
#endif
}

// C++ 路由入口
void rms_norm(std::byte *c, const std::byte *a, const std::byte *b, size_t rows, size_t dim, float eps, llaisysDataType_t type) {
    int threads_per_block = 256; 
    int blocks_per_grid = rows; // 每一个 Token 分配一个独立的 Block

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel_f32<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float*>(c),
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(b),
            rows, dim, eps
        );
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel_f16<<<blocks_per_grid, threads_per_block>>>(c, a, b, rows, dim, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(c, a, b, rows, dim, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia