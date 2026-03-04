#include "rms_norm_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

namespace llaisys::ops::nvidia {

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<half>(half v) { return __half2float(v); }
template<> __device__ inline float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline half from_float<half>(float v) { return __float2half(v); }
template<> __device__ inline nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight, 
                                size_t rows, size_t cols, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    // Step 1: sum of squares
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(in[row * cols + j]);
        local_sum += val * val;
    }
    
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);

    // Step 2: normalize and scale
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float x_val = to_float(in[row * cols + j]);
        float w_val = to_float(weight[j]);
        out[row * cols + j] = from_float<T>(x_val * w_val * inv_rms);
    }
}

template<typename T>
void launch_rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
                     size_t rows, size_t cols, float eps) {
    T* d_out = reinterpret_cast<T*>(out);
    const T* d_in = reinterpret_cast<const T*>(in);
    const T* d_weight = reinterpret_cast<const T*>(weight);

    int threads = 256;
    if (cols < 256) {
        threads = 1;
        while (threads < cols) threads <<= 1;
    }
    
    size_t smem_size = threads * sizeof(float);

    rms_norm_kernel<T><<<rows, threads, smem_size>>>(d_out, d_in, d_weight, rows, cols, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA RMSNorm Kernel Launch failed: %s\n", cudaGetErrorString(err));
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_rms_norm<float>(out, in, weight, rows, cols, eps);
            break;
        case LLAISYS_DTYPE_F16:
            launch_rms_norm<half>(out, in, weight, rows, cols, eps);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_rms_norm<nv_bfloat16>(out, in, weight, rows, cols, eps);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA RMSNorm: %d\n", dtype);
            break;
    }
}

} // namespace llaisys::ops::nvidia