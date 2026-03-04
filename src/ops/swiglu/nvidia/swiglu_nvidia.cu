#include "swiglu_nvidia.hpp"

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
__global__ void swiglu_kernel(
    T *out, const T *gate, const T *up,
    size_t rows, size_t cols) {
    
    size_t total = rows * cols;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t i = tid / cols;
    size_t j = tid % cols;

    // Contiguous access: row * cols + col
    float val_gate = to_float(gate[i * cols + j]);
    float val_up   = to_float(up[i * cols + j]);

    // SiLU(x) = x / (1 + exp(-x))
    float silu_gate = val_gate / (1.0f + expf(-val_gate));
    
    out[i * cols + j] = from_float<T>(val_up * silu_gate);
}

template<typename T>
void launch_swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                   size_t rows, size_t cols) {
    T* d_out = reinterpret_cast<T*>(out);
    const T* d_gate = reinterpret_cast<const T*>(gate);
    const T* d_up = reinterpret_cast<const T*>(up);

    size_t total = rows * cols;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    swiglu_kernel<T><<<blocks, threads>>>(d_out, d_gate, d_up, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA SwiGLU Kernel Launch failed: %s\n", cudaGetErrorString(err));
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t rows, size_t cols) {
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_swiglu<float>(out, gate, up, rows, cols);
            break;
        case LLAISYS_DTYPE_F16:
            launch_swiglu<half>(out, gate, up, rows, cols);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_swiglu<nv_bfloat16>(out, gate, up, rows, cols);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA SwiGLU: %d\n", dtype);
            break;
    }
}

} // namespace llaisys::ops::nvidia