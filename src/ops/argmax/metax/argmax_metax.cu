#include "argmax_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace llaisys::ops::metax {

// Kernel for f16 - output to float buffer (workaround for MetaX scalar write bug)
__global__ void argmaxF16Kernel(const fp16_t_metax *vals, float *max_val_float, int64_t *max_idx, size_t size) {
    __shared__ float shared_vals[256];
    __shared__ int64_t shared_idxs[256];

    unsigned int tid = threadIdx.x;

    shared_vals[tid] = -FLT_MAX;
    shared_idxs[tid] = -1;

    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float v = to_float_metax(vals[i]);
        if (v > shared_vals[tid]) {
            shared_vals[tid] = v;
            shared_idxs[tid] = i;
        }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idxs[tid] = shared_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = shared_idxs[0];
        *max_val_float = shared_vals[0];
    }
}

// Kernel for bf16
__global__ void argmaxBF16Kernel(const bf16_t_metax *vals, bf16_t_metax *max_val, int64_t *max_idx, size_t size) {
    __shared__ float shared_vals[256];
    __shared__ int64_t shared_idxs[256];

    unsigned int tid = threadIdx.x;

    shared_vals[tid] = -FLT_MAX;
    shared_idxs[tid] = -1;

    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float v = to_float_metax(vals[i]);
        if (v > shared_vals[tid]) {
            shared_vals[tid] = v;
            shared_idxs[tid] = i;
        }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idxs[tid] = shared_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = shared_idxs[0];
        max_val[0] = from_float_metax<bf16_t_metax>(shared_vals[0]);
    }
}

// Kernel for f32
__global__ void argmaxF32Kernel(const float *vals, float *max_val, int64_t *max_idx, size_t size) {
    __shared__ float shared_vals[256];
    __shared__ int64_t shared_idxs[256];

    unsigned int tid = threadIdx.x;

    shared_vals[tid] = -FLT_MAX;
    shared_idxs[tid] = -1;

    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float v = vals[i];
        if (v > shared_vals[tid]) {
            shared_vals[tid] = v;
            shared_idxs[tid] = i;
        }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idxs[tid] = shared_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_val[0] = shared_vals[0];
        *max_idx = shared_idxs[0];
    }
}

void argmax(std::byte *max_idx_bytes, std::byte *max_val_bytes, const std::byte *vals_bytes, llaisysDataType_t type, size_t size) {
    auto max_idx = reinterpret_cast<int64_t *>(max_idx_bytes);
    auto max_val = reinterpret_cast<std::byte *>(max_val_bytes);
    auto vals = reinterpret_cast<const std::byte *>(vals_bytes);

    const int blockSize = 256;
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmaxF32Kernel<<<1, blockSize>>>(
            reinterpret_cast<const float*>(vals),
            reinterpret_cast<float*>(max_val),
            max_idx, size);
        break;
    case LLAISYS_DTYPE_F16: {
        // Workaround: Output to temp float buffer, then convert on host
        float *d_max_val_float;
        cudaMalloc(&d_max_val_float, sizeof(float));
        
        argmaxF16Kernel<<<1, blockSize>>>(
            reinterpret_cast<const fp16_t_metax*>(vals),
            d_max_val_float,
            max_idx, size);
        cudaDeviceSynchronize();
        
        float h_max_val;
        cudaMemcpy(&h_max_val, d_max_val_float, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_max_val_float);
        
        // Host-side float to f16 conversion
        union { float f; uint32_t u; } fu;
        fu.f = h_max_val;
        uint32_t f32_bits = fu.u;
        
        uint32_t sign = (f32_bits >> 31) & 0x1;
        int32_t exp = ((f32_bits >> 23) & 0xFF) - 127;
        uint32_t mant = f32_bits & 0x7FFFFF;
        
        uint16_t f16_bits;
        if (exp == 128) {
            f16_bits = (sign << 15) | (mant ? 0x7E00 : 0x7C00);
        } else if (exp <= -15) {
            f16_bits = sign << 15;
        } else if (exp >= 16) {
            f16_bits = (sign << 15) | 0x7C00;
        } else {
            int32_t new_exp = exp + 15;
            uint32_t new_mant = (mant + 0xFFF) >> 13;
            if (new_mant >= 0x400) {
                new_mant = 0;
                if (++new_exp >= 31) {
                    f16_bits = (sign << 15) | 0x7C00;
                } else {
                    f16_bits = (sign << 15) | (new_exp << 10);
                }
            } else {
                f16_bits = (sign << 15) | (new_exp << 10) | (new_mant & 0x3FF);
            }
        }
        
        cudaMemcpy(max_val, &f16_bits, sizeof(uint16_t), cudaMemcpyHostToDevice);
        break;
    }
    case LLAISYS_DTYPE_BF16:
        argmaxBF16Kernel<<<1, blockSize>>>(
            reinterpret_cast<const bf16_t_metax*>(vals),
            reinterpret_cast<bf16_t_metax*>(max_val),
            max_idx, size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
    cudaDeviceSynchronize();
}

}
