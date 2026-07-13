#include "argmax_cu.cuh"

#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace argmaxops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// Single-block parallel reduction argmax.
// Each thread strides through the input, keeping a local max, then the block
// reduces via shared memory.  One block is sufficient for typical vocab sizes.
template <typename T>
__global__ void
argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    extern __shared__ char smem[];
    float   *smem_val = reinterpret_cast<float *>(smem);
    int64_t *smem_idx = reinterpret_cast<int64_t *>(smem_val + blockDim.x);

    unsigned tid = threadIdx.x;
    float    local_max = -FLT_MAX;
    int64_t  local_idx = 0;

    for (size_t i = tid; i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = static_cast<int64_t>(i);
        }
    }

    smem_val[tid] = local_max;
    smem_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem_val[tid + s] > smem_val[tid]) {
                smem_val[tid] = smem_val[tid + s];
                smem_idx[tid] = smem_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = static_cast<T>(smem_val[0]);
        *max_idx = smem_idx[0];
    }
}

} // namespace argmaxops::nvidia

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype) {
    constexpr int threads = 1024;
    size_t smem_size = threads * (sizeof(float) + sizeof(int64_t));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmaxops::nvidia::argmax_kernel<float><<<1, threads, smem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmaxops::nvidia::argmax_kernel<__half><<<1, threads, smem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val),
            reinterpret_cast<const __half *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmaxops::nvidia::argmax_kernel<__nv_bfloat16><<<1, threads, smem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia