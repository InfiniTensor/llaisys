#include "rms_norm_cu.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace rmsnormops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// One block per row. Shared memory holds one float per thread for reduction.
// output[i,j] = input[i,j] * weight[j] * rsqrt(mean(input[i,:]^2) + eps)
template <typename T>
__global__ void
rms_norm_kernel(T *output, const T *input, const T *weight,
                size_t N, size_t M, float eps) {
    extern __shared__ float smem[]; // blockDim.x floats

    size_t row = blockIdx.x;
    if (row >= N) return;

    unsigned tid = threadIdx.x;
    unsigned nthreads = blockDim.x;

    // Phase 1: compute partial sum of squares
    float local_sq = 0.0f;
    for (size_t j = tid; j < M; j += nthreads) {
        float v = to_float(input[row * M + j]);
        local_sq += v * v;
    }
    smem[tid] = local_sq;
    __syncthreads();

    // Binary-tree reduction
    for (unsigned s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rms_scale = rsqrtf(smem[0] / static_cast<float>(M) + eps);

    // Phase 2: normalize and scale
    for (size_t j = tid; j < M; j += nthreads) {
        float x = to_float(input[row * M + j]);
        float w = to_float(weight[j]);
        output[row * M + j] = from_float<T>(x * w * rms_scale);
    }
}

} // namespace rmsnormops::nvidia

namespace llaisys::ops::nvidia {

void rms_norm(std::byte *output,
              const std::byte *input,
              const std::byte *weight,
              size_t N,
              size_t M,
              float eps,
              llaisysDataType_t dtype) {
    constexpr int threads = 256; // must be power of 2
    size_t smem_size = threads * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rmsnormops::nvidia::rms_norm_kernel<float>
            <<<static_cast<unsigned int>(N), threads, smem_size>>>(
                reinterpret_cast<float *>(output),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(weight),
                N, M, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rmsnormops::nvidia::rms_norm_kernel<__half>
            <<<static_cast<unsigned int>(N), threads, smem_size>>>(
                reinterpret_cast<__half *>(output),
                reinterpret_cast<const __half *>(input),
                reinterpret_cast<const __half *>(weight),
                N, M, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rmsnormops::nvidia::rms_norm_kernel<__nv_bfloat16>
            <<<static_cast<unsigned int>(N), threads, smem_size>>>(
                reinterpret_cast<__nv_bfloat16 *>(output),
                reinterpret_cast<const __nv_bfloat16 *>(input),
                reinterpret_cast<const __nv_bfloat16 *>(weight),
                N, M, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
