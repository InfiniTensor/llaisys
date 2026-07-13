#include "linear_cu.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace linearops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// output[N,K] = input[N,M] @ weight[K,M]^T + bias[K]
// blockIdx.y = n (batch row), blockIdx.x * blockDim.x + threadIdx.x = k (output col)
template <typename T>
__global__ void
linear_kernel(T *output, const T *input, const T *weight, const T *bias,
              size_t N, size_t M, size_t K) {
    size_t n = blockIdx.y;
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N || k >= K) return;

    float sum = 0.0f;
    for (size_t m = 0; m < M; ++m)
        sum += to_float(input[n * M + m]) * to_float(weight[k * M + m]);
    if (bias != nullptr)
        sum += to_float(bias[k]);
    output[n * K + k] = from_float<T>(sum);
}

} // namespace linearops::nvidia

namespace llaisys::ops::nvidia {

void linear(std::byte *output,
            const std::byte *input,
            const std::byte *weight,
            const std::byte *bias,
            size_t N,
            size_t M,
            size_t K,
            llaisysDataType_t dtype) {
    constexpr int block_k = 128;
    dim3 block(block_k);
    dim3 grid(static_cast<unsigned int>((K + block_k - 1) / block_k),
              static_cast<unsigned int>(N));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linearops::nvidia::linear_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            N, M, K);
        break;
    case LLAISYS_DTYPE_F16:
        linearops::nvidia::linear_kernel<__half><<<grid, block>>>(
            reinterpret_cast<__half *>(output),
            reinterpret_cast<const __half *>(input),
            reinterpret_cast<const __half *>(weight),
            reinterpret_cast<const __half *>(bias),
            N, M, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linearops::nvidia::linear_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16 *>(output),
            reinterpret_cast<const __nv_bfloat16 *>(input),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias),
            N, M, K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
