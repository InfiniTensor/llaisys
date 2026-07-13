#include "swiglu_cu.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace swigluops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// out[i] = up[i] * SiLU(gate[i])  where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
template <typename T>
__global__ void
swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float g = to_float(gate[idx]);
    float u = to_float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    out[idx] = from_float<T>(u * silu);
}

} // namespace swigluops::nvidia

namespace llaisys::ops::nvidia {

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            size_t numel,
            llaisysDataType_t dtype) {
    constexpr int threads = 256;
    unsigned int blocks = static_cast<unsigned int>((numel + threads - 1) / threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swigluops::nvidia::swiglu_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        swigluops::nvidia::swiglu_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swigluops::nvidia::swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
