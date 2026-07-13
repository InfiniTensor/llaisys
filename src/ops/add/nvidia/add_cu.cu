#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"
#include "llaisys.h"

namespace addops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template <>
__global__ void add_kernel<__half>(__half *c, const __half *a, const __half *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

template <>
__global__ void add_kernel<__nv_bfloat16>(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

} // namespace addops::nvidia

namespace llaisys::ops::nvidia {

void add(std::byte *c,
         const std::byte *a,
         const std::byte *b,
         llaisysDataType_t type,
         size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        addops::nvidia::add_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            n);
        break;
    case LLAISYS_DTYPE_F16:
        // fp16_t has the same 16-bit IEEE 754 layout as __half
        addops::nvidia::add_kernel<__half><<<blocksPerGrid, threadsPerBlock>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            n);
        break;
    case LLAISYS_DTYPE_BF16:
        // bf16_t has the same 16-bit bfloat layout as __nv_bfloat16
        addops::nvidia::add_kernel<__nv_bfloat16><<<blocksPerGrid, threadsPerBlock>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            n);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia