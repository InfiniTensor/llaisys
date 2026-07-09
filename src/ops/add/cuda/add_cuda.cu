#include "add_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::ops::cuda {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}

__global__ void add_kernel_fp16(__half *c, const __half *a, const __half *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void add_kernel_bf16(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
#if __CUDA_ARCH__ >= 800
    c[idx] = __hadd(a[idx], b[idx]);
#else
    c[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
#endif
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel) {
    dim3 block(256);
    dim3 grid((numel + block.x - 1) / block.x);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid, block>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                                    reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel_fp16<<<grid, block>>>(reinterpret_cast<__half *>(c), reinterpret_cast<const __half *>(a),
                                         reinterpret_cast<const __half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16<<<grid, block>>>(reinterpret_cast<__nv_bfloat16 *>(c),
                                         reinterpret_cast<const __nv_bfloat16 *>(a),
                                         reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "add cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
