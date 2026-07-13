#include "linear_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <algorithm>
#include <cublas_v2.h>

namespace llaisys::ops::nvidia {
namespace {

#define CUBLAS_CHECK(EXPR__)                                                                  \
    do {                                                                                      \
        cublasStatus_t status__ = (EXPR__);                                                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                              \
            throw std::runtime_error("cuBLAS call failed");                                   \
        }                                                                                     \
    } while (0)

inline cudaDataType_t toCudaDataType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t numel, size_t cols) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        const size_t col = idx % cols;
        const float value = llaisys::device::nvidia::scalarToFloat<T>(out[idx])
                          + llaisys::device::nvidia::scalarToFloat<T>(bias[col]);
        out[idx] = llaisys::device::nvidia::floatToScalar<T>(value);
    }
}

template <typename T>
void launchBias(std::byte *out, const std::byte *bias, size_t numel, size_t cols, cudaStream_t stream) {
    constexpr int threads = 256;
    const int max_blocks = 4096;
    const int blocks = static_cast<int>(std::min<size_t>((numel + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    addBiasKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type,
            size_t m, size_t n, size_t k, llaisysStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    auto &runtime = llaisys::core::context().runtime();
    cublasHandle_t handle = llaisys::device::nvidia::resource(runtime.deviceId()).cublasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, llaisys::device::nvidia::toCudaStream(stream)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t data_type = toCudaDataType(type);

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        weight,
        data_type,
        static_cast<int>(k),
        in,
        data_type,
        static_cast<int>(k),
        &beta,
        out,
        data_type,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    if (bias != nullptr) {
        const size_t numel = m * n;
        const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);
        switch (type) {
        case LLAISYS_DTYPE_F32:
            launchBias<float>(out, bias, numel, n, cuda_stream);
            break;
        case LLAISYS_DTYPE_F16:
            launchBias<llaisys::fp16_t>(out, bias, numel, n, cuda_stream);
            break;
        case LLAISYS_DTYPE_BF16:
            launchBias<llaisys::bf16_t>(out, bias, numel, n, cuda_stream);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}

} // namespace llaisys::ops::nvidia
