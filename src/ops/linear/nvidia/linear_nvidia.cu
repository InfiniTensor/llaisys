#include "linear_nvidia.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t batch_size, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = batch_size * out_features;
    if (idx >= numel) {
        return;
    }

    size_t col = idx % out_features;
    out[idx] = utils::cast_device<T>(utils::cast_device<float>(out[idx]) + utils::cast_device<float>(bias[col]));
}

static cudaDataType_t toCudaDataType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for cuBLAS linear: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }
}

static void addBias(std::byte *out, const std::byte *bias, llaisysDataType_t type,
                    size_t batch_size, size_t out_features) {
    const int block_size = 256;
    const size_t numel = batch_size * out_features;
    const int num_blocks = static_cast<int>((numel + block_size - 1) / block_size);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_bias_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(bias),
            batch_size, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        add_bias_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(bias),
            batch_size, out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        add_bias_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(bias),
            batch_size, out_features);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported bias type for cuBLAS linear: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] linear bias kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    int device_id = 0;
    cudaError_t cuda_status = cudaGetDevice(&device_id);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] cudaGetDevice failed: %s\n", cudaGetErrorString(cuda_status));
        throw std::runtime_error(cudaGetErrorString(cuda_status));
    }

    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle(device_id);
    cublasStatus_t stream_status = cublasSetStream(handle, static_cast<cudaStream_t>(nullptr));
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS ERROR] cublasSetStream failed: status=%d\n", stream_status);
        throw std::runtime_error("cublasSetStream failed");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaDataType_t cuda_type = toCudaDataType(type);

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(out_features),
        static_cast<int>(batch_size),
        static_cast<int>(in_features),
        &alpha,
        weight,
        cuda_type,
        static_cast<int>(in_features),
        in,
        cuda_type,
        static_cast<int>(in_features),
        &beta,
        out,
        cuda_type,
        static_cast<int>(out_features),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS ERROR] linear gemm failed: status=%d\n", status);
        throw std::runtime_error("cuBLAS linear gemm failed");
    }

    if (bias != nullptr) {
        addBias(out, bias, type, batch_size, out_features);
    }
}

} // namespace llaisys::ops::nvidia
