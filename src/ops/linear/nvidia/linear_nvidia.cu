#include "linear_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <cublas_v2.h>

namespace llaisys::ops::nvidia {

inline void check_cublas(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[CUBLAS ERROR] " << msg << ": " << static_cast<int>(status) << std::endl;
        throw std::runtime_error("cuBLAS API failed");
    }
}

static thread_local cublasHandle_t TL_HANDLE = nullptr;

cublasHandle_t get_cublas_handle() {
    if (TL_HANDLE == nullptr) {
        check_cublas(cublasCreate(&TL_HANDLE), "cublasCreate");
    }
    return TL_HANDLE;
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = M * N;
    if (idx >= total) {
        return;
    }
    size_t col = idx % N;
    out[idx] = from_float<T>(to_float(out[idx]) + to_float(bias[col]));
}

void launch_bias(
    std::byte *out,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t N,
    cudaStream_t stream) {
    const size_t total = M * N;
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(total, threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(bias),
            M,
            N);
        break;
    case LLAISYS_DTYPE_F16:
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            M,
            N);
        break;
    case LLAISYS_DTYPE_BF16:
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            M,
            N);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    check_cuda(cudaGetLastError(), "linear add_bias kernel launch");
}

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t N,
    size_t K) {
    // row-major: out[M, N] = in[M, K] * weight[N, K]^T
    // map to column-major GEMM:
    // out_col[N, M] = op(weight_col[K, N]) * in_col[K, M]
    // where op = transpose, so result is equivalent to row-major formula.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaDataType_t data_type;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        data_type = CUDA_R_32F;
        break;
    case LLAISYS_DTYPE_F16:
        data_type = CUDA_R_16F;
        break;
    case LLAISYS_DTYPE_BF16:
        data_type = CUDA_R_16BF;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    auto stream = current_stream();
    cublasHandle_t handle = get_cublas_handle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight,
            data_type,
            static_cast<int>(K),
            in,
            data_type,
            static_cast<int>(K),
            &beta,
            out,
            data_type,
            static_cast<int>(N),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx");

    if (bias != nullptr) {
        launch_bias(out, bias, type, M, N, stream);
    }
}

} // namespace llaisys::ops::nvidia
