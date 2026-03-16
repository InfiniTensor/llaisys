#include "linear_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <cublas_v2.h>
#include <cstdio>

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS] cublasCreate failed: %d\n", (int)st);
        }
    }
    return handle;
}

__global__ void add_bias_kernel(void *out, const void *bias,
                                llaisysDataType_t dtype, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t n = idx % N;
    float val = load_as_f32(out, idx, dtype);
    float b = load_as_f32(bias, n, dtype);
    store_from_f32(out, idx, val + b, dtype);
}

__global__ void convert_to_f32_kernel(float *out, const void *in,
                                      llaisysDataType_t dtype, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = load_as_f32(in, idx, dtype);
}

__global__ void convert_from_f32_kernel(void *out, const float *in,
                                        llaisysDataType_t dtype, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    store_from_f32(out, idx, in[idx], dtype);
}

static cudaDataType_t to_cuda_dtype(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BF16: return CUDA_R_16BF;
    case LLAISYS_DTYPE_F16:  return CUDA_R_16F;
    default:                 return CUDA_R_32F;
    }
}

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t N, size_t K, bool has_bias) {
    // out[M,N] = in[M,K] * weight[N,K]^T
    // cuBLAS column-major: C(N,M) = A^T(N,K) * B(K,M)
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;

    if (dtype == LLAISYS_DTYPE_F16) {
        // FP16: cublasGemmEx natively supported on all recent GPUs
        cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     (int)N, (int)M, (int)K,
                     &alpha,
                     weight, CUDA_R_16F, (int)K,
                     in,     CUDA_R_16F, (int)K,
                     &beta,
                     out,    CUDA_R_16F, (int)N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS] GemmEx FP16 failed: %d\n", (int)st);
        }
    } else if (dtype == LLAISYS_DTYPE_F32) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)N, (int)M, (int)K,
                    &alpha,
                    reinterpret_cast<const float*>(weight), (int)K,
                    reinterpret_cast<const float*>(in),     (int)K,
                    &beta,
                    reinterpret_cast<float*>(out),          (int)N);
    } else {
        // BF16: use cublasGemmEx with native BF16 support (SM 80+, Ampere tensor cores)
        cudaDataType_t cuda_dt = to_cuda_dtype(dtype);
        cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     (int)N, (int)M, (int)K,
                     &alpha,
                     weight, cuda_dt, (int)K,
                     in,     cuda_dt, (int)K,
                     &beta,
                     out,    cuda_dt, (int)N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS] GemmEx BF16 failed: %d\n", (int)st);
        }
    }

    if (has_bias && bias) {
        add_bias_kernel<<<cuda_grid_size(M * N), CUDA_BLOCK_SIZE>>>(out, bias, dtype, M, N);
        CUDA_KERNEL_CHECK();
    }
}
} // namespace llaisys::ops::cuda
