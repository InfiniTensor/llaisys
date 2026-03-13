#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

namespace llaisys::ops::nvidia {

// Bias add kernel
template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t batch_size, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;
    
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t col = i % out_dim;
        float val = to_float_cuda(out[i]) + to_float_cuda(bias[col]);
        out[i] = from_float_cuda<T>(val);
    }
}

// cuBLAS handle wrapper for lazy initialization
class CublasHandle {
public:
    static cublasHandle_t& get() {
        static CublasHandle instance;
        return instance.handle;
    }
private:
    cublasHandle_t handle;
    CublasHandle() {
        cublasCreate(&handle);
    }
    ~CublasHandle() {
        cublasDestroy(handle);
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

// Helper to get CUDA data type
cudaDataType getCudaDataType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;  // CUDA 11.0+ supports bfloat16
    default:
        return CUDA_R_32F;
    }
}

// cuBLAS GEMM wrapper
template <typename T>
void gemm_cublas(T *out, const T *weight, const T *in,
                 size_t out_dim, size_t batch_size, size_t in_dim,
                 llaisysDataType_t dtype) {
    cublasHandle_t handle = CublasHandle::get();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudaDataType cuda_dtype = getCudaDataType(dtype);
    
    // Linear: Y = X * W^T
    // X: [batch_size, in_dim]
    // W: [out_dim, in_dim] 
    // Y: [batch_size, out_dim]
    //
    // cuBLAS is column-major, so we compute:
    // Y^T = W * X^T
    // C = A * B where:
    //   A = W [in_dim x out_dim] (transposed to [out_dim x in_dim])
    //   B = X^T [in_dim x batch_size]
    //   C = Y^T [out_dim x batch_size]
    
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,           // transA: transpose weight
        CUBLAS_OP_N,           // transB: no transpose input
        static_cast<int>(out_dim),     // m
        static_cast<int>(batch_size),  // n  
        static_cast<int>(in_dim),      // k
        &alpha,
        weight,                // A
        cuda_dtype,            // A type
        static_cast<int>(in_dim),      // lda
        in,                    // B
        cuda_dtype,            // B type
        static_cast<int>(in_dim),      // ldb
        &beta,
        out,                   // C
        cuda_dtype,            // C type
        static_cast<int>(out_dim),     // ldc
        CUBLAS_COMPUTE_32F,    // compute type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // use Tensor Core
    );
}

// Template specializations for different types
template <typename T>
void linear_cublas(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                   const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                   llaisysDataType_t dtype);

template <>
void linear_cublas<float>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                          const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                          llaisysDataType_t dtype) {
    auto out = reinterpret_cast<float *>(out_bytes);
    auto in = reinterpret_cast<const float *>(in_bytes);
    auto weight = reinterpret_cast<const float *>(weight_bytes);
    
    gemm_cublas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
    // Add bias
    if (bias_bytes) {
        auto bias = reinterpret_cast<const float *>(bias_bytes);
        size_t total = batch_size * out_dim;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;
        addBiasKernel<float><<<numBlocks, blockSize>>>(out, bias, batch_size, out_dim);
    }
}

template <>
void linear_cublas<fp16_t_cuda>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                                const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                                llaisysDataType_t dtype) {
    auto out = reinterpret_cast<fp16_t_cuda *>(out_bytes);
    auto in = reinterpret_cast<const fp16_t_cuda *>(in_bytes);
    auto weight = reinterpret_cast<const fp16_t_cuda *>(weight_bytes);
    
    gemm_cublas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
    if (bias_bytes) {
        auto bias = reinterpret_cast<const fp16_t_cuda *>(bias_bytes);
        size_t total = batch_size * out_dim;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;
        addBiasKernel<fp16_t_cuda><<<numBlocks, blockSize>>>(out, bias, batch_size, out_dim);
    }
}

template <>
void linear_cublas<bf16_t_cuda>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                                const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                                llaisysDataType_t dtype) {
    auto out = reinterpret_cast<bf16_t_cuda *>(out_bytes);
    auto in = reinterpret_cast<const bf16_t_cuda *>(in_bytes);
    auto weight = reinterpret_cast<const bf16_t_cuda *>(weight_bytes);
    
    gemm_cublas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
    if (bias_bytes) {
        auto bias = reinterpret_cast<const bf16_t_cuda *>(bias_bytes);
        size_t total = batch_size * out_dim;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;
        addBiasKernel<bf16_t_cuda><<<numBlocks, blockSize>>>(out, bias, batch_size, out_dim);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_cublas<float>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
    case LLAISYS_DTYPE_BF16:
        return linear_cublas<bf16_t_cuda>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
    case LLAISYS_DTYPE_F16:
        return linear_cublas<fp16_t_cuda>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
