#include "linear_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../device/metax/metax_resource.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

namespace llaisys::ops::metax {

// Bias add kernel for mcblas version
template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t batch_size, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;
    
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t col = i % out_dim;
        float val = to_float_metax(out[i]) + to_float_metax(bias[col]);
        out[i] = from_float_metax<T>(val);
    }
}

// Helper to get mcblas data type (mapped via cuBLAS wrapper)
macaDataType getMacaDataType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return MACA_R_32F;
    case LLAISYS_DTYPE_F16:
        return MACA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return MACA_R_16BF;
    default:
        return MACA_R_32F;
    }
}

// mcBLAS GEMM wrapper
template <typename T>
void gemm_mcblas(T *out, const T *weight, const T *in,
                 size_t out_dim, size_t batch_size, size_t in_dim,
                 llaisysDataType_t dtype) {
    // Get singleton mcblas handle
    mcblasHandle_t handle = llaisys::device::metax::McblasHandle::get();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    macaDataType maca_dtype = getMacaDataType(dtype);
    
    // Linear: Y = X * W^T
    // X: [batch_size, in_dim]
    // W: [out_dim, in_dim] 
    // Y: [batch_size, out_dim]
    //
    // mcBLAS is column-major, so we compute:
    // Y^T = W * X^T
    // C = A * B where:
    //   A = W [in_dim x out_dim] (transposed to [out_dim x in_dim])
    //   B = X^T [in_dim x batch_size]
    //   C = Y^T [out_dim x batch_size]
    
    mcblasGemmEx(
        handle,
        MCBLAS_OP_T,           // transa: transpose weight
        MCBLAS_OP_N,           // transb: no transpose input
        static_cast<int>(out_dim),     // m
        static_cast<int>(batch_size),  // n  
        static_cast<int>(in_dim),      // k
        &alpha,
        weight,                // A
        maca_dtype,            // A type
        static_cast<int>(in_dim),      // lda
        in,                    // B
        maca_dtype,            // B type
        static_cast<int>(in_dim),      // ldb
        &beta,
        out,                   // C
        maca_dtype,            // C type
        static_cast<int>(out_dim),     // ldc
        MCBLAS_COMPUTE_32F,    // compute type
        MCBLAS_GEMM_DEFAULT    // algorithm
    );
}

// Template specializations for mcblas version
template <typename T>
void linear_mcblas(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                   const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                   llaisysDataType_t dtype);

template <>
void linear_mcblas<float>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                          const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                          llaisysDataType_t dtype) {
    auto out = reinterpret_cast<float *>(out_bytes);
    auto in = reinterpret_cast<const float *>(in_bytes);
    auto weight = reinterpret_cast<const float *>(weight_bytes);
    
    gemm_mcblas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
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
void linear_mcblas<fp16_t_metax>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                                 const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                                 llaisysDataType_t dtype) {
    auto out = reinterpret_cast<fp16_t_metax *>(out_bytes);
    auto in = reinterpret_cast<const fp16_t_metax *>(in_bytes);
    auto weight = reinterpret_cast<const fp16_t_metax *>(weight_bytes);
    
    gemm_mcblas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
    if (bias_bytes) {
        auto bias = reinterpret_cast<const fp16_t_metax *>(bias_bytes);
        size_t total = batch_size * out_dim;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;
        addBiasKernel<fp16_t_metax><<<numBlocks, blockSize>>>(out, bias, batch_size, out_dim);
    }
}

template <>
void linear_mcblas<bf16_t_metax>(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                                 const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim,
                                 llaisysDataType_t dtype) {
    auto out = reinterpret_cast<bf16_t_metax *>(out_bytes);
    auto in = reinterpret_cast<const bf16_t_metax *>(in_bytes);
    auto weight = reinterpret_cast<const bf16_t_metax *>(weight_bytes);
    
    gemm_mcblas(out, weight, in, out_dim, batch_size, in_dim, dtype);
    
    if (bias_bytes) {
        auto bias = reinterpret_cast<const bf16_t_metax *>(bias_bytes);
        size_t total = batch_size * out_dim;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;
        addBiasKernel<bf16_t_metax><<<numBlocks, blockSize>>>(out, bias, batch_size, out_dim);
    }
}

// Fallback: Improved linear kernel with more accurate accumulation
// Uses Kahan summation for better precision
template <typename T>
__global__ void linearKernel(T *out, const T *in, const T *weight, const T *bias,
                             size_t batch_size, size_t in_dim, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;

    for (size_t linear_idx = idx; linear_idx < total; linear_idx += blockDim.x * gridDim.x) {
        size_t i = linear_idx / out_dim;
        size_t j = linear_idx % out_dim;

        // Kahan summation for better precision
        float sum = 0.0f;
        float c = 0.0f;  // Compensation for lost low-order bits
        
        for (size_t k = 0; k < in_dim; ++k) {
            float x = to_float_metax(in[i * in_dim + k]);
            float w = to_float_metax(weight[j * in_dim + k]);
            float y = x * w - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        if (bias) {
            sum += to_float_metax(bias[j]);
        }

        out[linear_idx] = from_float_metax<T>(sum);
    }
}

// Alternative: blocked accumulation for very large dimensions
template <typename T>
__global__ void linearKernelBlocked(T *out, const T *in, const T *weight, const T *bias,
                                    size_t batch_size, size_t in_dim, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;
    
    const size_t BLOCK_SIZE = 256;  // Process in blocks for better precision

    for (size_t linear_idx = idx; linear_idx < total; linear_idx += blockDim.x * gridDim.x) {
        size_t i = linear_idx / out_dim;
        size_t j = linear_idx % out_dim;

        float sum = 0.0f;
        
        // Process in blocks
        for (size_t k_start = 0; k_start < in_dim; k_start += BLOCK_SIZE) {
            float block_sum = 0.0f;
            size_t k_end = min(k_start + BLOCK_SIZE, in_dim);
            
            for (size_t k = k_start; k < k_end; ++k) {
                float x = to_float_metax(in[i * in_dim + k]);
                float w = to_float_metax(weight[j * in_dim + k]);
                block_sum += x * w;
            }
            sum += block_sum;
        }

        if (bias) {
            sum += to_float_metax(bias[j]);
        }

        out[linear_idx] = from_float_metax<T>(sum);
    }
}

template <typename T>
void linear_fallback(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
                     const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);
    auto bias = reinterpret_cast<const T *>(bias_bytes);

    size_t total_outputs = batch_size * out_dim;
    const int blockSize = 256;
    const int numBlocks = (total_outputs + blockSize - 1) / blockSize;
    
    // Use blocked kernel for large dimensions (common in transformers)
    if (in_dim > 512) {
        linearKernelBlocked<T><<<numBlocks, blockSize>>>(out, in, weight, bias, batch_size, in_dim, out_dim);
    } else {
        linearKernel<T><<<numBlocks, blockSize>>>(out, in, weight, bias, batch_size, in_dim, out_dim);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim) {
    // Use mcBLAS for better performance on large matrices
    // Fallback to custom kernel for small matrices or special cases
    const size_t USE_MCBLAS_THRESHOLD = 256;  // Use mcBLAS for dim >= 256
    
    bool use_mcblas = (batch_size >= USE_MCBLAS_THRESHOLD || in_dim >= USE_MCBLAS_THRESHOLD || 
                       out_dim >= USE_MCBLAS_THRESHOLD);
    
    if (use_mcblas) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return linear_mcblas<float>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
        case LLAISYS_DTYPE_BF16:
            return linear_mcblas<bf16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
        case LLAISYS_DTYPE_F16:
            return linear_mcblas<fp16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim, type);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    } else {
        // Fallback to custom kernel
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return linear_fallback<float>(out, in, weight, bias, batch_size, in_dim, out_dim);
        case LLAISYS_DTYPE_BF16:
            return linear_fallback<bf16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim);
        case LLAISYS_DTYPE_F16:
            return linear_fallback<fp16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}

} // namespace llaisys::ops::metax
