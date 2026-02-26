#include "swiglu_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::iluvatar {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float g = to_float_cuda(gate[idx]);
    float u = to_float_cuda(up[idx]);
    
    // SwiGLU: up * sigmoid(gate) * gate
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    float result = u * sigmoid_g * g;
    
    out[idx] = from_float_cuda<T>(result);
}

template <typename T>
void swiglu_(std::byte *out_bytes, const std::byte *gate_bytes, const std::byte *up_bytes, size_t numel) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto gate = reinterpret_cast<const T *>(gate_bytes);
    auto up = reinterpret_cast<const T *>(up_bytes);
    
    const int blockSize = 256;
    const int numBlocks = (numel + blockSize - 1) / blockSize;
    
    swigluKernel<T><<<numBlocks, blockSize>>>(out, gate, up, numel);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<bf16_t_cuda>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<fp16_t_cuda>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
