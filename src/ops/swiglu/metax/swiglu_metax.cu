#include "swiglu_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::metax {

// SiLU (Swish) activation: x * sigmoid(x)
__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float g = to_float_metax(gate[idx]);
        float u = to_float_metax(up[idx]);
        float result = u * silu(g);
        out[idx] = from_float_metax<T>(result);
    }
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
        return swiglu_<bf16_t_metax>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<fp16_t_metax>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
