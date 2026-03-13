#include "add_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::metax {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float af = to_float_metax(a[idx]);
        float bf = to_float_metax(b[idx]);
        c[idx] = from_float_metax<T>(af + bf);
    }
}

template <typename T>
void add_(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    auto c_ptr = reinterpret_cast<T *>(c);
    auto a_ptr = reinterpret_cast<const T *>(a);
    auto b_ptr = reinterpret_cast<const T *>(b);

    const int blockSize = 256;
    const int numBlocks = (numel + blockSize - 1) / blockSize;

    addKernel<T><<<numBlocks, blockSize>>>(c_ptr, a_ptr, b_ptr, numel);
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_<float>(c, a, b, size);
    case LLAISYS_DTYPE_BF16:
        return add_<bf16_t_metax>(c, a, b, size);
    case LLAISYS_DTYPE_F16:
        return add_<fp16_t_metax>(c, a, b, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
