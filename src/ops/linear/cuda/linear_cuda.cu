#include "linear_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::ops::cuda {

__global__ void linear_kernel(void *out, const void *in, const void *weight, const void *bias, llaisysDataType_t type,
                              size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_features;
    if (idx >= total) {
        return;
    }

    size_t b = idx / out_features;
    size_t o = idx % out_features;

    float sum = 0.0f;
    for (size_t i = 0; i < in_features; ++i) {
        float x = load_as_float(in, b * in_features + i, type);
        float w = load_as_float(weight, o * in_features + i, type);
        sum += x * w;
    }
    if (has_bias) {
        sum += load_as_float(bias, o, type);
    }
    store_from_float(out, idx, sum, type);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type,
            size_t batch_size, size_t in_features, size_t out_features) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "linear cuda only supports F32/F16/BF16");
    size_t total = batch_size * out_features;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    linear_kernel<<<grid, block>>>(out, in, weight, bias, type, batch_size, in_features, out_features, bias != nullptr);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "linear cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
