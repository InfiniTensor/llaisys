#include "swiglu_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::ops::cuda {

__global__ void swiglu_kernel(void *out, const void *gate, const void *up, llaisysDataType_t type, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    float g = load_as_float(gate, idx, type);
    float u = load_as_float(up, idx, type);
    float sig = g / (1.0f + expf(-g));
    store_from_float(out, idx, u * sig, type);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "swiglu cuda only supports F32/F16/BF16");
    dim3 block(256);
    dim3 grid((numel + block.x - 1) / block.x);
    swiglu_kernel<<<grid, block>>>(out, gate, up, type, numel);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "swiglu cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
