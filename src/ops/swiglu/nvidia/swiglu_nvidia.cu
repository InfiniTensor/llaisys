#include "swiglu_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    const float g = to_float(gate[idx]);
    const float u = to_float(up[idx]);
    const float silu = g / (1.0f + expf(-g));
    out[idx] = from_float<T>(u * silu);
}

template <typename T>
void launch_swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(numel, threads);
    auto stream = current_stream();
    swiglu_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    check_cuda(cudaGetLastError(), "swiglu_kernel launch");
}

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t num_elements) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu<float>(out, gate, up, num_elements);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu<llaisys::fp16_t>(out, gate, up, num_elements);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu<llaisys::bf16_t>(out, gate, up, num_elements);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
