#include "add_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    c[idx] = from_float<T>(to_float(a[idx]) + to_float(b[idx]));
}

template <typename T>
void launch_add(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(numel, threads);
    auto stream = current_stream();
    add_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    check_cuda(cudaGetLastError(), "add_kernel launch");
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_add<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return launch_add<llaisys::fp16_t>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_add<llaisys::bf16_t>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
