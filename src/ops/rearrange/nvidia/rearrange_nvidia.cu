#include "rearrange_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <vector>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rearrange_kernel(
    T *out,
    const T *in,
    const int64_t *shape,
    const int64_t *in_strides,
    const int64_t *out_strides,
    int rank,
    size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    size_t t = idx;
    int64_t in_off = 0;
    int64_t out_off = 0;

    for (int d = rank - 1; d >= 0; --d) {
        int64_t cur = static_cast<int64_t>(t % static_cast<size_t>(shape[d]));
        t /= static_cast<size_t>(shape[d]);
        in_off += cur * in_strides[d];
        out_off += cur * out_strides[d];
    }

    out[static_cast<size_t>(out_off)] = in[static_cast<size_t>(in_off)];
}

template <typename T>
void launch_rearrange(
    std::byte *out,
    const std::byte *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &in_strides,
    const std::vector<ptrdiff_t> &out_strides) {
    const int rank = static_cast<int>(shape.size());
    std::vector<int64_t> h_shape(rank);
    std::vector<int64_t> h_in_strides(rank);
    std::vector<int64_t> h_out_strides(rank);

    size_t total = 1;
    for (int i = 0; i < rank; ++i) {
        h_shape[i] = static_cast<int64_t>(shape[i]);
        h_in_strides[i] = static_cast<int64_t>(in_strides[i]);
        h_out_strides[i] = static_cast<int64_t>(out_strides[i]);
        total *= shape[i];
    }

    int64_t *d_shape = nullptr;
    int64_t *d_in_strides = nullptr;
    int64_t *d_out_strides = nullptr;

    const size_t meta_bytes = static_cast<size_t>(rank) * sizeof(int64_t);
    auto stream = current_stream();

    check_cuda(cudaMalloc(&d_shape, meta_bytes), "cudaMalloc d_shape");
    check_cuda(cudaMalloc(&d_in_strides, meta_bytes), "cudaMalloc d_in_strides");
    check_cuda(cudaMalloc(&d_out_strides, meta_bytes), "cudaMalloc d_out_strides");

    check_cuda(cudaMemcpyAsync(d_shape, h_shape.data(), meta_bytes, cudaMemcpyHostToDevice, stream), "copy shape");
    check_cuda(cudaMemcpyAsync(d_in_strides, h_in_strides.data(), meta_bytes, cudaMemcpyHostToDevice, stream), "copy in_strides");
    check_cuda(cudaMemcpyAsync(d_out_strides, h_out_strides.data(), meta_bytes, cudaMemcpyHostToDevice, stream), "copy out_strides");

    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(total, threads);
    rearrange_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        d_shape,
        d_in_strides,
        d_out_strides,
        rank,
        total);
    check_cuda(cudaGetLastError(), "rearrange_kernel launch");

    check_cuda(cudaFree(d_shape), "cudaFree d_shape");
    check_cuda(cudaFree(d_in_strides), "cudaFree d_in_strides");
    check_cuda(cudaFree(d_out_strides), "cudaFree d_out_strides");
}

void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t type,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &in_strides,
    const std::vector<ptrdiff_t> &out_strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rearrange<float>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_F16:
        return launch_rearrange<llaisys::fp16_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_BF16:
        return launch_rearrange<llaisys::bf16_t>(out, in, shape, in_strides, out_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
