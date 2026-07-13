#include "rearrange_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void rearrangeKernel(T *out, const T *in, size_t numel, size_t ndim, const size_t *shape, const ptrdiff_t *out_strides,
                                const ptrdiff_t *in_strides) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t linear_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < numel; linear_idx += stride) {
        size_t remaining = linear_idx;
        ptrdiff_t out_offset = 0;
        ptrdiff_t in_offset = 0;
        for (ptrdiff_t dim = static_cast<ptrdiff_t>(ndim) - 1; dim >= 0; --dim) {
            const size_t coord = remaining % shape[dim];
            remaining /= shape[dim];
            out_offset += static_cast<ptrdiff_t>(coord) * out_strides[dim];
            in_offset += static_cast<ptrdiff_t>(coord) * in_strides[dim];
        }
        out[out_offset] = in[in_offset];
    }
}

template <typename T>
void launchRearrange(std::byte *out, const std::byte *in, size_t numel, size_t ndim, const size_t *shape_dev,
                     const ptrdiff_t *out_strides_dev, const ptrdiff_t *in_strides_dev, cudaStream_t stream) {
    constexpr int threads = 256;
    const int max_blocks = 4096;
    const int blocks = static_cast<int>(std::min<size_t>((numel + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    rearrangeKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        numel,
        ndim,
        shape_dev,
        out_strides_dev,
        in_strides_dev);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides, llaisysDataType_t type, llaisysStream_t stream) {
    size_t numel = 1;
    for (size_t dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return;
    }

    const size_t ndim = shape.size();
    auto cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    size_t *shape_dev = nullptr;
    ptrdiff_t *out_strides_dev = nullptr;
    ptrdiff_t *in_strides_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&shape_dev, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&out_strides_dev, ndim * sizeof(ptrdiff_t)));
    CUDA_CHECK(cudaMalloc(&in_strides_dev, ndim * sizeof(ptrdiff_t)));
    CUDA_CHECK(cudaMemcpyAsync(shape_dev, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(out_strides_dev, out_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(in_strides_dev, in_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchRearrange<float>(out, in, numel, ndim, shape_dev, out_strides_dev, in_strides_dev, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchRearrange<llaisys::fp16_t>(out, in, numel, ndim, shape_dev, out_strides_dev, in_strides_dev, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchRearrange<llaisys::bf16_t>(out, in, numel, ndim, shape_dev, out_strides_dev, in_strides_dev, cuda_stream);
        break;
    case LLAISYS_DTYPE_I64:
        launchRearrange<int64_t>(out, in, numel, ndim, shape_dev, out_strides_dev, in_strides_dev, cuda_stream);
        break;
    default:
        CUDA_CHECK(cudaFree(shape_dev));
        CUDA_CHECK(cudaFree(out_strides_dev));
        CUDA_CHECK(cudaFree(in_strides_dev));
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaFree(shape_dev));
    CUDA_CHECK(cudaFree(out_strides_dev));
    CUDA_CHECK(cudaFree(in_strides_dev));
}

} // namespace llaisys::ops::nvidia
