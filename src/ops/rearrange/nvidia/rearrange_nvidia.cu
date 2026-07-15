#include "rearrange_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void rearrange_kernel(T *out,
                                 const T *in,
                                 llaisys::ops::nvidia::TensorDescriptor desc_in,
                                 llaisys::ops::nvidia::TensorDescriptor desc_out,
                                 size_t total) {
    const size_t linear_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (linear_idx >= total) {
        return;
    }

    int64_t remaining = static_cast<int64_t>(linear_idx);
    int64_t in_offset = 0;
    int64_t out_offset = 0;
    for (int dim = desc_in.ndim - 1; dim >= 0; --dim) {
        const int64_t dim_idx = remaining % desc_in.shape[static_cast<size_t>(dim)];
        remaining /= desc_in.shape[static_cast<size_t>(dim)];
        in_offset += dim_idx * desc_in.strides[static_cast<size_t>(dim)];
        out_offset += dim_idx * desc_out.strides[static_cast<size_t>(dim)];
    }

    out[out_offset] = in[in_offset];
}

template <typename T>
void rearrange_impl(std::byte *out,
                    const std::byte *in,
                    const std::vector<size_t> &shape,
                    const std::vector<size_t> &stride_in,
                    const std::vector<size_t> &stride_out) {
    const auto desc_in = llaisys::ops::nvidia::make_descriptor(shape, stride_in);
    const auto desc_out = llaisys::ops::nvidia::make_descriptor(shape, stride_out);
    constexpr int threads = 256;
    const size_t total = llaisys::ops::nvidia::numel(shape);
    rearrange_kernel<<<llaisys::ops::nvidia::blocks_for(total, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        desc_in,
        desc_out,
        total);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void rearrange(std::byte *out,
               const std::byte *in,
               llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<size_t> &stride_in,
               const std::vector<size_t> &stride_out) {
    if (shape.empty()) {
        LLAISYS_CUDA_CHECK(cudaMemcpyAsync(out, in, utils::dsize(dtype), cudaMemcpyDeviceToDevice, current_stream()));
        return;
    }

    if (is_contiguous(shape, stride_in) && is_contiguous(shape, stride_out)) {
        LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
            out,
            in,
            numel(shape) * utils::dsize(dtype),
            cudaMemcpyDeviceToDevice,
            current_stream()));
        return;
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_impl<float>(out, in, shape, stride_in, stride_out);
    case LLAISYS_DTYPE_F16:
        return rearrange_impl<llaisys::fp16_t>(out, in, shape, stride_in, stride_out);
    case LLAISYS_DTYPE_BF16:
        return rearrange_impl<llaisys::bf16_t>(out, in, shape, stride_in, stride_out);
    case LLAISYS_DTYPE_I64:
        return rearrange_impl<int64_t>(out, in, shape, stride_in, stride_out);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
