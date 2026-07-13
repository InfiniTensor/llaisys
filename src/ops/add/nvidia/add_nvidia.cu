#include "add_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        const float value = llaisys::device::nvidia::scalarToFloat<T>(a[idx])
                          + llaisys::device::nvidia::scalarToFloat<T>(b[idx]);
        c[idx] = llaisys::device::nvidia::floatToScalar<T>(value);
    }
}

} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    const int max_blocks = 4096;
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((numel + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    if (blocks == 0) {
        return;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        addKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        addKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        addKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
