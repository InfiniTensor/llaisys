#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        const float gate_val = llaisys::device::nvidia::scalarToFloat<T>(gate[idx]);
        const float up_val = llaisys::device::nvidia::scalarToFloat<T>(up[idx]);
        const float silu = gate_val / (1.0f + expf(-gate_val));
        out[idx] = llaisys::device::nvidia::floatToScalar<T>(up_val * silu);
    }
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    constexpr int threads = 256;
    const int max_blocks = 4096;
    const int blocks = static_cast<int>(std::min<size_t>((numel + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swigluKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        swigluKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swigluKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
