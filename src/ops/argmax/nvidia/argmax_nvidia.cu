#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <limits>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float shared_vals[256];
    __shared__ int64_t shared_idx[256];

    const int tid = threadIdx.x;
    float thread_max = -std::numeric_limits<float>::infinity();
    int64_t thread_idx = 0;

    for (size_t idx = static_cast<size_t>(tid); idx < numel; idx += blockDim.x) {
        const float value = llaisys::device::nvidia::scalarToFloat<T>(vals[idx]);
        if (value > thread_max) {
            thread_max = value;
            thread_idx = static_cast<int64_t>(idx);
        }
    }

    shared_vals[tid] = thread_max;
    shared_idx[tid] = thread_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float other_val = shared_vals[tid + offset];
            const int64_t other_idx = shared_idx[tid + offset];
            if (other_val > shared_vals[tid]) {
                shared_vals[tid] = other_val;
                shared_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = shared_idx[0];
        max_val[0] = llaisys::device::nvidia::floatToScalar<T>(shared_vals[0]);
    }
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    constexpr int threads = 256;
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);
    auto *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmaxKernel<<<1, threads, 0, cuda_stream>>>(
            max_idx_ptr,
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmaxKernel<<<1, threads, 0, cuda_stream>>>(
            max_idx_ptr,
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmaxKernel<<<1, threads, 0, cuda_stream>>>(
            max_idx_ptr,
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
