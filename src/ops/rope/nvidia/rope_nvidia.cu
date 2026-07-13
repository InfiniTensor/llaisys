#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half_dim = head_dim / 2;
    const size_t total = seq_len * n_heads * half_dim;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linear_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < total; linear_idx += stride) {
        const size_t j = linear_idx % half_dim;
        const size_t tmp = linear_idx / half_dim;
        const size_t h = tmp % n_heads;
        const size_t i = tmp / n_heads;
        const size_t base_idx = (i * n_heads + h) * head_dim;

        const float pos = static_cast<float>(pos_ids[i]);
        const float phi = pos / powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
        const float cos_phi = cosf(phi);
        const float sin_phi = sinf(phi);

        const float a = llaisys::device::nvidia::scalarToFloat<T>(in[base_idx + j]);
        const float b = llaisys::device::nvidia::scalarToFloat<T>(in[base_idx + j + half_dim]);
        out[base_idx + j] = llaisys::device::nvidia::floatToScalar<T>(a * cos_phi - b * sin_phi);
        out[base_idx + j + half_dim] = llaisys::device::nvidia::floatToScalar<T>(b * cos_phi + a * sin_phi);
    }
}

} // namespace

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, llaisysStream_t stream) {
    const size_t total = seq_len * n_heads * (head_dim / 2);
    if (total == 0) {
        return;
    }

    constexpr int threads = 256;
    const int max_blocks = 4096;
    const int blocks = static_cast<int>(std::min<size_t>((total + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        ropeKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids,
            theta,
            seq_len,
            n_heads,
            head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        ropeKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids,
            theta,
            seq_len,
            n_heads,
            head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        ropeKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids,
            theta,
            seq_len,
            n_heads,
            head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
