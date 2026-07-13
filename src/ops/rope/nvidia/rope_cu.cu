#include "rope_cu.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../../../utils/check.hpp"
#include "../../../utils/types.hpp"

namespace ropeops::nvidia {

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// blockIdx.y = flattened (seq * num_head + head)
// blockIdx.x * blockDim.x + threadIdx.x = i (index in [0, head_dim/2))
template <typename T>
__global__ void
rope_kernel(T *output, const T *input, const int64_t *pos_ids,
            size_t seqlen, size_t num_head, size_t head_dim, float theta) {
    size_t seq_head = blockIdx.y;
    size_t seq  = seq_head / num_head;
    size_t head = seq_head % num_head;
    size_t i    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t dim_half = head_dim / 2;

    if (seq >= seqlen || i >= dim_half) return;

    float pos   = static_cast<float>(pos_ids[seq]);
    float angle = pos / powf(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    size_t base = seq * num_head * head_dim + head * head_dim;
    float x1 = to_float(input[base + i]);
    float x2 = to_float(input[base + i + dim_half]);

    output[base + i]           = from_float<T>(x1 * cos_a - x2 * sin_a);
    output[base + i + dim_half] = from_float<T>(x1 * sin_a + x2 * cos_a);
}

} // namespace ropeops::nvidia

namespace llaisys::ops::nvidia {

void rope(std::byte *output,
          const std::byte *input,
          const std::byte *pos_ids,
          size_t seqlen,
          size_t num_head,
          size_t head_dim,
          float theta,
          llaisysDataType_t dtype) {
    size_t dim_half = head_dim / 2;
    constexpr int block_x = 64;
    dim3 block(block_x);
    dim3 grid(static_cast<unsigned int>((dim_half + block_x - 1) / block_x),
              static_cast<unsigned int>(seqlen * num_head));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        ropeops::nvidia::rope_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const int64_t *>(pos_ids),
            seqlen, num_head, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        ropeops::nvidia::rope_kernel<__half><<<grid, block>>>(
            reinterpret_cast<__half *>(output),
            reinterpret_cast<const __half *>(input),
            reinterpret_cast<const int64_t *>(pos_ids),
            seqlen, num_head, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        ropeops::nvidia::rope_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16 *>(output),
            reinterpret_cast<const __nv_bfloat16 *>(input),
            reinterpret_cast<const int64_t *>(pos_ids),
            seqlen, num_head, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
