#include "llaisys.h"
#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"
#include <cstddef>

namespace {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T local_val) {
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    local_val += __shfl_xor_sync(0xffffffff, local_val, stride);
  }
  return local_val;
}

template <typename T, const int BLOCK_SIZE = 256>
__device__ __forceinline__ T block_reduce_sum(T local_val) {
  constexpr int warp_per_block = CEIL(BLOCK_SIZE, 32);
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  __shared__ T shared_val[warp_per_block];

  local_val = warp_reduce_sum(local_val);
  if (lane_id == 0) {
    shared_val[warp_id] = local_val;
  }
  __syncthreads();

  T block_sum{0};
  T lane_val = lane_id < warp_per_block ? shared_val[lane_id] : 0;
  block_sum = warp_reduce_sum(lane_val);
  return block_sum;
}

template <typename T> __device__ __forceinline__ float to_float_t(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float to_float_t(half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float to_float_t(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T from_float_t(float v) {
  return static_cast<T>(v);
}
template <> __device__ __forceinline__ half from_float_t<half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float_t<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t M,
                                size_t N, float eps) {
  const size_t row_id = blockIdx.x;
  if (row_id >= M)
    return;

  const int tid = threadIdx.x;

  // 1. 每个线程求局部平方和（用 float 累加）
  float sum_thread = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    float v = to_float_t(in[row_id * N + i]);
    sum_thread += v * v;
  }

  // 2. block 内归约得到整行平方和，所有线程得到同一 sum_sq
  float sum_block = block_reduce_sum<float>(sum_thread);
  float mean_sq = sum_block / static_cast<float>(N);
  float scale_rms = 1.0f / sqrtf(mean_sq + eps);

  // 3. 归一化并写回：out[i] = in[i] * weight[i] * scale_rms
  for (int i = tid; i < N; i += blockDim.x) {
    float x = to_float_t(in[row_id * N + i]);
    float w = to_float_t(weight[i]);
    float y = x * w * scale_rms;
    out[row_id * N + i] = from_float_t<T>(y);
  }
}

} // namespace

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t N, float eps) {
  if (M == 0 || N == 0)
    return;
  constexpr int block_size = 256;
  const int grid_size = static_cast<int>(M);
  switch (type) {
  case LLAISYS_DTYPE_F32:
    rms_norm_kernel<float><<<grid_size, block_size>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
        reinterpret_cast<const float *>(weight), M, N, eps);
    break;
  case LLAISYS_DTYPE_F16:
    rms_norm_kernel<half><<<grid_size, block_size>>>(
        reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
        reinterpret_cast<const half *>(weight), M, N, eps);
    break;
  case LLAISYS_DTYPE_BF16:
    rms_norm_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(in),
        reinterpret_cast<const __nv_bfloat16 *>(weight), M, N, eps);
    break;
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
  CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
