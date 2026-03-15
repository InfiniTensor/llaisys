#include "rope_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cmath>

namespace {

// 将不同 T 转为 float 做计算
template <typename T> __device__ __forceinline__ float to_float_t(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float to_float_t(half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float to_float_t(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// 将 float 转回不同 T
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

// in/out: [seqlen, nhead, head_dim]
// pos_ids: [seqlen]
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seqlen, size_t nhead, size_t head_dim,
                            float theta) {
  const size_t bid = blockIdx.x;
  if (bid >= seqlen * nhead) {
    return;
  }

  const size_t seqlen_idx = bid / nhead;
  const size_t head_id = bid % nhead;

  const size_t half = head_dim / 2;
  const size_t offset = (seqlen_idx * nhead + head_id) * head_dim;
  const float pos_val = to_float_t(pos_ids[seqlen_idx]);

  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
    const float denom = powf(theta, exponent);
    const float phi = pos_val / denom;
    const float sinv = sinf(phi);
    const float cosv = cosf(phi);

    const float a = to_float_t(in[offset + j]);
    const float b = to_float_t(in[offset + j + half]);

    const float outa = a * cosv - b * sinv;
    const float outb = b * cosv + a * sinv;
    
    out[offset + j] = from_float_t<T>(outa);
    out[offset + j + half] = from_float_t<T>(outb);
  }
}

} // namespace

namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim,
          float theta) {
  if (seqlen == 0 || nhead == 0 || head_dim == 0) {
    return;
  }

  const size_t total_heads = seqlen * nhead;
  constexpr int block_size = 256;
  const int grid_size = static_cast<int>(total_heads);

  switch (type) {
  case LLAISYS_DTYPE_F32:
    rope_kernel<float><<<grid_size, block_size>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
        pos_ids, seqlen, nhead, head_dim, theta);
    break;
  case LLAISYS_DTYPE_F16:
    rope_kernel<half><<<grid_size, block_size>>>(
        reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
        pos_ids, seqlen, nhead, head_dim, theta);
    break;
  case LLAISYS_DTYPE_BF16:
    rope_kernel<__nv_bfloat16>
        <<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out),
                                    reinterpret_cast<const __nv_bfloat16 *>(in),
                                    pos_ids, seqlen, nhead, head_dim, theta);
    break;
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }

  CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
