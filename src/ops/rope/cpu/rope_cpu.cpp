#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
static void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seqlen,
                  size_t nhead, size_t head_dim, float theta) {
  const size_t half = head_dim / 2;

  // denom[j] = theta^(2j/d)
  std::vector<float> denom(half);
  for (size_t j = 0; j < half; ++j) {
    const float exponent =
        (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
    denom[j] = ::powf(theta, exponent);
  }

  for (size_t s = 0; s < seqlen; ++s) {
    // pos对应seqlen位置的position id
    const float p = static_cast<float>(pos_ids[s]);
    for (size_t h = 0; h < nhead; ++h) {
      const size_t offset = (s * nhead + h) * head_dim;
      // 将相邻的两个特征维度合并为一组，然后一起旋转
      for (size_t j = 0; j < half; ++j) {
        const float phi = p / denom[j];
        const float sinv = ::sinf(phi);
        const float cosv = ::cosf(phi);

        const float a = llaisys::utils::cast<float>(in[offset + j]);
        const float b = llaisys::utils::cast<float>(in[offset + j + half]);

        out[offset + j] = llaisys::utils::cast<T>(a * cosv - b * sinv);
        out[offset + j + half] = llaisys::utils::cast<T>(a * sinv + b * cosv);
      }
    }
  }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim,
          float theta) {
  switch (type) {
  case LLAISYS_DTYPE_F32:
    return rope_(reinterpret_cast<float *>(out),
                 reinterpret_cast<const float *>(in), pos_ids, seqlen, nhead,
                 head_dim, theta);
  case LLAISYS_DTYPE_F16:
    return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                 reinterpret_cast<const llaisys::fp16_t *>(in), pos_ids, seqlen,
                 nhead, head_dim, theta);
  case LLAISYS_DTYPE_BF16:
    return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                 reinterpret_cast<const llaisys::bf16_t *>(in), pos_ids, seqlen,
                 nhead, head_dim, theta);
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
}
} // namespace llaisys::ops::cpu
