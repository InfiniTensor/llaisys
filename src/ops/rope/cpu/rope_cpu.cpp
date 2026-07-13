#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstdint>
#include <omp.h>

template <typename T>
void rope_impl(
    T *output, const T *input, const int64_t *pos_ids, size_t seqlen, size_t num_head, size_t head_dim, float theta) {

    size_t dim_half = head_dim / 2;

#pragma omp parallel for collapse(2)
    for (size_t seq_idx = 0; seq_idx < seqlen; seq_idx++) {
        float pos_id = static_cast<float>(pos_ids[seq_idx]);

        for (size_t head_idx = 0; head_idx < num_head; head_idx++) {
            for (size_t i = 0; i < dim_half; i++) {
                float angle = pos_id / std::pow(theta, (2.0f * i) / head_dim);
                float cos_angle = std::cos(angle);
                float sin_angle = std::sin(angle);

                size_t base_idx = seq_idx * num_head * head_dim + head_idx * head_dim;

                float x1 = casting(float, input[base_idx + i]);
                float x2 = casting(float, input[base_idx + i + dim_half]);

                output[base_idx + i] = casting(T, x1 * cos_angle - x2 * sin_angle);
                output[base_idx + i + dim_half] = casting(T, x1 * sin_angle + x2 * cos_angle);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(std::byte *output,
          const std::byte *input,
          const std::byte *pos_ids,
          size_t seqlen,
          size_t num_head,
          size_t head_dim,
          float theta,
          llaisysDataType_t dtype) {

    using namespace llaisys;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(recast(float *, output), recast(const float *, input), recast(const int64_t *, pos_ids),
                         seqlen, num_head, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(recast(fp16_t *, output), recast(const fp16_t *, input), recast(const int64_t *, pos_ids),
                         seqlen, num_head, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(recast(bf16_t *, output), recast(const bf16_t *, input), recast(const int64_t *, pos_ids),
                         seqlen, num_head, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu