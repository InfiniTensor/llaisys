#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void rope(const std::byte *out_data,
          const std::byte *in_data,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,
          size_t n_heads,
          size_t head_dim,
          float theta);

} // namespace llaisys::ops::cpu