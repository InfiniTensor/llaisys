#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim, float theta);
} // namespace llaisys::ops::nvidia