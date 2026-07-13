#pragma once

#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

void rope(std::byte *output,
          const std::byte *input,
          const std::byte *pos_ids,
          size_t seqlen,
          size_t num_head,
          size_t head_dim,
          float theta,
          llaisysDataType_t dtype);

} // namespace llaisys::ops::nvidia
