#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::metax {

void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t type,
          size_t seqlen,
          size_t nhead,
          size_t head_dim,
          float theta);

} // namespace llaisys::ops::metax
