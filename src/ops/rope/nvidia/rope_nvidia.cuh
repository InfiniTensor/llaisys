#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::ops::nvidia {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          float theta,
          llaisysDataType_t dtype,
          const std::vector<size_t> &shape);
}
