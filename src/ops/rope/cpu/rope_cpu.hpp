#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos,
          llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d,
          float theta);
}