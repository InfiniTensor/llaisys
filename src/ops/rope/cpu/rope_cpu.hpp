#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, const size_t seq_len, const size_t nhead, const size_t d, float theta);
}