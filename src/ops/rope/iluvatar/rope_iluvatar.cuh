#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t nhead, size_t d);
}
