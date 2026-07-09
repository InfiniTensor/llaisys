#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_length, size_t head_nums, size_t head_dim, float theta);
}