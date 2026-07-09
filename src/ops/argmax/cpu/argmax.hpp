#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(size_t *max_idx, std::byte *max_val, const std::byte *input, llaisysDataType_t type, size_t size);
}