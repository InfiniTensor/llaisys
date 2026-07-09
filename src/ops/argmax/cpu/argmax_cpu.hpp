#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void argmax(const std::byte *vals, std::byte *max_val, std::byte *max_idx, llaisysDataType_t dtype, size_t numel);
}