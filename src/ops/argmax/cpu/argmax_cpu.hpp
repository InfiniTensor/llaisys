#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t idx_type, llaisysDataType_t val_type, size_t numel);
}
