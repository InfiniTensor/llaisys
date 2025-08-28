#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *out_idx,
            std::byte *out_val,
            const std::byte *vals,
            llaisysDataType_t type,
            size_t numel);
}