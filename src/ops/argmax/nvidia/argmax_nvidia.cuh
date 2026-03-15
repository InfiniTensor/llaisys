#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

void argmax(int64_t *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t type,
            size_t numel);

} // namespace llaisys::ops::nvidia
