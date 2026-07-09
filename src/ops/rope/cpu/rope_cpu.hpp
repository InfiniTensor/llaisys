#pragma once

#include "llaisys.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::ops::cpu {
    void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t dtype, const std::vector<size_t> &shape);
} // namespace llaisys::ops::cpu