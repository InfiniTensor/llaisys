#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// out[i] = up[i] * (gate[i] / (1 + exp(-gate[i])))
void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel);
} // namespace llaisys::ops::cpu
