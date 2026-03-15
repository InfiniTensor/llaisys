#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t type,
            size_t numel);

} // namespace llaisys::ops::metax

