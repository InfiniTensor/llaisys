#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Elementwise add: c = a + b
// Pointers are device pointers.
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);

} // namespace llaisys::ops::nvidia

