#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void swiglu(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia