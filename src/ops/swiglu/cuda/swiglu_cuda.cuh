#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel);
}
