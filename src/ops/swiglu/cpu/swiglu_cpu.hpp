#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype);

}