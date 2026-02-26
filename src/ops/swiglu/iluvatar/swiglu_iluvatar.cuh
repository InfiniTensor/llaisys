#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel);
}
