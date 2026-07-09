#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(
    std::byte *output,
    const std::byte *input,
    const std::byte *gate,
    llaisysDataType_t type,
    size_t tatal_size
);
}