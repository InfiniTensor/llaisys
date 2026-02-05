#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(
    std::byte *output,
    const std::byte *input,
    llaisysDataType_t type,
    size_t batch_size,
    size_t feature_size
);
}