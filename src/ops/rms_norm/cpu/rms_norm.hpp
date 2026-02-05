#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t batch_size,
    size_t feature_size,
    float epsilon
);
} // namespace llaisys::ops::cpu