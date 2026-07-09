#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *W, 
    const std::byte *bias, llaisysDataType_t type, size_t batch,
    size_t in_dim, size_t out_dim);
}
