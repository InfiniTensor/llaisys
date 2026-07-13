#pragma once
#include "llaisys.h"
#include <cstddef> 

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *input, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t K, size_t N);
} // namespace llaisys::ops::cpu