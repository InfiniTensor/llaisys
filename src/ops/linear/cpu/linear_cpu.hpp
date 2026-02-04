#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            const size_t M, const size_t N, const size_t K, llaisysDataType_t type);
}