#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias, // may be nullptr
            llaisysDataType_t type,
            size_t N, size_t M, size_t K);
}