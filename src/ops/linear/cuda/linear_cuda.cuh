#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t N, size_t K, bool has_bias);
}
