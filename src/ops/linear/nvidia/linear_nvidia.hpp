#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K);
} // namespace llaisys::ops::nvidia