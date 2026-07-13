#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void linear(std::byte *output,
            const std::byte *input,
            const std::byte *weight,
            const std::byte *bias,
            size_t N,
            size_t M,
            size_t K,
            llaisysDataType_t dtype);

} // namespace llaisys::ops::nvidia
