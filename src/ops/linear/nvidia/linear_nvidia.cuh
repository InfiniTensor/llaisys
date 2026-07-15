#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t dimi,
            size_t dimk,
            size_t dimj);
}
