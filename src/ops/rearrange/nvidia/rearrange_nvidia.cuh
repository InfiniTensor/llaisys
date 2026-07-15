#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out,
               const std::byte *in,
               llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<size_t> &stride_in,
               const std::vector<size_t> &stride_out);
}
