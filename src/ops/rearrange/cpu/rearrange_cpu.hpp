#pragma once
#include "llaisys.h"
#include <vector>
#include <cstddef>

namespace llaisys::ops::cpu {

void rearrange(std::byte *out, const std::byte *in, 
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t dtype);

} // namespace llaisys::ops::cpu