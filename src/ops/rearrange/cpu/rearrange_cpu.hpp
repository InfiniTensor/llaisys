#pragma once
#include "llaisys.h"
#include <vector>
#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type,
               const std::vector<size_t>& shape, 
               const std::vector<ptrdiff_t>& in_strides, 
               const std::vector<ptrdiff_t>& out_strides);
}