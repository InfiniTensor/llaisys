#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               size_t esize, size_t numel);
}
