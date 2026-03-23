#pragma once

#include "../../../utils.hpp"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, const size_t *shape, const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides, size_t ndim, size_t elem_size, size_t numel);
}
