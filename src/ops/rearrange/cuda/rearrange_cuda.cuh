#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda {
void rearrange(std::byte *out, const std::byte *in,
               const size_t *shape, const ptrdiff_t *out_strides, const ptrdiff_t *in_strides,
               size_t ndim, size_t esize, size_t numel);
}
