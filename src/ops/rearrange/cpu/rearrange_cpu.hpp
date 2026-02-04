#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

// Copy elements from `in` to `out` with arbitrary strides.
//
// - `shape` is in elements, length = `ndim`
// - `*_strides` are in elements (not bytes), length = `ndim`
// - `out` and `in` must have the same dtype and shape
void rearrange(std::byte *out,
               const std::byte *in,
               llaisysDataType_t dtype,
               const size_t *shape,
               const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides,
               size_t ndim);

} // namespace llaisys::ops::cpu

