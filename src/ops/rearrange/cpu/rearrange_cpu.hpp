#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type,
               const size_t *shape, const ptrdiff_t *out_strides, 
               const ptrdiff_t *in_strides, size_t numel, size_t ndim);
}