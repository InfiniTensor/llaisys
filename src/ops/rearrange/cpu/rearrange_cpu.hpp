#pragma once
#include "../../../core/llaisys_core.hpp"
#include <cstddef>
#include <cstdint>
#include <vector> 

namespace llaisys::ops::cpu {
void rearrange(std::byte *dst,
               const std::byte *src,
               llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &dst_strides,
               const std::vector<ptrdiff_t> &src_strides);
}