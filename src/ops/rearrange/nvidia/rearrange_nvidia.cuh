#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides, llaisysDataType_t type, llaisysStream_t stream);
}
#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides, llaisysDataType_t type);
}
