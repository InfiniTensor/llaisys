#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t num,
               size_t dim);
}
