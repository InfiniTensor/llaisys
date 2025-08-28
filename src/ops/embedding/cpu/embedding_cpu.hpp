#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t N, size_t D, size_t V);
}