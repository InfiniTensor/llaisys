#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);
}