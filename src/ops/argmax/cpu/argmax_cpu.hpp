#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *mi, std::byte *mv, const std::byte *v, llaisysDataType_t type, size_t size);
}