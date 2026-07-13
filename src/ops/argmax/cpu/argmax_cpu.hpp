#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype);

}