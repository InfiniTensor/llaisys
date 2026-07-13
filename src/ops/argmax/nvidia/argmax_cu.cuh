#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype);

}