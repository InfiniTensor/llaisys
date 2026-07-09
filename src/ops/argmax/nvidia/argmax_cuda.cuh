#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// Argmax CUDA 实现
// 使用两阶段并行规约：block 内 warp reduce → atomicMax 全局归约
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia
