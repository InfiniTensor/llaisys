#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// max_idx 存储索引，通常是 int64_t；max_val 和 vals 类型相同
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t type, size_t numel);
}