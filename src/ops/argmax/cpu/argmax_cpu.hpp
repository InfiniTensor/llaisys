#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

// max_val应为std::byte*，用于支持多种数据类型的通用内存写入，不能简单换成float*等具体类型，否则类型不兼容。
namespace llaisys::ops::cpu {
void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
}