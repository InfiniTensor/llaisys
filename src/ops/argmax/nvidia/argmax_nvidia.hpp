#pragma once

#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::nvidia {

// GPU 版 Argmax 声明
// max_index: 输出，存储最大值的索引 (int64_t)
// max_value: 输出，存储最大值 (类型同输入)
// values: 输入数据
// dtype: 数据类型
// num_elements: 元素个数
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia