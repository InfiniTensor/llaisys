#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// 声明函数接口。该接口接收原始数据指针(std::byte*)和类型信息，
// 以便处理通用数据
// const std::byte *vals-含义：指向输入数据数组的指针
// std::byte *max_val-含义：指向输出最大值存储位置的指针
// std::byte *max_idx-含义：指向输出索引存储位置的指针
// llaisysDataType_t val_dtype-含义：数值的数据类型枚举
// llaisysDataType_t idx_dtype-含义：索引的数据类型枚举。
// size_t numel-含义：元素总数 (Number of Elements)。

void argmax(const std::byte *vals, std::byte *max_val, std::byte *max_idx, 
            llaisysDataType_t val_dtype, llaisysDataType_t idx_dtype, size_t numel);

} // namespace llaisys::ops::cpu