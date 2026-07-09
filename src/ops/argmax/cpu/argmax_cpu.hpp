#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// 声明 Argmax 的 CPU 实现函数
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            size_t numel, 
            llaisysDataType_t dtype,      // 输入数据类型 (fp32/fp16...)
            llaisysDataType_t idx_dtype); // 索引数据类型 (int64/int32)

} // namespace llaisys::ops::cpu