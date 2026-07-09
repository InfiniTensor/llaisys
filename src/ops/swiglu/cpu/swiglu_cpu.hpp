#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);

// 元数据参数：
// llaisysDataType_t dtype
// 含义：数据类型枚举（如 LLAISYS_DTYPE_F32, LLAISYS_DTYPE_F16, LLAISYS_DTYPE_BF16）。
// 作用：决定了如何解释上述 std::byte* 指针。

// size_t numel
// 含义：需要处理的元素总数 (Number of Elements)。
// 计算方式：通常等于 seqlen * intermediate_size。

} // namespace llaisys::ops::cpu