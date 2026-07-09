#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// 向量化 add 算子，支持 f32 / f16 / bf16
// A100 上使用 float4/__half2/__nv_bfloat162 宽加载提高带宽利用率
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia
