#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// 声明 Linear 的 CPU 实现函数
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t M, size_t K, size_t N, llaisysDataType_t dtype);

} // namespace llaisys::ops::cpu