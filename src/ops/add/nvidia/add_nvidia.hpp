#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
// 这里的签名完全对齐 cpu::add 的设计，方便统一调用
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia