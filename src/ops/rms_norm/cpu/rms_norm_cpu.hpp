#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// 声明 RMSNorm 的 CPU 实现函数
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, size_t num_rows, size_t hidden_size,
              llaisysDataType_t dtype);

} // namespace llaisys::ops::cpu