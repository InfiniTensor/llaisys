#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// SwiGLU CUDA 实现
// out_i = up_i * silu(gate_i)，向量化处理 f32/f16/bf16
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia
