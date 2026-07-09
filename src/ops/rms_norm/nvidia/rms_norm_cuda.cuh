#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// RMS Norm CUDA 实现
// 利用 warp-level __shfl_down_sync + shared memory 两阶段规约
// 每个 block 处理一行，适用于 A100 的 164KB 共享内存
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t num_rows, size_t row_dim, float eps);
} // namespace llaisys::ops::nvidia
