#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// Linear CUDA 实现（Y = X * W^T + b）
// 使用 cuBLAS SGEMM（f32 TF32 Tensor Core）/ HGEMM（f16/bf16 BF16 Tensor Core）
// A100 SM_80 在 BF16 下峰值 312 TFLOPS，在 TF32 下峰值 156 TFLOPS
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t batch_size, size_t in_features, size_t out_features);
} // namespace llaisys::ops::nvidia
