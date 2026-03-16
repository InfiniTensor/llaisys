/**
 * NVIDIA CUDA 算子声明，供 op.cpp 在 LLAISYS_DEVICE_NVIDIA 分支调用。
 * 仅在使用 ENABLE_NVIDIA_API 编译时由 op 引用；实现位于 src/ops/nvidia/ops_nvidia.cu。
 */
#ifndef LLAISYS_OPS_NVIDIA_H
#define LLAISYS_OPS_NVIDIA_H

#include "../llaisys.h"
#include <cstddef>

#ifdef ENABLE_NVIDIA_API

namespace llaisys::ops::nvidia {

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel);

void embedding(std::byte *out, const std::byte *weight, const int64_t *index, size_t num_index, size_t embed_dim, size_t vocab_size, size_t elem_size);

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, size_t B, size_t M, size_t K);

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t vals_dtype, size_t numel);

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype, size_t rows, size_t dim, float eps);

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta);

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel);

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t num_heads, size_t nkvh, size_t head_dim, float scale);

} // namespace llaisys::ops::nvidia

#endif // ENABLE_NVIDIA_API

#endif // LLAISYS_OPS_NVIDIA_H
