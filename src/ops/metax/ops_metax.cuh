#pragma once

#include "../../utils.hpp"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::metax {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype, llaisysDataType_t idx_dtype);
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_tokens, size_t vocab_size, size_t hidden_size, llaisysDataType_t dtype);
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t M, size_t K, size_t N, llaisysDataType_t dtype);
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t num_rows, size_t hidden_size, llaisysDataType_t dtype);
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seq_len, size_t n_head, size_t head_dim, llaisysDataType_t dtype);
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale, llaisysDataType_t dtype);
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype);
}
