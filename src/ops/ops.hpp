#pragma once

#include "../tensor/tensor.hpp"

namespace llaisys::ops {

// Basic operations
void add(tensor_t c, tensor_t a, tensor_t b);

// Model operations
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);

void embedding(tensor_t out, tensor_t index, tensor_t weight);

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);

void swiglu(tensor_t out, tensor_t gate, tensor_t up);

} // namespace llaisys::ops
