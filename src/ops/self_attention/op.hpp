#pragma once

#include "../../tensor/tensor.hpp"
#include <cstdint>
#include <cstddef>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_segmented(tensor_t attn_val,
                              tensor_t q,
                              tensor_t k,
                              tensor_t v,
                              float scale,
                              const int64_t *q_offsets,
                              const int64_t *kv_offsets,
                              size_t nseg);
}
