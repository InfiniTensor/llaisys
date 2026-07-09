#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void self_attention(const std::byte *q_data,
                    const std::byte *k_data,
                    const std::byte *v_data,
                    std::byte *attn_val_data,
                    llaisysDataType_t dtype,
                    size_t seqlen,
                    size_t total_len,
                    size_t nhead,
                    size_t nkvhead,
                    size_t head_dim,   // d (for Q and K)
                    size_t head_dim_v, // dv (for V and Output)
                    float scale);

} // namespace llaisys::ops::cpu