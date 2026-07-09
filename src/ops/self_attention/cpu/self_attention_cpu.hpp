#pragma once

#include "llaisys.h"
#include <vector>
#include <cstddef>

namespace llaisys::ops::cpu {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t dtype, 
    const std::vector<size_t> &q_shape, 
    const std::vector<size_t> &k_shape, 
    const std::vector<size_t> &v_shape);

} // namespace llaisys::ops::cpu
