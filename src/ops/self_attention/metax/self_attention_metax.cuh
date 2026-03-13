#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seqlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, size_t total_len);

} // namespace llaisys::ops::metax
