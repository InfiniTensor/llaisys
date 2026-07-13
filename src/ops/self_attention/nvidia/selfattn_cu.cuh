#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void self_attn(std::byte *attn_val,
               const std::byte *q,
               const std::byte *k,
               const std::byte *v,
               size_t seqlen,
               size_t num_head,
               size_t head_dim,
               size_t kvlen,
               size_t num_kv_head,
               size_t vdim,
               float scale,
               llaisysDataType_t dtype);

} // namespace llaisys::ops::nvidia
