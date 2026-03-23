#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvh,
                    size_t dim, size_t dv, float scale);
void self_attention_segmented(std::byte *out,
                              const std::byte *q,
                              const std::byte *k,
                              const std::byte *v,
                              llaisysDataType_t type,
                              size_t qlen,
                              size_t kvlen,
                              size_t nhead,
                              size_t nkvh,
                              size_t dim,
                              size_t dv,
                              float scale,
                              const int64_t *q_offsets,
                              const int64_t *kv_offsets,
                              size_t nseg);
}
