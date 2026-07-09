#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t dv);
}