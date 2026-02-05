#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(
    void* out, const void* q, const void* k, const void* v,
    size_t seqlen, size_t nhead, size_t d,
    size_t total_len, size_t nkvhead, size_t dv,
    float scale,
    llaisysDataType_t dtype
);
} 