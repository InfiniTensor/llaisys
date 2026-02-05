#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
    void rope(
        void* out,
        const void* in,
        const void* pos_id,
        float theta,
        size_t seqlen,
        size_t nhead,
        size_t d,
        llaisysDataType_t dtype
    );
} 