#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,float eps, llaisysDataType_t dtype, size_t N, size_t D);
}