#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
    // m: batch_size, n: out_features, k: in_features
    void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                llaisysDataType_t type, size_t m, size_t n, size_t k);
}