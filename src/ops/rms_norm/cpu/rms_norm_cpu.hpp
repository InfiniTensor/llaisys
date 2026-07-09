#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps, llaisysDataType_t dtype, std::vector<size_t> shapes);
}