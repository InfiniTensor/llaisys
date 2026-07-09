#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t batch, size_t in_features, size_t out_features, bool has_bias);
}
