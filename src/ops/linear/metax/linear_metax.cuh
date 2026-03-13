#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim);

} // namespace llaisys::ops::metax
