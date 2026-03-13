#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t batch_size, size_t hidden_dim);

} // namespace llaisys::ops::metax
