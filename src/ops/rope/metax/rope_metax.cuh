#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, float theta, size_t seqlen, size_t nhead, size_t d);

} // namespace llaisys::ops::metax
