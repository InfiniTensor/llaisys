#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_size, size_t embed_dim);

} // namespace llaisys::ops::metax
