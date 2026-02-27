#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);

} // namespace llaisys::ops::metax
