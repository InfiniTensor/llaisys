#pragma once

#include "../../../device/runtime_api.hpp"
#include "llaisys/tensor.h"

#include <cstddef>

namespace llaisys::ops::metax {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size);

// Debug functions
void test_f16_write(std::byte *out, int mode);
void test_f32_write(std::byte *out);

} // namespace llaisys::ops::metax
