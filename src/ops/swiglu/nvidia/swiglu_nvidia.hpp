#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
} // namespace llaisys::ops::nvidia