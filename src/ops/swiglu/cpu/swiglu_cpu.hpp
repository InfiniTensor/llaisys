#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {


void swiglu(const std::byte* gate_data,
            const std::byte* up_data,
            std::byte* out_data,
            llaisysDataType_t dtype,
            size_t numel);

} // namespace llaisys::ops::cpu