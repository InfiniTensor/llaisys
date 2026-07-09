#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void linear(const std::byte *input_data,
            const std::byte *weight_data,
            const std::byte *bias_data, // 可能是 nullptr
            std::byte *out_data,
            llaisysDataType_t dtype,
            size_t M, size_t K, size_t N);

}