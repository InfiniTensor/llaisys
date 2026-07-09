#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void rms_norm(const std::byte *input_data,
              const std::byte *weight_data,
              std::byte *out_data,
              llaisysDataType_t dtype,
              size_t num_rows,
              size_t hidden_dim,
              float eps);

}