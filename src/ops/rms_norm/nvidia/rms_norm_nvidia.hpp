#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
// 参数：c(输出), a(输入), b(权重), rows(Token数量), dim(特征维度), eps(防除零小浮点数)
void rms_norm(std::byte *c, const std::byte *a, const std::byte *b, size_t rows, size_t dim, float eps, llaisysDataType_t type);
} // namespace llaisys::ops::nvidia