#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_numel,
               size_t embedding_dim);

} // namespace llaisys::ops::nvidia