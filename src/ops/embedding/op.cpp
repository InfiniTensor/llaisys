#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

void embedding_cpu(std::byte *out, const std::byte *weight, const int64_t *index, size_t num_index, size_t embed_dim, size_t vocab_size, size_t elem_size) {
    size_t row_bytes = embed_dim * elem_size;
    for (size_t i = 0; i < num_index; i++) {
        int64_t row_idx = index[i];
        ASSERT(row_idx >= 0 && static_cast<size_t>(row_idx) < vocab_size, "embedding: index out of range");
        const std::byte *src = weight + static_cast<size_t>(row_idx) * row_bytes;
        std::memcpy(out + i * row_bytes, src, row_bytes);
    }
}

} // namespace

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be int64");
    ASSERT(out->dtype() == weight->dtype(), "embedding: out dtype must match weight");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "embedding: all tensors must be contiguous");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D");
    ASSERT(index->ndim() == 1, "embedding: index must be 1D");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D");
    size_t num_index = index->numel();
    size_t vocab_size = weight->shape()[0];
    size_t embed_dim = weight->shape()[1];
    ASSERT(out->shape()[0] == num_index && out->shape()[1] == embed_dim, "embedding: out shape must be (index_len, embed_dim)");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return embedding_cpu(out->data(), weight->data(), reinterpret_cast<const int64_t *>(index->data()), num_index, embed_dim, vocab_size, out->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return embedding_cpu(out->data(), weight->data(), reinterpret_cast<const int64_t *>(index->data()), num_index, embed_dim, vocab_size, out->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::embedding(out->data(), weight->data(), reinterpret_cast<const int64_t *>(index->data()), num_index, embed_dim, vocab_size, out->elementSize());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
