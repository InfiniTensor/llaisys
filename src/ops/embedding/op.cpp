#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {

    CHECK_SAME_DEVICE(out, index, weight);

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: inputs must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be Int64.");

    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(out->ndim() == 2, "Embedding: output must be 2D.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output dim 0 must match index length.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output dim 1 must match weight dim 1.");

    size_t embedding_dim = weight->shape()[1];
    size_t num_embeddings = weight->shape()[0];
    size_t num_indices = index->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            weight->data(),
            index->data(),
            out->data(),
            weight->dtype(),
            embedding_dim,
            num_embeddings,
            num_indices);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            weight->data(),
            index->data(),
            out->data(),
            weight->dtype(),
            embedding_dim,
            num_embeddings,
            num_indices);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
