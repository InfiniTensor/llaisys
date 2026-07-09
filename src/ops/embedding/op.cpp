#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be I64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: out.shape[0] must equal index.shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out.shape[1] must equal weight.shape[1].");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    size_t index_len = index->shape()[0];
    size_t row_len = weight->shape()[1];

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index_len, row_len);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index_len, row_len);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), index_len, row_len);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
