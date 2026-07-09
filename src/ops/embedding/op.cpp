#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Check device
    CHECK_SAME_DEVICE(out, index, weight);

    // index: (seq_len,), weight: (vocab_size, hidden_size), out: (seq_len, hidden_size)
    ASSERT(index->ndim() == 1, "index must be 1D");
    ASSERT(weight->ndim() == 2, "weight must be 2D");
    ASSERT(out->ndim() == 2, "out must be 2D");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64");
    ASSERT(out->dtype() == weight->dtype(), "out and weight must have same dtype");

    size_t seq_len = index->shape()[0];
    size_t hidden_size = weight->shape()[1];
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == hidden_size,
           "out shape mismatch");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                             out->dtype(), seq_len, hidden_size);
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
