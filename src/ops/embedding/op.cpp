#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/embedding.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "Index tensor must be of type INT64."
    );

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape().back(),
            out->dtype()
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
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
