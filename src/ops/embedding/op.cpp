#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);

    // 抱歉时间来不及了, 这个实现很不elegant. 维度完全没有拓展性了.
    CHECK_SAME_SHAPE(out->shape()[0], index->shape()[0]);
    CHECK_SAME_SHAPE(out->shape()[1], weight->shape()[1]);

    // only support contiguous for now
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Add: tensors out and index must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), index->shape()[0],
                              weight->data(), weight->shape()[0], weight->shape()[1],
                              weight->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), index->shape()[0],
                              weight->data(), weight->shape()[0], weight->shape()[1],
                              weight->dtype());
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
