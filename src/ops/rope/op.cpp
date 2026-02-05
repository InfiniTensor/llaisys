#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(in, out, pos_ids);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(in->shape(), out->shape());
    CHECK_SAME_SHAPE(in->shape()[0], pos_ids->shape()[0]);
    // 假设 pos_ids 是 INT64
    CHECK_SAME_DTYPE(in->dtype(), out->dtype());
    CHECK_SAME_DTYPE(LLAISYS_DTYPE_I64, pos_ids->dtype());
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t d = in->shape()[2];

    ASSERT(d%2==0, "Rope: tensor in is of size [seqlen, nhead, d], but d is not divided by 2");

    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(in->data(), out->data(), pos_ids->data(), in->dtype(), seqlen, nhead, d, theta);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(in->data(), out->data(), pos_ids->data(), in->dtype(), seqlen, nhead, d, theta);
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
