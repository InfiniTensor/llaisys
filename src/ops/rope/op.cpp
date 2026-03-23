#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/rope_nvidia.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    ASSERT(pos_ids->deviceType() == out->deviceType() && pos_ids->deviceId() == out->deviceId(),
           "ROPE: pos_ids must be on the same device.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "ROPE: pos_ids must be int64.");

    ASSERT(out->ndim() == 3 && in->ndim() == 3, "ROPE: out and in must be 3D [seqlen, nhead, dim].");
    ASSERT(pos_ids->ndim() == 1, "ROPE: pos_ids must be 1D [seqlen].");

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t dim = in->shape()[2];
    ASSERT(dim % 2 == 0, "ROPE: head dim must be even.");

    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == dim,
           "ROPE: output shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seqlen, "ROPE: pos_ids length must equal seqlen.");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "ROPE: tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, dim, theta);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, dim, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
