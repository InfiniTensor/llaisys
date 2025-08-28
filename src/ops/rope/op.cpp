#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // --- device checks ---
    CHECK_SAME_DEVICE(out, in);
    
    CHECK_SAME_DEVICE(out, pos_ids);

    // --- shape checks ---
    ASSERT(in->ndim() == 3, "rope: in must be 3D [seqlen, nhead, d]");
    ASSERT(out->ndim() == 3, "rope: out must be 3D [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D [seqlen]");

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t d = in->shape()[2];

    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == d, "rope: out shape must equal in shape");
    ASSERT(pos_ids->shape()[0] == seqlen, "rope: pos_ids length must equal seqlen");
    ASSERT(d % 2 == 0, "rope: last dimension d must be even");

    // --- dtype & contiguity ---
    ASSERT(out->dtype() == in->dtype(), "rope: out/in dtypes must match");
    ASSERT(out->isContiguous() && in->isContiguous(), "rope: out/in must be contiguous");
    ASSERT(pos_ids->isContiguous(), "rope: pos_ids must be contiguous");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");

    // --- dispatch (CPU first) ---
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         seqlen, nhead, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         seqlen, nhead, d, theta);
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
