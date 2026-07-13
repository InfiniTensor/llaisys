#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_cu.cuh"
#endif

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE(): pos_ids must be int64.");

    size_t seqlen = in->shape()[0];
    size_t num_head = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(pos_ids->numel() == seqlen, "RoPE(): pos_ids length must match input sequence length.");

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        ops::cpu::rope(out->data(), in->data(), pos_ids->data(), seqlen, num_head, head_dim, theta, in->dtype());
#ifdef ENABLE_NVIDIA_API
    } else if (in->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        ops::nvidia::rope(out->data(), in->data(), pos_ids->data(), seqlen, num_head, head_dim, theta, in->dtype());
#endif
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
