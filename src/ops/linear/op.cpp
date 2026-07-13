#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_cu.cuh"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias)
        CHECK_SAME_DEVICE(out, bias);

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias)
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());

    auto N = out->shape()[0];
    auto K = out->shape()[1];
    auto M = in->shape()[1];

    ASSERT(in->shape()[0] == N && in->shape()[1] == M, "linear(): Input shape mismatch for linear op");
    ASSERT(weight->shape()[0] == K && weight->shape()[1] == M, "linear(): Weight shape mismatch for linear op");

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()),
           "linear(): All tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        llaisys::ops::cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, N, M, K,
                                  out->dtype());
#ifdef ENABLE_NVIDIA_API
    } else if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::ops::nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, N, M, K,
                                     out->dtype());
#endif
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
