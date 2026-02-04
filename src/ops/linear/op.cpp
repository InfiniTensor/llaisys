#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr) {
    // C[:,M,N] = A[:,M,K] * B[:,K,N]
    // out[:,M,N] = in[:,M,K] * weight[:,N,K]^T + bias[:,M,N]
    const size_t M = out->shape()[out->shape().size() - 2];
    const size_t N = out->shape()[out->shape().size() - 1];
    const size_t K = in->shape()[in->shape().size() - 1];

    CHECK_SAME_DEVICE(out, in, weight);

    CHECK_SAME_SHAPE(M, in->shape()[in->shape().size() - 2]);
    CHECK_SAME_SHAPE(N, weight->shape()[weight->shape().size() - 2]);
    CHECK_SAME_SHAPE(K, weight->shape()[weight->shape().size() - 1]);
    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Add: all tensors must be contiguous.");

    // bias 形状应该是 (N,), 不是(M, N,)
    // if (bias != nullptr) {
    //     CHECK_SAME_DEVICE(out, bias);
    //     CHECK_SAME_SHAPE(M, bias->shape()[bias->shape().size() - 2]);
    //     CHECK_SAME_SHAPE(N, bias->shape()[bias->shape().size() - 1]);
    //     ASSERT(bias->isContiguous(), "Add: all tensors must be contiguous.");
    // }

    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_SHAPE(N, bias->shape()[bias->shape().size() - 1]);
        ASSERT(bias->isContiguous(), "Add: all tensors must be contiguous.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), M, N, K, weight->dtype());
        }
        return cpu::linear(out->data(), in->data(), weight->data(), nullptr, M, N, K, weight->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), M, N, K, weight->dtype());
        }
        return cpu::linear(out->data(), in->data(), weight->data(), nullptr, M, N, K, weight->dtype());
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
