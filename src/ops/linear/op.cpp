#include "op.hpp"

#include "cpu/linear_cpu.hpp"
#include "../rearrange/op.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {

tensor_t gatherLastDim(const tensor_t &local) {
    ASSERT(local != nullptr, "gatherLastDim input must not be null.");
    ASSERT(local->isContiguous(), "gatherLastDim requires a contiguous tensor.");

    auto &ctx = llaisys::core::context();
    if (!ctx.distributedInitialized() || ctx.distributedWorldSize() == 1) {
        return local;
    }

    auto gathered = ctx.allGather(local);
    const size_t local_ndim = local->ndim();
    ASSERT(local_ndim >= 1, "gatherLastDim requires tensor rank >= 1.");

    std::vector<size_t> order;
    order.reserve(local_ndim + 1);
    for (size_t dim = 1; dim < local_ndim; ++dim) {
        order.push_back(dim);
    }
    order.push_back(0);
    order.push_back(local_ndim);

    auto permuted = gathered->permute(order);
    auto merged = llaisys::Tensor::create(permuted->shape(), permuted->dtype(), permuted->deviceType(), permuted->deviceId());
    ops::rearrange(merged, permuted);

    std::vector<size_t> final_shape = local->shape();
    final_shape.back() *= static_cast<size_t>(ctx.distributedWorldSize());
    return merged->view(final_shape);
}

void columnParallelLinear(tensor_t out_local, tensor_t in, tensor_t weight_local, tensor_t bias_local) {
    linear(out_local, in, weight_local, bias_local);
}

void rowParallelLinear(tensor_t out, tensor_t in_local, tensor_t weight_local) {
    auto &ctx = llaisys::core::context();
    linear(out, in_local, weight_local, nullptr);
    if (ctx.distributedInitialized() && ctx.distributedWorldSize() > 1) {
        ctx.allReduce(out);
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // TO_BE_IMPLEMENTED();
    if (bias == nullptr) {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
        ASSERT(bias->shape()[0] == weight->shape()[0], "Linear: bias.shape[0] must be equal to weight.shape[0].");
    }
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out.shape[0] must be equal to in.shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out.shape[1] must be equal to weight.shape[0].");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must be equal to weight.shape[1].");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        if (bias == nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, out->dtype(), out->shape(), in->shape(), weight->shape(), {});
        }
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), out->shape(), in->shape(), weight->shape(), bias->shape());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        if (bias == nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, out->dtype(), out->shape(), in->shape(), weight->shape(), {});
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), out->shape(), in->shape(), weight->shape(), bias->shape());
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(), out->dtype(),
                              out->shape()[0], out->shape()[1], in->shape()[1], llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
