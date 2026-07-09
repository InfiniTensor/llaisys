#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    ASSERT(out->isContiguous() && in->isContiguous() 
    && weight->isContiguous(), "out, in, weight must be contiguous tensors");
    
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && 
    weight->ndim() == 2, "out, in, weight must be 2D tensors");
    ASSERT(in->shape()[1] == weight->shape()[1],"in.shape[1] must be equal to weight.shape[1]");
    ASSERT(out->shape()[1] == weight->shape()[0], "out.shape[1] must be equal to weight.shape[0]");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if(bias)
    {
        CHECK_SAME_DEVICE(out, bias, weight);
        ASSERT(bias->ndim() == 1, "bias must be 1D tensor");
        ASSERT(bias->isContiguous(), "bias must be contiguous tensor");
        ASSERT(bias->shape()[0] == out->shape()[1], "bias.shape[0] must be equal to out.shape[1]");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    size_t batch = out->shape()[0];
    size_t in_dim = in->shape()[1];
    size_t out_dim = out->shape()[1];

    const std::byte *bias_data = bias ? bias->data():nullptr;

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), 
        weight->data(), bias_data, bias->dtype(), batch, in_dim, out_dim);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), 
        weight->data(), bias_data, bias->dtype(), batch, in_dim, out_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), batch, in_dim, out_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
