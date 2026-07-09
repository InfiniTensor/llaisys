#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    ASSERT(in->shape().size() == 2, "Linear: input tensor must be 2-D.");
    ASSERT(weight->shape().size() == 2, "Linear: weight tensor must be 2-D.");
    ASSERT(out->shape().size() == 2, "Linear: output tensor must be 2-D.");
    size_t dimi = in->shape()[0];
    size_t dimk = in->shape()[1];
    size_t dimj = weight->shape()[0];
    ASSERT(weight->shape()[1] == dimk, "Linear: weight tensor shape is invalid.");
    ASSERT(out->shape()[0] == dimi && out->shape()[1] == dimj, "Linear: output tensor shape is invalid.");
    if(bias != nullptr){
        ASSERT(bias->shape().size() == 1 && bias->shape()[0] == dimj, "Linear: bias tensor shape is invalid.");
    }

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), {dimi, dimk, dimj});
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), {dimi, dimk, dimj});
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
