#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
#endif
}

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    auto device = core::context().runtime().deviceType();

    llaisys::core::context().setDevice(device, out->deviceId());

    if (device == LLAISYS_DEVICE_CPU) {

        size_t M = in->shape()[0];
        size_t K = in->shape()[1];
   
        size_t N = weight->shape()[0];
        
        const std::byte* bias_ptr = nullptr;
        if (bias && bias->numel() > 0) {
            bias_ptr = bias->data();
        }

        cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                    in->dtype(), M, K, N);

    }
    #ifdef ENABLE_NVIDIA_API
    else if (device == LLAISYS_DEVICE_NVIDIA) {
        llaisys::ops::nvidia::linear(out, in, weight, bias);
    }
#endif 
    else {
        throw std::runtime_error("Linear: Unsupported device type");
    }
}

} // namespace llaisys::ops