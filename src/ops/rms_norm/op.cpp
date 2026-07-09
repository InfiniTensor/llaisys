#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/rms_norm.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t feature_size = in->shape().back();
    size_t batch_size = in->numel() / feature_size;

    // Call CPU implementation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            nullptr,
            in->dtype(),
            batch_size,
            feature_size,
            eps);
    }
    //TODO: Add more device implementations here
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
