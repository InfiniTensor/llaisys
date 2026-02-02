#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->ndim() == 2 && in->ndim() == 2, "Tensor must be 2D");
    ASSERT(weight->ndim() == 1, "Weight tensor must be 1D");
    ASSERT(out->shape() == in->shape(), "Output and input shapes must match");
    size_t batch = in->shape()[0];
    size_t dim = in->shape()[1];
    ASSERT(weight->shape()[0] == dim, "Weight shape must match input feature dimension");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "All tensors must be contiguous");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            batch,
            dim,
            eps
        );
        return;
    } 
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
