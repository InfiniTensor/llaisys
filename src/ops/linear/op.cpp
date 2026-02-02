#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    bool has_bias = (bias != nullptr);

    CHECK_SAME_DEVICE(out, in, weight);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (has_bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "Tensor must be 2D");
    ASSERT(out->shape()[0] == in->shape()[0], "Batch dimensions must match");
    ASSERT(out->shape()[1] == weight->shape()[0], "out_feature does not match weight rows count");
    ASSERT(in->shape()[1] == weight->shape()[1], "in_feature does not match weight columns count");

    if (has_bias){
        ASSERT(bias->ndim() == 1, "Bias must be 1D");
        ASSERT(bias->shape()[0] == weight->shape()[0], "Bias length must equal out_features");
        ASSERT(bias->isContiguous(), "Bias must be contiguous memory");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "out, in, weight must be contiguous memory");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            has_bias ? bias->data() : nullptr,
            out->dtype(),
            out->shape()[0],
            in->shape()[1],
            weight->shape()[0],
            has_bias
        );
        return;
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
