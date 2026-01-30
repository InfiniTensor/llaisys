#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 1. 基础校验
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }

    // 2. 形状校验
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: input must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "Linear: output must be 2D");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    CHECK_ARGUMENT(weight->shape()[1] == K, "Linear: weight shape[1] must match in shape[1]");
    CHECK_ARGUMENT(out->shape()[0] == M, "Linear: out shape[0] must match in shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == N, "Linear: out shape[1] mismatch");

    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D");
        CHECK_ARGUMENT(bias->shape()[0] == N, "Linear: bias shape[0] mismatch");
    }

    // 3. 内存连续性校验
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: tensors must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    }

    // 4. CPU 分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                    out->dtype(), M, N, K);
        return;
    }

    // 5. GPU 分支待实现
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
