#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1, "Output tensors must have a single element");
    ASSERT(vals->ndim() == 1, "Input tensor must be 1-dimensional");
    ASSERT(vals->isContiguous(), "Input tensor must be contiguous");
    ASSERT(max_idx->isContiguous(), "Output tensor max_idx must be contiguous");

    if(vals->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
        return;
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
