#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/rope.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "pos_ids must be 1D");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids must be of type int64");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "out and in must have the same dtype");

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(), in->data(), pos_ids->data(),
            theta, seqlen, nhead, d, in->dtype()
        );
    }
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
