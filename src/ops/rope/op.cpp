#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids must be int64");
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "out/in must be 3D");
    ASSERT(out->shape() == in->shape(), "out & in shapes differ");
    ASSERT(pos_ids->ndim() == 1, "pos_ids must be 1D");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "pos_ids length mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Tensors must be contiguous");

    size_t head_dim = in->shape()[2];
    ASSERT((head_dim % 2) == 0, "head_dim must be even");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rope(out->data(),
                  in->data(),
                  pos_ids->data(),
                  out->dtype(),
                  in->shape()[0],  // seqlen
                  in->shape()[1],  // nhead
                  head_dim,
                  theta);
        return;
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
