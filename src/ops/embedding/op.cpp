#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Index tensor must be of type I16M");
    ASSERT(index->ndim() == 1, "Index tensor must be 1D");
    ASSERT(weight->ndim() == 2, "Weight tensor must be 2D");
    ASSERT(out->ndim() == 2, "Output tensor must be 2D");
    ASSERT(out->shape()[0] == index->shape()[0],
           "Output tensor's first dimension must match index tensor's size");
    ASSERT(out->shape()[1] == weight->shape()[1],
           "Output tensor's second dimension must match weight tensor's second dimension");
    ASSERT(index->isContiguous(), "Index tensor must be contiguous");

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(reinterpret_cast<std::uint8_t*>(out->data()),
                            reinterpret_cast<const int64_t*>(index->data()),
                            reinterpret_cast<std::uint8_t*>(weight->data()),
                            out->dtype(),
                            index->shape()[0],
                            weight->shape()[1], 
                            weight->shape()[0]);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;

}
} // namespace llaisys::ops
