#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "./cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "./metax/linear_metax.hpp"
#endif
#include "llaisys.h"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
  // 1. 参数校验
  CHECK_SAME_DEVICE(out, in, weight);
  if (bias != nullptr) {
    CHECK_SAME_DEVICE(out, bias);
    CHECK_ARGUMENT(bias->ndim() == 1, "bias must be a 1D tensor");
    CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1],
                   "N dim of bias and out must be the same");
    CHECK_ARGUMENT(out->dtype() == bias->dtype(),
                   "bias must have the same data type as out");
  }
  CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
  CHECK_ARGUMENT(in->ndim() == 2, "in must be a 2D tensor");
  CHECK_ARGUMENT(weight->ndim() == 2, "weight must be a 2D tensor");
  // X: [M, K], W: [N, K], b: [N], Y: [M, N]
  CHECK_ARGUMENT(out->shape()[0] == in->shape()[0],
                 "M dim of out and in must be the same");
  CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0],
                 "N dim of out and weight must be the same");
  CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1],
                 "K dim of inin and weight must be the same");
  CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
                 "out, in and weight must have the same data type");
  if (bias != nullptr) {
    ASSERT(out->isContiguous() && in->isContiguous() &&
               weight->isContiguous() && bias->isContiguous(),
           "out, in, weight and bias must be contiguous");
  } else {
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "out, in and weight must be contiguous");
  }

  // 2. 设置上下文
  llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

  switch (out->deviceType()) {
  case LLAISYS_DEVICE_CPU:
    return cpu::linear(out->data(), in->data(), weight->data(),
                       (bias != nullptr) ? bias->data() : nullptr, out->dtype(),
                       out->shape()[0], out->shape()[1], in->shape()[1]);
#ifdef ENABLE_NVIDIA_API
  case LLAISYS_DEVICE_NVIDIA:
    return nvidia::linear(out->data(), in->data(), weight->data(),
                          (bias != nullptr) ? bias->data() : nullptr,
                          out->dtype(), out->shape()[0], out->shape()[1],
                          in->shape()[1]);
#endif
#ifdef ENABLE_METAX_API
  case LLAISYS_DEVICE_METAX:
    return metax::linear(out->data(), in->data(), weight->data(),
                         (bias != nullptr) ? bias->data() : nullptr,
                         out->dtype(), out->shape()[0], out->shape()[1],
                         in->shape()[1]);
#endif
  default:
    EXCEPTION_UNSUPPORTED_DEVICE;
  }
}
} // namespace llaisys::ops
