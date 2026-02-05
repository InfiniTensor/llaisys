#include "op.hpp"
#include "cpu/swiglu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
   if (out->deviceType() == LLAISYS_DEVICE_CPU) {
      
   }
}
} // namespace llaisys::ops
