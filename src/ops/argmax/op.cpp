#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/argmax_cuda.cuh"
#endif
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
     CHECK_SAME_DEVICE(max_idx, max_val, vals);  
      
    // 验证vals是1D张量  
    ASSERT(vals->shape().size() == 1, "Argmax: vals must be a 1D tensor");  

    //验证vals连续性
    ASSERT(vals->isContiguous(),"Argmax: vals must be contiguous.");
      
    // 验证max_idx是包含单个元素的1D张量  
    ASSERT(max_idx->shape().size() == 1 && max_idx->shape()[0] == 1,   
           "Argmax: max_idx must be a 1D tensor with single element");  
      
    // 验证max_val是包含单个元素的1D张量  
    ASSERT(max_val->shape().size() == 1 && max_val->shape()[0] == 1,   
           "Argmax: max_val must be a 1D tensor with single element");  
      
    // 验证max_idx是Int64类型  
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,   
           "Argmax: max_idx must be Int64 type");  
      
    // 验证max_val类型与vals相同  
    ASSERT(max_val->dtype() == vals->dtype(),   
           "Argmax: max_val must have same type as vals");  
  
    // always support cpu calculation  
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {  
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),   
                          vals->dtype(), vals->numel());  
    }  
  
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());  
  
    switch (vals->deviceType()) {  
    case LLAISYS_DEVICE_CPU:  
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),   
                          vals->dtype(), vals->numel());  
#ifdef ENABLE_NVIDIA_API  
    case LLAISYS_DEVICE_NVIDIA:  
    return cuda::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());  
        return;  
#endif  
    default:  
        EXCEPTION_UNSUPPORTED_DEVICE;  
    }  
}
} // namespace llaisys::ops
