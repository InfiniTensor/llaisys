#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include <vector>
#ifdef ENABLE_NVIDIA_API
#include "cuda/embedding_cuda.cuh"
#endif
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);  
      
    ASSERT(index->shape().size() == 1, "Embedding: index must be 1D tensor");  
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be Int64 type");  
      
 
    ASSERT(weight->shape().size() == 2, "Embedding: weight must be 2D tensor");  
      

    ASSERT(out->shape().size() == 2, "Embedding: out must be 2D tensor");  
      

    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: out and index first dimension must match");  
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out and weight second dimension must match");  

    size_t vocab_size = weight->shape()[0];
    std::vector<int64_t> host_index(index->numel());
    if (index->deviceType() == LLAISYS_DEVICE_CPU) {
        auto index_data = reinterpret_cast<const int64_t *>(index->data());
        for (size_t i = 0; i < index->numel(); i++) {
            host_index[i] = index_data[i];
        }
    } else {
        llaisys::core::context().setDevice(index->deviceType(), index->deviceId());
        llaisys::core::context().runtime().api()->memcpy_sync(
            host_index.data(),
            index->data(),
            index->numel() * sizeof(int64_t),
            LLAISYS_MEMCPY_D2H);
    }

    for (size_t i = 0; i < host_index.size(); i++) {
        ASSERT(host_index[i] >= 0 && static_cast<size_t>(host_index[i]) < vocab_size,
               "Embedding: index out of bounds");
    }
  
    // always support cpu calculation  
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {  
        return cpu::embedding(out->data(), index->data(), weight->data(),   
                             out->dtype(), index->numel(), weight->shape()[1]);  
    }  
  
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());  
  
    switch (out->deviceType()) {  
    case LLAISYS_DEVICE_CPU:  
        return cpu::embedding(out->data(), index->data(), weight->data(),   
                             out->dtype(), index->numel(), weight->shape()[1]);  
#ifdef ENABLE_NVIDIA_API  
    case LLAISYS_DEVICE_NVIDIA:  
    return cuda::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(), weight->shape()[1]);  
        return;  
#endif  
    default:  
        EXCEPTION_UNSUPPORTED_DEVICE;  
    }  
}  

} // namespace llaisys::ops
