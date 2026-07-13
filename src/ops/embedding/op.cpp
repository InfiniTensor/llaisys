#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void embedding(tensor_t output, tensor_t index, tensor_t weight);
#endif
}

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("Index tensor must be of type INT64.");
    }
    
    CHECK_SAME_DEVICE(out, index);
    CHECK_SAME_DEVICE(out, weight);

    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape().back();

    auto device = out->deviceType();

    llaisys::core::context().setDevice(device, out->deviceId());

    if (device == LLAISYS_DEVICE_CPU) {
        cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), num_indices, embedding_dim);
    } 
    #ifdef ENABLE_NVIDIA_API
    else if (device == LLAISYS_DEVICE_NVIDIA) {
        llaisys::ops::nvidia::embedding(out, index, weight);
    }
    #endif
    else {
        throw std::runtime_error("Embedding: Unsupported device type");
    }
}

} // namespace llaisys::ops