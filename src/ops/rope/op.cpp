#include "op.hpp"
#include "cpu/rope_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>


namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void rope(tensor_t output, tensor_t input, tensor_t pos_ids, float theta);
#endif
}

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, pos_ids);

    if (in->ndim() < 2) {
        throw std::invalid_argument("RoPE: Input tensor must be at least 2D [..., head, dim]");
    }
    
    size_t head_dim = in->shape().back();
    size_t n_heads = in->shape()[in->ndim() - 2];
    
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE: Head dimension must be mathematically even for half-split logic");
    }

    if (pos_ids->dtype() != LLAISYS_DTYPE_I64 && pos_ids->dtype() != LLAISYS_DTYPE_I32) {
        throw std::invalid_argument("RoPE: pos_ids must be an integer type (int32 or int64)");
    }

    size_t num_tokens = in->numel() / (n_heads * head_dim);
    if (pos_ids->numel() != num_tokens) {
        throw std::invalid_argument("RoPE: pos_ids total elements must match the number of tokens");
    }

    auto device = core::context().runtime().deviceType();

    if (device == LLAISYS_DEVICE_CPU) {
        cpu::rope(out->data(), in->data(), pos_ids->data(),
                  in->dtype(), num_tokens, n_heads, head_dim, theta);
    } 
#ifdef ENABLE_NVIDIA_API
    else if (device == LLAISYS_DEVICE_NVIDIA) {
        nvidia::rope(out, in, pos_ids, theta);
    }
#endif
    else {
        throw std::runtime_error("RoPE: Unsupported device physical routing");
    }
}

} // namespace llaisys::ops