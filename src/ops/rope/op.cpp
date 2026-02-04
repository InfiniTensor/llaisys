#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template <typename T>
void rope_cpu_(T *out, const T *in, const int64_t *pos_ids, float theta,
               size_t seqlen, size_t nhead, size_t d) {
    // RoPE: Apply rotary position encoding
    // Shape: [seqlen, nhead, d]
    // pos_ids: [seqlen]
    
    size_t half_d = d / 2;
    
    for (size_t s = 0; s < seqlen; s++) {
        int64_t pos = pos_ids[s];  // Position id for this token
        
        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_d; j++) {
                // Calculate frequency: phi = pos / theta^(2j/d)
                float freq = static_cast<float>(pos) / std::pow(theta, (2.0f * j) / d);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                
                // Get indices for a[j] and b[j]
                size_t idx_a = s * nhead * d + h * d + j;           // a_j in first half
                size_t idx_b = s * nhead * d + h * d + half_d + j;  // b_j in second half
                
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    // For half precision, compute in float
                    float a_val = llaisys::utils::cast<float>(in[idx_a]);
                    float b_val = llaisys::utils::cast<float>(in[idx_b]);
                    
                    // Apply rotation: a' = a*cos - b*sin, b' = b*cos + a*sin
                    float a_out = a_val * cos_val - b_val * sin_val;
                    float b_out = b_val * cos_val + a_val * sin_val;
                    
                    out[idx_a] = llaisys::utils::cast<T>(a_out);
                    out[idx_b] = llaisys::utils::cast<T>(b_out);
                } else {
                    // For standard types
                    float a_val = static_cast<float>(in[idx_a]);
                    float b_val = static_cast<float>(in[idx_b]);
                    
                    // Apply rotation
                    float a_out = a_val * cos_val - b_val * sin_val;
                    float b_out = b_val * cos_val + a_val * sin_val;
                    
                    out[idx_a] = static_cast<T>(a_out);
                    out[idx_b] = static_cast<T>(b_out);
                }
            }
        }
    }
}

void rope_cpu(std::byte *out, const std::byte *in, const std::byte *pos_ids,
              float theta, llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d) {
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_cpu_(reinterpret_cast<float *>(out),
                        reinterpret_cast<const float *>(in),
                        pos_ids_ptr, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_cpu_(reinterpret_cast<llaisys::fp16_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        pos_ids_ptr, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_cpu_(reinterpret_cast<llaisys::bf16_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        pos_ids_ptr, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // Validate inputs
    CHECK_ARGUMENT(in->ndim() == 3, "rope: in must be 3-D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(out->ndim() == 3, "rope: out must be 3-D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "rope: pos_ids must be 1-D tensor [seqlen]");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64 type");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "rope: out and in must have same dtype");
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    
    CHECK_ARGUMENT(d % 2 == 0, "rope: head dimension d must be even");
    CHECK_ARGUMENT(pos_ids->shape()[0] == seqlen, "rope: pos_ids length must match seqlen");
    CHECK_ARGUMENT(out->shape()[0] == seqlen, "rope: out shape[0] must match in shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == nhead, "rope: out shape[1] must match in shape[1]");
    CHECK_ARGUMENT(out->shape()[2] == d, "rope: out shape[2] must match in shape[2]");
    
    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return rope_cpu(out->data(), in->data(), pos_ids->data(),
                       theta, out->dtype(), seqlen, nhead, d);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return rope_cpu(out->data(), in->data(), pos_ids->data(),
                       theta, out->dtype(), seqlen, nhead, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
