#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
template <typename T>
void embedding_cpu_(T *out, const int64_t *index, const T *weight, 
                    size_t seq_len, size_t embedding_dim) {
    for (size_t i = 0; i < seq_len; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * embedding_dim;
        T *dst = out + i * embedding_dim;
        
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // Copy element by element for half precision types
            for (size_t j = 0; j < embedding_dim; j++) {
                dst[j] = src[j];
            }
        } else {
            // Use memcpy for standard types
            std::memcpy(dst, src, embedding_dim * sizeof(T));
        }
    }
}

void embedding_cpu(std::byte *out, const std::byte *index, const std::byte *weight,
                   llaisysDataType_t dtype, size_t seq_len, size_t embedding_dim) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_cpu_(reinterpret_cast<float *>(out), idx_ptr,
                             reinterpret_cast<const float *>(weight), seq_len, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_cpu_(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr,
                             reinterpret_cast<const llaisys::fp16_t *>(weight), seq_len, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_cpu_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,
                             reinterpret_cast<const llaisys::bf16_t *>(weight), seq_len, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Validate inputs
    CHECK_ARGUMENT(index->ndim() == 1, "embedding: index must be 1-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding: weight must be 2-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "embedding: out must be 2-D tensor");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be i64 type");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "embedding: out and weight must have same dtype");
    CHECK_SAME_DEVICE(out, index, weight);
    
    size_t seq_len = index->shape()[0];
    size_t embedding_dim = weight->shape()[1];
    
    CHECK_ARGUMENT(out->shape()[0] == seq_len, "embedding: out shape[0] must match index length");
    CHECK_ARGUMENT(out->shape()[1] == embedding_dim, "embedding: out shape[1] must match weight shape[1]");
    
    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return embedding_cpu(out->data(), index->data(), weight->data(),
                            weight->dtype(), seq_len, embedding_dim);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return embedding_cpu(out->data(), index->data(), weight->data(),
                            weight->dtype(), seq_len, embedding_dim);
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
