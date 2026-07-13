#include "op.hpp"

#include "cpu/embedding_cpu.hpp"
#include "llaisys.h"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {

namespace {

template <typename T>
void parallelEmbeddingCpuImpl(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_indices, size_t hidden_size,
                              size_t vocab_start, size_t vocab_end) {
    auto *out_ptr = reinterpret_cast<T *>(out);
    const auto *index_ptr = reinterpret_cast<const int64_t *>(index);
    const auto *weight_ptr = reinterpret_cast<const T *>(weight);
    for (size_t row = 0; row < num_indices; ++row) {
        const int64_t vocab_row = index_ptr[row];
        for (size_t col = 0; col < hidden_size; ++col) {
            if (vocab_row >= static_cast<int64_t>(vocab_start) && vocab_row < static_cast<int64_t>(vocab_end)) {
                const size_t local_row = static_cast<size_t>(vocab_row) - vocab_start;
                out_ptr[row * hidden_size + col] = weight_ptr[local_row * hidden_size + col];
            } else {
                out_ptr[row * hidden_size + col] = llaisys::utils::cast<T>(0.0f);
            }
        }
    }
}

void parallelEmbeddingCpu(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices,
                          size_t hidden_size, size_t vocab_start, size_t vocab_end) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return parallelEmbeddingCpuImpl<float>(out, index, weight, num_indices, hidden_size, vocab_start, vocab_end);
    case LLAISYS_DTYPE_F16:
        return parallelEmbeddingCpuImpl<llaisys::fp16_t>(out, index, weight, num_indices, hidden_size, vocab_start, vocab_end);
    case LLAISYS_DTYPE_BF16:
        return parallelEmbeddingCpuImpl<llaisys::bf16_t>(out, index, weight, num_indices, hidden_size, vocab_start, vocab_end);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->shape()[1] == weight->shape()[1], "");
    ASSERT(out->shape()[0] == index->shape()[0], "");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], weight->shape()[1],
                                 llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void parallelEmbedding(tensor_t out, tensor_t index, tensor_t weight_local, size_t vocab_start, size_t vocab_end) {
    CHECK_SAME_DEVICE(out, index, weight_local);
    CHECK_SAME_DTYPE(out->dtype(), weight_local->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "ParallelEmbedding: index must be int64.");
    ASSERT(out->ndim() == 2, "ParallelEmbedding: out must be [num_tokens, hidden_size].");
    ASSERT(index->ndim() == 1, "ParallelEmbedding: index must be [num_tokens].");
    ASSERT(weight_local->ndim() == 2, "ParallelEmbedding: weight must be [local_vocab, hidden_size].");
    ASSERT(out->shape()[0] == index->shape()[0], "ParallelEmbedding: num_tokens mismatch.");
    ASSERT(out->shape()[1] == weight_local->shape()[1], "ParallelEmbedding: hidden size mismatch.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight_local->isContiguous(),
           "ParallelEmbedding: all tensors must be contiguous.");
    ASSERT(vocab_start <= vocab_end, "ParallelEmbedding: invalid vocab shard range.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return parallelEmbeddingCpu(out->data(), index->data(), weight_local->data(), out->dtype(), out->shape()[0], out->shape()[1], vocab_start, vocab_end);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return parallelEmbeddingCpu(out->data(), index->data(), weight_local->data(), out->dtype(), out->shape()[0], out->shape()[1], vocab_start, vocab_end);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::parallelEmbedding(out->data(), index->data(), weight_local->data(), out->dtype(), out->shape()[0], out->shape()[1],
                                         vocab_start, vocab_end, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
