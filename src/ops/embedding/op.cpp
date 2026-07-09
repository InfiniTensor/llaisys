#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be of dtype int64.");
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "Embedding: out.shape[0] must equal index.shape[0].");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding:   out.shape[1] must equal weight.shape[1].");
    CHECK_ARGUMENT(weight->shape()[0] > 0, "Embedding: weight.shape[0] must be greater than 0.");
    CHECK_ARGUMENT(index->numel() > 0, "Embedding: index must not be empty.");
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(out->deviceType(), out->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t out_bytes = out->numel() * out->elementSize();
        const size_t idx_bytes = index->numel() * index->elementSize();
        const size_t w_bytes = weight->numel() * weight->elementSize();

        std::vector<std::byte> h_out(out_bytes);
        std::vector<std::byte> h_index(idx_bytes);
        std::vector<std::byte> h_weight(w_bytes);

        api->memcpy_sync(h_index.data(), index->data(), idx_bytes, LLAISYS_MEMCPY_D2H);
        api->memcpy_sync(h_weight.data(), weight->data(), w_bytes, LLAISYS_MEMCPY_D2H);

        auto index_ptr = reinterpret_cast<const int64_t *>(h_index.data());
        auto out_ptr = h_out.data();
        auto weight_ptr = h_weight.data();
        size_t embed_dim = weight->shape()[1];
        size_t dtype_size = out->elementSize();
        for (size_t i = 0; i < index->numel(); ++i) {
            int64_t idx = index_ptr[i];
            CHECK_ARGUMENT(idx >= 0 && static_cast<size_t>(idx) < weight->shape()[0],
                           "Embedding: index value out of range.");
            std::memcpy(out_ptr + i * embed_dim * dtype_size,
                        weight_ptr + static_cast<size_t>(idx) * embed_dim * dtype_size,
                        embed_dim * dtype_size);
        }

        api->memcpy_sync(out->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
        return;
    }

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // perform embedding lookup
    // get data pointers
    auto index_ptr = reinterpret_cast<const int64_t *>(index->data());
    auto out_ptr = out->data();
    auto weight_ptr = weight->data();
    size_t embed_dim = weight->shape()[1];
    size_t dtype_size = out->elementSize();
    for (size_t i = 0; i < index->numel(); ++i) {
        int64_t idx = index_ptr[i];
        CHECK_ARGUMENT(idx >= 0 && static_cast<size_t>(idx) < weight->shape()[0],
                       "Embedding: index value out of range.");
        // copy embedding vector
        // memcpy(目标地址, 源地址, 复制的字节数)
        // out_ptr + i * embed_dim * dtype_size: 目标地址，指向out张量中第i个embedding向量的起始位置
        // weight_ptr + static_cast<size_t>(idx) * embed_dim * dtype_size: 源地址，指向weight张量中索引为idx的embedding向量的起始位置
        // embed_dim * dtype_size: 复制的字节数，即一个embedding向量的大小
        std::memcpy(out_ptr + i * embed_dim * dtype_size,
                    weight_ptr + static_cast<size_t>(idx) * embed_dim * dtype_size,
                    embed_dim * dtype_size);
    }
}
} // namespace llaisys::ops
