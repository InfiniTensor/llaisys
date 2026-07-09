#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_cuda.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, index, weight);

    // 检查 index 必须是 int64 类型
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be of type int64");

    // 检查 out 和 weight 的数据类型一致
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // 检查维度
    CHECK_ARGUMENT(index->ndim() == 1, "embedding: index must be 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding: weight must be 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "embedding: out must be 2D tensor");

    // 检查形状兼容性
    size_t num_indices = index->shape()[0];
    size_t embedding_dim = weight->shape()[1];

    CHECK_ARGUMENT(out->shape()[0] == num_indices,
                   "embedding: out's first dimension must match index length");
    CHECK_ARGUMENT(out->shape()[1] == embedding_dim,
                   "embedding: out's second dimension must match weight's embedding dimension");

    // 检查所有张量都是连续的
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), num_indices, embedding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                 out->dtype(), num_indices, embedding_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
