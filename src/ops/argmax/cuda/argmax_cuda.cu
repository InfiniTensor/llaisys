#include "argmax_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::ops::cuda {

__global__ void argmax_kernel(int64_t *mi, void *mv, const void *v, llaisysDataType_t type, size_t numel) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    size_t best_idx = 0;
    float best_val = load_as_float(v, 0, type);
    for (size_t i = 1; i < numel; ++i) {
        float cur = load_as_float(v, i, type);
        if (cur > best_val) {
            best_val = cur;
            best_idx = i;
        }
    }

    mi[0] = static_cast<int64_t>(best_idx);
    store_from_float(mv, 0, best_val, type);
}

void argmax(std::byte *mi, std::byte *mv, const std::byte *v, llaisysDataType_t type, size_t numel) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "argmax cuda only supports F32/F16/BF16");
    argmax_kernel<<<1, 1>>>(reinterpret_cast<int64_t *>(mi), mv, v, type, numel);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "argmax cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
