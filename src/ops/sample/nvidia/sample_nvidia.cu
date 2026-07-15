#include "sample_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"
#include "../cpu/sample_cpu.hpp"

#include <vector>

namespace llaisys::ops::nvidia {

int64_t sample(const std::byte *logits, llaisysDataType_t dtype, size_t numel, int top_k, float top_p, float temperature) {
    std::vector<std::byte> host_logits(numel * utils::dsize(dtype));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        host_logits.data(),
        logits,
        host_logits.size(),
        cudaMemcpyDeviceToHost,
        current_stream()));
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
    return cpu::sample(host_logits.data(), dtype, numel, top_k, top_p, temperature);
}

} // namespace llaisys::ops::nvidia
