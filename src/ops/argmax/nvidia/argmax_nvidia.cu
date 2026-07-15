#include "argmax_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"
#include "../cpu/argmax_cpu.hpp"

#include <vector>

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel) {
    std::vector<std::byte> host_vals(numel * utils::dsize(dtype));
    std::vector<std::byte> host_idx(sizeof(size_t));
    std::vector<std::byte> host_max_val(utils::dsize(dtype));

    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        host_vals.data(),
        vals,
        host_vals.size(),
        cudaMemcpyDeviceToHost,
        current_stream()));
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(current_stream()));

    cpu::argmax(host_idx.data(), host_max_val.data(), host_vals.data(), dtype, numel);

    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        max_idx,
        host_idx.data(),
        host_idx.size(),
        cudaMemcpyHostToDevice,
        current_stream()));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        max_val,
        host_max_val.data(),
        host_max_val.size(),
        cudaMemcpyHostToDevice,
        current_stream()));
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
}

} // namespace llaisys::ops::nvidia
