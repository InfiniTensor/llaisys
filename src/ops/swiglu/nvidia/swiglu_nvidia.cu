#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    
    float gate_val = to_float(gate[idx]);
    float up_val = to_float(up[idx]);
    float exp_gate = ::expf(-gate_val);
    float out_val = up_val * gate_val / (1 + exp_gate);
    out[idx] = from_float<T>(out_val);
}

} // namespace

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel) {
    constexpr int block_size = 256;
    const int grid_size = CEIL(numel, block_size);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<grid_size, block_size>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<half><<<grid_size, block_size>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(gate), reinterpret_cast<const half *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(gate), reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
