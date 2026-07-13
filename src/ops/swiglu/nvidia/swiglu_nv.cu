#include "../op.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template<typename T>
__global__ void swiglu_kernel(T* out, const T* gate, const T* up, int numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    float g = static_cast<float>(gate[i]);
    float u = static_cast<float>(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    out[i] = static_cast<T>(u * silu_g);
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    int numel = static_cast<int>(out->numel());
    int block = 256;
    int grid = (numel + block - 1) / block;

    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            swiglu_kernel<float><<<grid, block>>>(
                reinterpret_cast<float*>(out->data()),
                reinterpret_cast<const float*>(gate->data()),
                reinterpret_cast<const float*>(up->data()), numel);
            break;
        case LLAISYS_DTYPE_F16:
            swiglu_kernel<__half><<<grid, block>>>(
                reinterpret_cast<__half*>(out->data()),
                reinterpret_cast<const __half*>(gate->data()),
                reinterpret_cast<const __half*>(up->data()), numel);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<__nv_bfloat16*>(out->data()),
                reinterpret_cast<const __nv_bfloat16*>(gate->data()),
                reinterpret_cast<const __nv_bfloat16*>(up->data()), numel);
            break;
        default:
            abort();
    }
}

} // namespace llaisys::ops::nvidia
