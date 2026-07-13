#include "../op.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../core/context/context.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template<typename T>
__global__ void rope_kernel(
    T* output,
    const T* input,
    const int64_t* pos_ids,
    int num_heads,
    int head_dim,
    float theta)
{
    int token_idx = blockIdx.x / num_heads;
    int half_dim = head_dim / 2;
    int64_t m = pos_ids[token_idx];

    const T* in_ptr = input + blockIdx.x * head_dim;
    T* out_ptr = output + blockIdx.x * head_dim;

    
    for (int tid = threadIdx.x; tid < half_dim; tid += blockDim.x) {
        
        double freq_expon = static_cast<double>(2 * tid) / static_cast<double>(head_dim);
        double inv_freq = pow(theta, -freq_expon);
        double freq = static_cast<double>(m) * inv_freq;

        float cos_val = static_cast<float>(cos(freq));
        float sin_val = static_cast<float>(sinf(freq));

        float val_a = static_cast<float>(in_ptr[tid]);
        float val_b = static_cast<float>(in_ptr[tid + half_dim]);

        float out_a = val_a * cos_val - val_b * sin_val;
        float out_b = val_b * cos_val + val_a * sin_val;

        out_ptr[tid] = static_cast<T>(out_a);
        out_ptr[tid + half_dim] = static_cast<T>(out_b);
    }
}

template<typename T>
void launch_rope_kernel(
    tensor_t output,
    tensor_t input,
    tensor_t pos_ids,
    float theta)
{
    int head_dim = input->shape().back();
    int num_heads = input->shape()[input->shape().size() - 2];
    int num_tokens = input->numel() / (num_heads * head_dim);
    int half_dim = head_dim / 2;

    int block_size = half_dim < 1024 ? half_dim : 1024;

    dim3 grid(num_tokens * num_heads);
    dim3 block(block_size);

    T* d_out = reinterpret_cast<T*>(output->data());
    const T* d_in = reinterpret_cast<const T*>(input->data());
    const int64_t* d_pos_ids = reinterpret_cast<const int64_t*>(pos_ids->data());

    rope_kernel<T><<<grid, block>>>(d_out, d_in, d_pos_ids, num_heads, head_dim, theta);
}

void rope(tensor_t output, tensor_t input, tensor_t pos_ids, float theta) {
    auto dtype = input->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_rope_kernel<float>(output, input, pos_ids, theta);
            break;
        case LLAISYS_DTYPE_F16:
            launch_rope_kernel<__half>(output, input, pos_ids, theta);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_rope_kernel<__nv_bfloat16>(output, input, pos_ids, theta);
            break;
        default:
            abort();
    }
}

} // namespace llaisys::ops::nvidia