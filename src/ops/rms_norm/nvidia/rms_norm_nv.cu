 #include "../op.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../core/context/context.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <type_traits>


namespace llaisys::ops::nvidia { 
//warp reduce
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val)
{
    for(int mask=16;mask>0;mask>>=1)
    {
        val+=__shfl_down_sync(0xffffffff,val,mask);
    }
    return val;
}
//block reduce
template<typename T>
__device__ __forceinline__ T blockReduceSum(T val)
{
    __shared__ T shared_val[32];
    __shared__ T final_val;

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if(lane == 0)
    {
        shared_val[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared_val[lane] : (T)(0.0f);
    
    if(wid == 0)
    {
        val = warpReduceSum(val);
        if(threadIdx.x == 0)
        {
            final_val = val;
        }
    }
    
    __syncthreads();
    
    return final_val;
}

template<typename T>
__global__ void rmsnorm_kernel(
    T* output,
    const T* input,
    const T* weight,
    float eps,
    int hidden_dim)
{ 
    int row_idx=blockIdx.x;
    int tid=threadIdx.x;

    const T* in_row=input+row_idx*hidden_dim;
    T* out_row=output+row_idx*hidden_dim;
    // Compute sum of squares
    float local_sq_sum=0.0f;
    for(int i=tid;i<hidden_dim;i+=blockDim.x)
    {
        float val=(float)in_row[i];
        local_sq_sum+=val*val;
    }
    // Reduce within block to get total sum of squares
    float total_sq_sum=blockReduceSum<float>(local_sq_sum);

    float mean_sq=total_sq_sum/(float)hidden_dim;
    float rsqrt_val=rsqrtf(mean_sq+eps);
    // Normalize and apply weight
    for(int i=tid;i<hidden_dim;i+=blockDim.x)
    {
        float val=(float)in_row[i];
        float norm_val=val*rsqrt_val;
        out_row[i]=(T)(norm_val*(float)weight[i]);
    }
}
// Host function to launch RMSNorm kernel
template<typename T>
void launch_rmsnorm_kernel(
    tensor_t output,
    tensor_t input,
    tensor_t weight,
    float eps)
{ 
    int hidden_dim=input->shape().back();
    int num_tokens=input->numel()/hidden_dim;

    T* d_out = reinterpret_cast<T*>(output->data());
    const T* d_in = reinterpret_cast<const T*>(input->data());
    const T* d_w = reinterpret_cast<const T*>(weight->data());
    dim3 grid(num_tokens);
    int threads=(hidden_dim<1024)?hidden_dim:1024;
    threads=((threads+31)/32)*32;
    dim3 block(threads);
    rmsnorm_kernel<T><<<grid,block>>>(d_out,d_in,d_w,eps,hidden_dim);
}
// Host function to launch RMSNorm kernel
void rmsnorm(tensor_t output, tensor_t input, tensor_t weight, float eps) {
    auto dtype = input->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_rmsnorm_kernel<float>(output, input, weight, eps);
            break;
        case LLAISYS_DTYPE_F16:
            launch_rmsnorm_kernel<__half>(output, input, weight, eps);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_rmsnorm_kernel<__nv_bfloat16>(output, input, weight, eps);
            break;
        default:
            fprintf(stderr, "[RMSNorm NVIDIA] Unsupported DataType: %d\n", dtype);
            abort();
    }
}

}//namespace llaisys::ops::nvidia
