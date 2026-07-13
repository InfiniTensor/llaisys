#include "../op.hpp"
#include "../../../tensor/tensor.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia{
    template<typename T>
    __global__ void add_kernel(T* c,const T* a,const T* b,size_t n)
    {
        int idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<n)
        {
            c[idx]=a[idx]+b[idx];
        }
    }

template<typename T>
void launch_add(void* c,const void* a,const void* b,size_t n)
{
    T* d_c=reinterpret_cast<T*>(c);
    const T* d_a=reinterpret_cast<const T*>(a);
    const T* d_b=reinterpret_cast<const T*>(b);

    int threads=256;
    int blocks=(n+threads-1)/threads;
    add_kernel<<<blocks,threads>>>(d_c,d_a,d_b,n);
}


void add(tensor_t output,const tensor_t input1,const tensor_t input2)
{
    size_t numel=output->numel();
    auto dtype=output->dtype();

    void* d_c=output->data();
    const void* d_a=input1->data();
    const void* d_b=input2->data();

    switch(dtype)
    {
        case LLAISYS_DTYPE_F32:
            launch_add<float>(d_c,d_a,d_b,numel);
            break;
        case LLAISYS_DTYPE_F16:
            launch_add<__half>(d_c,d_a,d_b,numel);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_add<__nv_bfloat16>(d_c,d_a,d_b,numel);
            break;
        default:
            fprintf(stderr, "[Add] Unsupported DataType on CUDA: %d\n", dtype);
            abort();
    }
}
} // namespace llaisys::ops::nvidia