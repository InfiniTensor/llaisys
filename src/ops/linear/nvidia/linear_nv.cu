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

template<typename T>
cudaDataType_t get_cuda_datatype();

template<> cudaDataType_t get_cuda_datatype<float>(){return CUDA_R_32F;}
template<> cudaDataType_t get_cuda_datatype<__half>(){return CUDA_R_16F;}
template<> cudaDataType_t get_cuda_datatype<__nv_bfloat16>(){return CUDA_R_16BF;}

// Bias addition kernel
template<typename T>
__global__ void bias_add_kernel(T* output,const T* bias,int M,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total_elements=M*N;

    if(idx<total_elements){
        int col=idx%N;
        float val=(float)output[idx];
        float b=(float)bias[col];
        output[idx]=(T)(val+b);
    }
}

template<typename T>
void launch_linear_kernel(tensor_t output,tensor_t input,tensor_t weight,tensor_t bias){
    // Get cuBLAS handle from NVIDIA resource
    auto& ctx=llaisys::core::context();
    auto* nv_resource=dynamic_cast<llaisys::device::nvidia::Resource*>(ctx.runtime().deviceResource());
    if(!nv_resource){
        throw std::runtime_error("NVIDIA Resource not found in linear operator");
    }
    cublasHandle_t handle=nv_resource->cublasHandle();

    // Get dimensions
    int K=input->shape().back();
    int M=input->numel()/K;
    int N=weight->shape().front();

    // Set cuBLAS parameters
    cublasComputeType_t compute_type=CUBLAS_COMPUTE_32F;
    cudaDataType_t input_type=get_cuda_datatype<T>();

    float alpha=1.0f,beta=0.0f;

    const void* d_in=input->data();
    const void* d_weight=weight->data();
    void* d_out=output->data();

    // Perform matrix multiplication using cuBLAS
    cublasStatus_t status=cublasGemmEx(
        handle,
        CUBLAS_OP_T,CUBLAS_OP_N,
        N,M,K,
        &alpha,
        d_weight,input_type,K,
        d_in,input_type,K,
        &beta,
        d_out,input_type,N,
        compute_type,
        CUBLAS_GEMM_DEFAULT
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS GemmEx Error: %d\n", status);
    }
    // Launch bias addition kernel if bias is provided
    if(bias){
        int threads=256;
        int blocks=(M*N+threads-1)/threads;

        T* t_out=reinterpret_cast<T*>(output->data());
        const T* t_bias=reinterpret_cast<const T*>(bias->data());
        bias_add_kernel<T><<<blocks,threads>>>(t_out,t_bias,M,N);
    }
}


void linear(tensor_t output,tensor_t input,tensor_t weight,tensor_t bias)
{
    auto dtype=input->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_linear_kernel<float>(output, input, weight, bias);
            break;
        case LLAISYS_DTYPE_F16:
            launch_linear_kernel<__half>(output, input, weight, bias);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_linear_kernel<__nv_bfloat16>(output, input, weight, bias);
            break;
        default:
            fprintf(stderr, "[Linear NVIDIA] Unsupported DataType: %d\n", dtype);
            abort();
    }
}

}// namespace llaisys::ops::nvidia