#include "../runtime_api.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count=0;
    //调用CUDA API 获取设备数量
    cudaError_t err=cudaGetDeviceCount(&count);
    //如果失败则返回0
    if(err!=cudaSuccess)
        return 0;
    return count;
}

void setDevice(int device_id) {
    //调用 CUDA API 设置设备
    cudaError_t err=cudaSetDevice(device_id);
    //失败则报错
    if(err!=cudaSuccess)
        fprintf(stderr,"Cuda Set Device failed:%s\n",cudaGetErrorString(err));
}

void deviceSynchronize() {
    //调用 CUDA API 等待设备所有操作完成
    cudaError_t err=cudaDeviceSynchronize();
    //失败则报错
    if(err!=cudaSuccess)
        fprintf(stderr,"Cuda Device Synchronize failed:%s\n",cudaGetErrorString(err));
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    //调用 CUDA API 创建流
    cudaError_t err=cudaStreamCreate(&stream);
    //失败则报错并返回空指针
    if(err!=cudaSuccess){
        fprintf(stderr,"CUDA Stream Create failed:%s\n",cudaGetErrorString(err));
        return nullptr;
    }
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    //如果流是空的，直接返回，避免崩溃
    if(stream==nullptr)
        return;
    cudaStream_t cuda_stream=(cudaStream_t)stream;
    cudaError_t err=cudaStreamDestroy(cuda_stream);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA Stream Destroy failed: %s\n", cudaGetErrorString(err));
}

void streamSynchronize(llaisysStream_t stream) {
    //如果流是空的（可能是默认流），直接返回或做特殊处理
    if(stream==nullptr)
        return;
    //调用 CUDA API 等待流完成
    cudaStream_t cuda_stream=(cudaStream_t)stream;
    cudaError_t err=cudaStreamSynchronize(cuda_stream);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA stream Synchronize failed: %s\n", cudaGetErrorString(err));
}

void *mallocDevice(size_t size) {
    void *ptr=nullptr;
    //调用 CUDA API 分配显存
    cudaError_t err=cudaMalloc(&ptr,size);
    //失败则报错并返回空指针
    if(err!=cudaSuccess){
        fprintf(stderr,"CUDA Malloc Device failed:%s\n",cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void freeDevice(void *ptr) {
    //如果指针为空，直接返回，无需释放
    if(ptr==nullptr)
        return;
    cudaError_t err=cudaFree(ptr);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA Free Device failed: %s\n", cudaGetErrorString(err));
}

void *mallocHost(size_t size) {
    void *ptr=nullptr;
    //调用 CUDA API 分配页锁定主机内存
    cudaError_t err=cudaMallocHost(&ptr,size);
    //失败则报错并返回空指针
    if(err!=cudaSuccess){
        fprintf(stderr,"CUDA Malloc Host failed:%s\n",cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void freeHost(void *ptr) {
    //如果指针为空，直接返回
    if(ptr==nullptr)
        return;
    cudaError_t err=cudaFreeHost(ptr);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA Free Host failed: %s\n", cudaGetErrorString(err));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    //将LLAISYS的枚举转换为 CUDA 的枚举
    cudaMemcpyKind cuda_kind=(cudaMemcpyKind)kind;
    //调用CUDA API 进行同步拷贝
    cudaError_t err=cudaMemcpy(dst,src,size,cuda_kind);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA Memcpy Sync failed: %s\n", cudaGetErrorString(err));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,llaisysStream_t stream) {
    //将LLAISYS的枚举转换为 CUDA 的枚举
    cudaMemcpyKind cuda_kind=(cudaMemcpyKind)kind;
    cudaStream_t cuda_stream=stream;
    //调用 CUDA API 进行异步拷贝
    cudaError_t err=cudaMemcpyAsync(dst,src,size,cuda_kind,cuda_stream);
    //失败则报错
    if (err!=cudaSuccess)
        fprintf(stderr, "CUDA Memcpy Async failed: %s\n", cudaGetErrorString(err));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
