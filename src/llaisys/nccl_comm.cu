/**
 * NCCL 通信实现（仅当 ENABLE_NCCL 且 ENABLE_NVIDIA_API 时编译）。
 */
#if defined(ENABLE_NCCL) && defined(ENABLE_NVIDIA_API)

#include "llaisys/nccl_comm.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

static ncclComm_t g_nccl_comm = nullptr;
static thread_local char g_last_error[256] = "";

static ncclDataType_t to_nccl_dtype(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return ncclFloat32;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16: return ncclFloat16;
    case LLAISYS_DTYPE_I64: return ncclInt64;
    default: return ncclFloat32;
    }
}

extern "C" {

void llaisysNcclGetUniqueId(void *buffer) {
    if (!buffer) return;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::memcpy(buffer, &id, LLAISYS_NCCL_UNIQUE_ID_BYTES);
}

int llaisysNcclInitRank(int rank, int world_size, const void *unique_id) {
    if (g_nccl_comm != nullptr) return 0;
    if (!unique_id || world_size < 1 || rank < 0 || rank >= world_size) return -1;
    // NCCL 要求每进程在 ncclCommInitRank 前已初始化 CUDA；测试中每进程通过 CUDA_VISIBLE_DEVICES 仅见一卡，设备号为 0
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) {
        std::snprintf(g_last_error, sizeof(g_last_error), "cudaSetDevice(0) failed: %s", cudaGetErrorString(ce));
        return -1;
    }
    ncclUniqueId id;
    std::memcpy(&id, unique_id, LLAISYS_NCCL_UNIQUE_ID_BYTES);
    ncclResult_t r = ncclCommInitRank(&g_nccl_comm, world_size, id, rank);
    if (r != ncclSuccess) {
        std::snprintf(g_last_error, sizeof(g_last_error), "ncclCommInitRank: %s", ncclGetErrorString(r));
        return -1;
    }
    g_last_error[0] = '\0';
    return 0;
}

int llaisysNcclAllReduce(const void *sendbuf, void *recvbuf, size_t count,
                         llaisysDataType_t dtype, void *stream) {
    if (!g_nccl_comm || !sendbuf || !recvbuf) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    ncclResult_t r = ncclAllReduce(sendbuf, recvbuf, count, to_nccl_dtype(dtype), ncclSum, g_nccl_comm, s);
    if (r != ncclSuccess) {
        std::fprintf(stderr, "[llaisys] ncclAllReduce failed: %s\n", ncclGetErrorString(r));
        return -1;
    }
    return 0;
}

int llaisysNcclAllGather(const void *sendbuf, void *recvbuf, size_t count_per_rank,
                         llaisysDataType_t dtype, void *stream) {
    if (!g_nccl_comm || !sendbuf || !recvbuf) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    ncclResult_t r = ncclAllGather(sendbuf, recvbuf, count_per_rank, to_nccl_dtype(dtype), g_nccl_comm, s);
    if (r != ncclSuccess) {
        std::fprintf(stderr, "[llaisys] ncclAllGather failed: %s\n", ncclGetErrorString(r));
        return -1;
    }
    return 0;
}

void llaisysNcclDestroy(void) {
    if (g_nccl_comm != nullptr) {
        ncclCommDestroy(g_nccl_comm);
        g_nccl_comm = nullptr;
    }
}

const char *llaisysNcclGetLastError(void) {
    return g_last_error[0] ? g_last_error : "(no error)";
}

} // extern "C"

#endif /* ENABLE_NCCL && ENABLE_NVIDIA_API */
