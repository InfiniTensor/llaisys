/**
 * NCCL 通信封装：用于张量并行下的 AllReduce / AllGather。
 * 仅当 ENABLE_NVIDIA_API 且 ENABLE_NCCL 时有效；否则接口为空实现或返回错误。
 */
#ifndef LLAISYS_NCCL_COMM_H
#define LLAISYS_NCCL_COMM_H

#include "../llaisys.h"

#ifdef __cplusplus
extern "C" {
#endif

/** NCCL 唯一 ID 字节数（用于进程间广播，再调用 llaisysNcclInitRank） */
#define LLAISYS_NCCL_UNIQUE_ID_BYTES 128

/**
 * 在 rank 0 上调用，将唯一 ID 写入 buffer（至少 LLAISYS_NCCL_UNIQUE_ID_BYTES 字节），
 * 然后通过文件/MPI/等广播给其他 rank，供 llaisysNcclInitRank 使用。
 */
__export void llaisysNcclGetUniqueId(void *buffer);

/**
 * 每个进程调用一次：用 rank 0 广播得到的 unique_id 初始化本进程的 NCCL 通信器。
 * rank in [0, world_size), world_size >= 1。GPU 由调用方在此前通过 setDevice 等设定。
 */
__export int llaisysNcclInitRank(int rank, int world_size, const void *unique_id);

/**
 * AllReduce：所有 rank 的 sendbuf 做 sum，结果写入各 rank 的 recvbuf。
 * count 为元素个数，dtype 为元素类型。stream 为 CUDA 流（void*），CPU 推理时传 NULL。
 */
__export int llaisysNcclAllReduce(const void *sendbuf, void *recvbuf, size_t count,
                                  llaisysDataType_t dtype, void *stream);

/**
 * AllGather：每个 rank 提供 sendbuf（count_per_rank 个元素），
 * 结果 recvbuf 为所有 rank 的 sendbuf 按 rank 顺序拼接（总长 count_per_rank * world_size）。
 */
__export int llaisysNcclAllGather(const void *sendbuf, void *recvbuf, size_t count_per_rank,
                                 llaisysDataType_t dtype, void *stream);

/** 释放 NCCL 通信器，进程退出前调用。 */
__export void llaisysNcclDestroy(void);

/** 返回最近一次 NCCL/CUDA 错误的描述（静态缓冲区，仅用于调试）。 */
__export const char *llaisysNcclGetLastError(void);

#ifdef __cplusplus
}
#endif

#endif /* LLAISYS_NCCL_COMM_H */
