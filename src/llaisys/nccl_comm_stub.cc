/**
 * NCCL 接口空实现：未启用 ENABLE_NCCL 时提供符号，避免链接失败。
 * 调用方不应在未启用 NCCL 时使用张量并行。
 */
#include "llaisys/nccl_comm.h"
#include <cstring>

extern "C" {

void llaisysNcclGetUniqueId(void *buffer) {
    if (buffer) std::memset(buffer, 0, LLAISYS_NCCL_UNIQUE_ID_BYTES);
}

int llaisysNcclInitRank(int rank, int world_size, const void *unique_id) {
    (void)rank;
    (void)world_size;
    (void)unique_id;
    return -1; /* not supported */
}

int llaisysNcclAllReduce(const void *sendbuf, void *recvbuf, size_t count,
                         llaisysDataType_t dtype, void *stream) {
    (void)sendbuf;
    (void)recvbuf;
    (void)count;
    (void)dtype;
    (void)stream;
    return -1;
}

int llaisysNcclAllGather(const void *sendbuf, void *recvbuf, size_t count_per_rank,
                         llaisysDataType_t dtype, void *stream) {
    (void)sendbuf;
    (void)recvbuf;
    (void)count_per_rank;
    (void)dtype;
    (void)stream;
    return -1;
}

void llaisysNcclDestroy(void) {}

const char *llaisysNcclGetLastError(void) {
    return "(NCCL not compiled)";
}

} // extern "C"
