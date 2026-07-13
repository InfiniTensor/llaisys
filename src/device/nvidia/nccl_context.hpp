#pragma once

#include "llaisys.h"

#include <mutex>
#include <string>

struct ncclComm;
typedef ncclComm *ncclComm_t;

namespace llaisys::device::nvidia {

enum class NcclReduceOp {
    Sum = 0,
};

class NcclContext {
private:
    int _rank;
    int _world_size;
    int _device_id;
    ncclComm_t _comm;
    void *_barrier_buffer;
    mutable std::mutex _mutex;

public:
    NcclContext(int rank, int world_size, int device_id);
    ~NcclContext();

    NcclContext(const NcclContext &) = delete;
    NcclContext &operator=(const NcclContext &) = delete;
    NcclContext(NcclContext &&) = delete;
    NcclContext &operator=(NcclContext &&) = delete;

    int rank() const;
    int worldSize() const;
    int deviceId() const;

    void allReduce(void *data, size_t count, llaisysDataType_t dtype, llaisysStream_t stream, NcclReduceOp op = NcclReduceOp::Sum) const;
    void allGather(const void *send_buffer, void *recv_buffer, size_t count, llaisysDataType_t dtype, llaisysStream_t stream) const;
    void broadcast(void *data, size_t count, llaisysDataType_t dtype, int root, llaisysStream_t stream) const;
    void barrier(llaisysStream_t stream) const;
};

} // namespace llaisys::device::nvidia
