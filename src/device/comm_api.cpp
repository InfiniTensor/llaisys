#include "comm_api.hpp"

namespace llaisys::device {

int commInit(llaisysComm_t *, int, int, const void *) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return -1;
}

void commDestroy(llaisysComm_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

int commGetRank(llaisysComm_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return -1;
}

int commGetSize(llaisysComm_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return -1;
}

void commAllreduce(const void *, void *, size_t, llaisysDataType_t, llaisysReduceOp_t, llaisysComm_t, llaisysStream_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void commBroadcast(void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void commSend(const void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void commRecv(void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

static const LlaisysCommAPI NOOP_COMM_API = {
    &commInit,
    &commDestroy,
    &commGetRank,
    &commGetSize,
    &commAllreduce,
    &commBroadcast,
    &commSend,
    &commRecv};

const LlaisysCommAPI *getUnsupportedCommAPI() {
    return &NOOP_COMM_API;
}

#ifdef ENABLE_ILUVATAR_API
namespace ixccl {
const LlaisysCommAPI *getCommAPI() {
    return getUnsupportedCommAPI();
}
} // namespace ixccl
#endif

const LlaisysCommAPI *getCommAPI(llaisysCommBackend_t backend) {
    switch (backend) {
    case LLAISYS_COMM_NCCL:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nccl::getCommAPI();
#else
        return getUnsupportedCommAPI();
#endif
    case LLAISYS_COMM_IXCCL:
#ifdef ENABLE_ILUVATAR_API
        return llaisys::device::ixccl::getCommAPI();
#else
        return getUnsupportedCommAPI();
#endif
    case LLAISYS_COMM_MPI:
        return getUnsupportedCommAPI();
    default:
        return getUnsupportedCommAPI();
    }
}

int commGenerateUniqueId(llaisysCommBackend_t backend, void *id_out, size_t *id_size) {
    switch (backend) {
    case LLAISYS_COMM_NCCL:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nccl::commGenerateUniqueId(id_out, id_size);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
        return -1;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return -1;
    }
}
} // namespace llaisys::device
