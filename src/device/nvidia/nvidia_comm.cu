#include "../comm_api.hpp"
#include "cuda_utils.hpp"

#include <cstring>
#include <nccl.h>
#include <stdexcept>

namespace llaisys::device::nvidia {

inline void nccl_check(ncclResult_t result) {
    if (result == ncclSuccess) {
        return;
    }
    throw std::runtime_error(ncclGetErrorString(result));
}

inline ncclDataType_t to_nccl_dtype(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return ncclFloat32;
    case LLAISYS_DTYPE_F16: return ncclFloat16;
    case LLAISYS_DTYPE_BF16: return ncclBfloat16;
    case LLAISYS_DTYPE_I32: return ncclInt32;
    case LLAISYS_DTYPE_I8: return ncclInt8;
    default: throw std::runtime_error("Unsupported data type");
    }
}

inline ncclRedOp_t to_nccl_op(llaisysReduceOp_t op) {
    switch (op) {
    case LLAISYS_REDUCE_SUM: return ncclSum;
    case LLAISYS_REDUCE_PROD: return ncclProd;
    case LLAISYS_REDUCE_MIN: return ncclMin;
    case LLAISYS_REDUCE_MAX: return ncclMax;
    default: throw std::runtime_error("Unsupported reduce op");
    }
}

namespace nccl {

int commInit(llaisysComm_t *comm, int rank, int size, const void *unique_id) {
    ncclComm_t nccl_comm;
    ncclUniqueId id;

    if (unique_id) {
        memcpy(&id, unique_id, sizeof(id));
    } else if (rank == 0) {
        nccl_check(ncclGetUniqueId(&id));
    }

    nccl_check(ncclCommInitRank(&nccl_comm, size, id, rank));
    *comm = reinterpret_cast<llaisysComm_t>(nccl_comm);
    return 0;
}

int commGenerateUniqueId(void *id_out, size_t *id_size) {
    ncclUniqueId id;
    nccl_check(ncclGetUniqueId(&id));
    memcpy(id_out, &id, sizeof(id));
    *id_size = sizeof(id);
    return 0;
}

void commDestroy(llaisysComm_t comm) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    nccl_check(ncclCommDestroy(nccl_comm));
}

int commGetRank(llaisysComm_t comm) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    int rank;
    nccl_check(ncclCommUserRank(nccl_comm, &rank));
    return rank;
}

int commGetSize(llaisysComm_t comm) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    int size;
    nccl_check(ncclCommCount(nccl_comm, &size));
    return size;
}

void commAllreduce(const void *sendbuf, void *recvbuf, size_t count,
                   llaisysDataType_t dtype, llaisysReduceOp_t op,
                   llaisysComm_t comm, llaisysStream_t stream) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    nccl_check(ncclAllReduce(sendbuf, recvbuf, count, to_nccl_dtype(dtype),
                             to_nccl_op(op), nccl_comm, cuda_stream));
}

void commBroadcast(void *buf, size_t count, llaisysDataType_t dtype, int root,
                   llaisysComm_t comm, llaisysStream_t stream) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    nccl_check(ncclBroadcast(buf, buf, count, to_nccl_dtype(dtype), root,
                             nccl_comm, cuda_stream));
}

void commSend(const void *buf, size_t count, llaisysDataType_t dtype, int peer,
              llaisysComm_t comm, llaisysStream_t stream) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    nccl_check(ncclSend(buf, count, to_nccl_dtype(dtype), peer, nccl_comm,
                        cuda_stream));
}

void commRecv(void *buf, size_t count, llaisysDataType_t dtype, int peer,
              llaisysComm_t comm, llaisysStream_t stream) {
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    nccl_check(ncclRecv(buf, count, to_nccl_dtype(dtype), peer, nccl_comm,
                        cuda_stream));
}

static const LlaisysCommAPI NCCL_COMM_API = {
    &commInit,
    &commDestroy,
    &commGetRank,
    &commGetSize,
    &commAllreduce,
    &commBroadcast,
    &commSend,
    &commRecv
};

const LlaisysCommAPI *getCommAPI() {
    return &NCCL_COMM_API;
}

} // namespace nccl
} // namespace llaisys::device::nvidia

namespace llaisys::device::nccl {
const LlaisysCommAPI *getCommAPI() {
    return llaisys::device::nvidia::nccl::getCommAPI();
}
int commGenerateUniqueId(void *id_out, size_t *id_size) {
    return llaisys::device::nvidia::nccl::commGenerateUniqueId(id_out, id_size);
}
}
