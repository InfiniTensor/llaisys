#ifndef LLAISYS_COMM_H
#define LLAISYS_COMM_H

#include "../llaisys.h"

__C {
    // Communication Types
    typedef void *llaisysComm_t;

    typedef enum {
        LLAISYS_COMM_NCCL = 0,
        LLAISYS_COMM_IXCCL = 1,
        LLAISYS_COMM_MPI = 2,
    } llaisysCommBackend_t;

    typedef enum {
        LLAISYS_REDUCE_SUM = 0,
        LLAISYS_REDUCE_PROD = 1,
        LLAISYS_REDUCE_MIN = 2,
        LLAISYS_REDUCE_MAX = 3,
    } llaisysReduceOp_t;

    #define LLAISYS_COMM_UNIQUE_ID_MAX_SIZE 128

    // Communication API Functions
    typedef int (*comm_init_api)(llaisysComm_t *, int, int, const void *);
    typedef void (*comm_destroy_api)(llaisysComm_t);
    typedef int (*comm_get_rank_api)(llaisysComm_t);
    typedef int (*comm_get_size_api)(llaisysComm_t);
    typedef void (*comm_allreduce_api)(const void *, void *, size_t, llaisysDataType_t, llaisysReduceOp_t, llaisysComm_t, llaisysStream_t);
    typedef void (*comm_broadcast_api)(void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t);
    typedef void (*comm_send_api)(const void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t);
    typedef void (*comm_recv_api)(void *, size_t, llaisysDataType_t, int, llaisysComm_t, llaisysStream_t);

    struct LlaisysCommAPI {
        comm_init_api init;
        comm_destroy_api destroy;
        comm_get_rank_api get_rank;
        comm_get_size_api get_size;
        comm_allreduce_api allreduce;
        comm_broadcast_api broadcast;
        comm_send_api send;
        comm_recv_api recv;
    };

    __export const LlaisysCommAPI *llaisysGetCommAPI(llaisysCommBackend_t);
    __export int llaisysCommGenerateUniqueId(llaisysCommBackend_t backend, void *id_out, size_t *id_size);
}

#endif // LLAISYS_COMM_H
