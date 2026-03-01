#ifndef LLAISYS_MODELS_QWEN2_TP_H
#define LLAISYS_MODELS_QWEN2_TP_H

#include "qwen2.h"

__C {
    // Tensor Parallel Qwen2 Model
    struct LlaisysQwen2ModelTP;

    // Create a TP model with multiple devices
    // device_ids: array of device IDs (e.g., [0, 1, 2, 3] for 4-GPU TP)
    // ndevice: number of devices (TP world size)
    __export struct LlaisysQwen2ModelTP *llaisysQwen2ModelTPCreate(
        const struct LlaisysQwen2Meta *meta, 
        const int *device_ids, 
        int world_size);

    __export void llaisysQwen2ModelTPDestroy(struct LlaisysQwen2ModelTP *model);

    // Get weights for each rank
    // Returns array of weight pointers, one for each rank
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelTPWeights(
        struct LlaisysQwen2ModelTP *model, 
        int rank);

    __export int64_t llaisysQwen2ModelTPInfer(
        struct LlaisysQwen2ModelTP *model, 
        const int64_t *token_ids, 
        size_t ntoken);

    __export void llaisysQwen2ModelTPResetCache(struct LlaisysQwen2ModelTP *model);

    // Get the number of ranks in the TP group
    __export int llaisysQwen2ModelTPGetWorldSize(struct LlaisysQwen2ModelTP *model);
}

#endif // LLAISYS_MODELS_QWEN2_TP_H
