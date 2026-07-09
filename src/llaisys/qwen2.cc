#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta || ndevice < 1) return nullptr;
    // For now support single device
    int device_id = device_ids ? device_ids[0] : 0;
    
    // Copy meta
    LlaisysQwen2Meta cpp_meta = *meta;
    
    auto* model = new llaisys::Qwen2Model(cpp_meta, device, device_id);
    return reinterpret_cast<struct LlaisysQwen2Model*>(model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) {
        delete reinterpret_cast<llaisys::Qwen2Model*>(model);
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    auto* cpp_model = reinterpret_cast<llaisys::Qwen2Model*>(model);
    return cpp_model->getWeights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model) return -1;
    auto* cpp_model = reinterpret_cast<llaisys::Qwen2Model*>(model);
    return cpp_model->infer(token_ids, ntoken);
}

}
