#include "llaisys/models/qwen2.h"
#include "../models/qwen2_model.hpp"
#include "llaisys_tensor.hpp"

#include <memory>

extern "C" {

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
    
    LlaisysQwen2Model(std::unique_ptr<llaisys::models::Qwen2Model>&& model) 
        : impl(std::move(model)) {}
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, 
                                                  llaisysDeviceType_t device, 
                                                  int *device_ids, 
                                                  int ndevice) {
    try {
        // 当前简化实现，只使用第一个设备
        int device_id = (ndevice > 0) ? device_ids[0] : 0;
        
        auto model = std::make_unique<llaisys::models::Qwen2Model>(meta, device, device_id);
        return new LlaisysQwen2Model(std::move(model));
    } catch (...) {
        return nullptr;
    }
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->impl) {
        return nullptr;
    }
    
    return model->impl->weights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !model->impl) {
        return -1;
    }
    
    try {
        return model->impl->infer(token_ids, ntoken);
    } catch (...) {
        return -1;
    }
}

} // extern "C"
