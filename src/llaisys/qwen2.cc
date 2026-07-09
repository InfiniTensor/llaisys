#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_model.hpp"

__C {

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *model;
};

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model;
    model->model = new llaisys::models::Qwen2Model(meta, device, device_ids ? device_ids[0] : 0);
    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model->model;
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    auto weights = new LlaisysQwen2Weights;
    *weights = model->model->getWeights();
    return weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    return model->model->infer(token_ids, ntoken);
}

}
