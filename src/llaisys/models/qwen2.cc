#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"

#include "../../models/qwen2/model.hpp"

__C {

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids,
                                                  int ndevice) {
    try {
        CHECK_ARGUMENT(meta != nullptr, "llaisysQwen2ModelCreate: meta is null");
        auto *m = new LlaisysQwen2Model;
        m->impl = llaisys::models::Qwen2Model::create(*meta, device, device_ids, ndevice);
        return m;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] llaisysQwen2ModelCreate failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    try {
        CHECK_ARGUMENT(model != nullptr, "llaisysQwen2ModelWeights: model is null");
        return model->impl->weights();
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] llaisysQwen2ModelWeights failed: " << e.what() << std::endl;
        return nullptr;
    }
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    try {
        CHECK_ARGUMENT(model != nullptr, "llaisysQwen2ModelInfer: model is null");
        return model->impl->infer(token_ids, ntoken);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] llaisysQwen2ModelInfer failed: " << e.what() << std::endl;
        // Be conservative: return EOS/end token so the caller can stop generation.
        return model->impl ? model->impl->endToken() : -1;
    }
}
}
