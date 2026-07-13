#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"

using namespace llaisys::models::qwen2;

__C {
    struct LlaisysQwen2Model {
        Qwen2Model *model;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        // Assume single device for now
        int dev_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        auto *cpp_model = new Qwen2Model(*meta, device, dev_id);
        return new LlaisysQwen2Model{cpp_model};
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (model) {
            delete model->model;
            delete model;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return model->model->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}