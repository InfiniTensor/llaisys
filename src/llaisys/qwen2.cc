#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"
#include "llaisys_tensor.hpp"

__C {
    struct LlaisysQwen2Model {
        std::shared_ptr<llaisys::models::Qwen2Model> model;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        // For now, only support single device
        int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        
        auto cpp_model = std::make_shared<llaisys::models::Qwen2Model>(meta, device, device_id);
        auto model = new LlaisysQwen2Model{cpp_model};
        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model) {
            delete model;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model || !model->model) return nullptr;
        return model->model->getWeights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !model->model) return -1;
        return model->model->infer(token_ids, ntoken);
    }
}
