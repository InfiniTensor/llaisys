#include "llaisys/models/qwen2.h"
#include "../../models/qwen2.hpp"
#include "../../utils.hpp"

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *model;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        if (meta == nullptr) {
            std::cerr << "[ERROR] Qwen2ModelCreate: meta is null" << std::endl;
            return nullptr;
        }
        try {
            auto *model = new llaisys::models::Qwen2Model(*meta, device, device_ids, ndevice);
            return new LlaisysQwen2Model{model};
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Qwen2ModelCreate: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (!model) {
            return;
        }
        delete model->model;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (model == nullptr) {
            std::cerr << "[ERROR] Qwen2ModelWeights: model is null" << std::endl;
            return nullptr;
        }
        return model->model->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, float temperature, int top_k, float top_p) {
        if (model == nullptr) {
            std::cerr << "[ERROR] Qwen2ModelInfer: model is null" << std::endl;
            return -1;
        }
        try {
            return model->model->infer(token_ids, ntoken, temperature, top_k, top_p);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Qwen2ModelInfer: " << e.what() << std::endl;
            return -1;
        }
    }
}
