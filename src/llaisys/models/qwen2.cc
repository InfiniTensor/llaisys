#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/model.hpp"

using namespace llaisys::models::qwen2;

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    int dev_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
    Qwen2Model* model = new Qwen2Model(*meta, device, dev_id);
    return reinterpret_cast<LlaisysQwen2Model*>(model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) {
        delete reinterpret_cast<Qwen2Model*>(model);
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return reinterpret_cast<Qwen2Model*>(model)->getWeightsStruct();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model) return -1;
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
    return reinterpret_cast<Qwen2Model*>(model)->infer(tokens);
}

}
