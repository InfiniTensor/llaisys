#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include "llaisys.h"
#include <cstdio>
#include <vector>

extern "C"{
    struct LlaisysQwen2Model{
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;
    llaisys::tensor_t hidden_states;
};
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
    auto model=new LlaisysQwen2Model();
    model->meta=*meta;
    model->device=device;
    model->k_cache.resize(meta->nlayer);
    model->v_cache.resize(meta->nlayer);
    printf("Cpp:Qwen2 Model Initialized on device: %d ! Layers: %lu\n",(int)model->device,meta->nlayer);
    fflush(stdout);
    return model;
}
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model){
    if(model){
        delete model;
    }
}
void llaisysQwen2LoadWeight(
    LlaisysQwen2Model*model,
    const char*name,
    void*data,
    int*shape,
    int dim,
    llaisysDataType_t dtype
){
    printf("Cpp Loading: %s | Type Enum: %d | Size : %lu bytes\n",name,(int)dtype,llaisys::utils::dsize(dtype));
    fflush(stdout);
}
}