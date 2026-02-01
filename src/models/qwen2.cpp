#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "llaisys.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
using namespace llaisys;

struct Qwen2Layer{
    tensor_t attention_norm;
    tensor_t w_q;
    tensor_t w_k;
    tensor_t w_v;
    tensor_t w_o;

    tensor_t b_q;
    tensor_t b_k;
    tensor_t b_v;
    tensor_t b_o;

    tensor_t ffn_norm;
    tensor_t w_gate;
    tensor_t w_up;
    tensor_t w_down;
};
struct LlaisysQwen2Model{
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;

    tensor_t tok_embeddings;
    tensor_t norm;
    tensor_t output;

    std::vector<Qwen2Layer> layers;

    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
};

extern "C"{

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
    auto model=new LlaisysQwen2Model();
    model->meta=*meta;
    model->device=device;
    model->layers.resize(meta->nlayer);
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
    size_t*shape,
    size_t ndim,
    llaisysDataType_t dtype
){
    std::string w_name(name);
    tensor_t* target=nullptr;

    if(w_name=="model.embed_tokens.weight"){
        target=&model->tok_embeddings;
    }
    else if(w_name=="model.norm.weight"){
        target=&model->norm;
    }
    else if(w_name=="lm_head.weight"){
        target=&model->output;
    }

    else if(w_name.find("model.layers.")==0){
        size_t first_dot=13;
        size_t second_dot=w_name.find(".",first_dot);
        std::string layer_id_str=w_name.substr(first_dot,second_dot-first_dot);
        size_t layer_id=std::stoull(layer_id_str);

        if(layer_id>=0&&layer_id<model->meta.nlayer){
            auto&layer=model->layers[layer_id];
            std::string suffix=w_name.substr(second_dot+1);

            if(suffix=="self_attn.q_proj.weight") target=&layer.w_q;
            else if(suffix=="self_attn.k_proj.weight") target=&layer.w_k;
            else if(suffix=="self_attn.v_proj.weight") target=&layer.w_v;
            else if(suffix=="self_attn.o_proj.weight") target=&layer.w_o;
            else if(suffix=="self_attn.q_proj.bias") target=&layer.b_q;
            else if(suffix=="self_attn.k_proj.bias") target=&layer.b_k;
            else if(suffix=="self_attn.v_proj.bias") target=&layer.b_v;
            else if(suffix=="self_attn.o_proj.bias") target=&layer.b_o;
            else if(suffix=="mlp.gate_proj.weight") target=&layer.w_gate;
            else if(suffix=="mlp.up_proj.weight") target=&layer.w_up;
            else if(suffix=="mlp.down_proj.weight") target=&layer.w_down;
            else if(suffix=="input_layernorm.weight") target=&layer.attention_norm;
            else if(suffix=="post_attention_layernorm.weight") target=&layer.ffn_norm;
        }
    }
    if(target){
        std::vector<size_t> shape_vec(shape,shape+ndim);

        *target=Tensor::create(shape_vec,dtype,model->device,0);

        (*target)->load(data);

        printf("Cpp loaded: %s -> shape [",name);
        for(size_t i=0;i<ndim;++i) printf("%lu%s",shape[i],i==ndim-1?"":",");
        printf("]\n");
    }
    else{
        printf("Warning: skipped unknown weight: %s\n",name);
    }

    fflush(stdout);

}
tensor_t llaisysQwen2ModelForward(
    LlaisysQwen2Model* model,
    int64_t*input_ids_ptr,
    size_t seq_len,
    size_t start_pos
){
    auto device=model->device;
    auto dtype=model->tok_embeddings->dtype();
    size_t hs=model->meta.hs;

    std::vector<size_t> input_shape={1,seq_len};
    
    tensor_t input_tensor=Tensor::create(input_shape,LLAISYS_DTYPE_I64,device,0);

    input_tensor->load(input_ids_ptr);

    std::vector<size_t> hidden_shape={1,seq_len,hs};

    tensor_t hidden_states=Tensor::create(hidden_shape,dtype,device,0);

    ops::embedding(hidden_states, input_tensor, model->tok_embeddings);

    return hidden_states;

}
}