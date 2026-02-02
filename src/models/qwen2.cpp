#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/add/op.hpp" 
#include "llaisys.h"
#include <cmath>
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

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice,llaisysDataType_t dtype){
    auto model=new LlaisysQwen2Model();
    model->meta=*meta;
    model->device=device;
    model->layers.resize(meta->nlayer);
    model->k_cache.resize(meta->nlayer);
    model->v_cache.resize(meta->nlayer);

    size_t head_dim=meta->hs/meta->nh;
    std::vector<size_t> cache_shape={1,meta->maxseq,meta->nkvh,head_dim};

    for(size_t i=0;i<meta->nlayer;++i){
        model->k_cache[i]=Tensor::create(cache_shape, dtype,device,0);
        model->v_cache[i]=Tensor::create(cache_shape, dtype,device,0);
    }
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
    size_t head_dim=hs/model->meta.nh;
    size_t kv_dim=head_dim*model->meta.nkvh;
    std::vector<size_t> q_shape={1,seq_len,hs};
    std::vector<size_t> kv_shape={1,seq_len,kv_dim};

    std::vector<size_t> input_shape={1,seq_len};
    
    tensor_t input_tensor=Tensor::create(input_shape,LLAISYS_DTYPE_I64,device,0);

    input_tensor->load(input_ids_ptr);

    std::vector<size_t> hidden_shape={1,seq_len,hs};

    tensor_t hidden_states=Tensor::create(hidden_shape,dtype,device,0);

    ops::embedding(hidden_states, input_tensor, model->tok_embeddings);

    std::vector<size_t> pos_shape={1,seq_len};
    tensor_t pos_ids=Tensor::create(pos_shape, LLAISYS_DTYPE_I64,device,0);

    std::vector<int64_t> pos_vec(seq_len);
    for(size_t i=0;i<seq_len;++i) pos_vec[i]=start_pos+i;
    pos_ids->load(pos_vec.data());

    for(size_t i=0;i<model->meta.nlayer;++i){
        auto&layer=model->layers[i];
        
        tensor_t norm_out=Tensor::create(hidden_shape,dtype,device,0);
        ops::rms_norm(norm_out, hidden_states, layer.attention_norm,model->meta.epsilon);

        tensor_t q=Tensor::create(q_shape,dtype,device,0);
        tensor_t k=Tensor::create(kv_shape,dtype,device,0);
        tensor_t v=Tensor::create(kv_shape,dtype,device,0);

        ops::linear(q, norm_out, layer.w_q, layer.b_q);
        ops::linear(k, norm_out, layer.w_k, layer.b_k);
        ops::linear(v, norm_out, layer.w_v, layer.b_v);

        ops::rope(q, q, pos_ids, model->meta.theta);
        ops::rope(k, k, pos_ids, model->meta.theta);
        // TODO: Implement KV Cache Append
        tensor_t k_slot=model->k_cache[i]->slice(1,start_pos,start_pos+seq_len);
        tensor_t v_slot=model->v_cache[i]->slice(1,start_pos,start_pos+seq_len);

        k_slot->load(k->data());
        v_slot->load(v->data());

        tensor_t full_k=model->k_cache[i]->slice(1,0,start_pos+seq_len);
        tensor_t full_v=model->v_cache[i]->slice(1,0,start_pos+seq_len);

        tensor_t attn_out=Tensor::create(hidden_shape,dtype,device,0);
        float scale=1.0f/std::sqrt((float)head_dim);

        ops::self_attention(attn_out, q, full_k, full_v, scale);

        tensor_t proj_out=Tensor::create(hidden_shape,dtype,device,0);
        ops::linear(proj_out, attn_out, layer.w_o, layer.b_o);

        ops::add(hidden_states, hidden_states, proj_out);

        tensor_t ffn_norm_out=Tensor::create(hidden_shape,dtype,device,0);
        ops::rms_norm(ffn_norm_out, hidden_states,layer.ffn_norm,model->meta.epsilon);

        size_t inter_size=layer.w_gate->shape()[0];
        std::vector<size_t> inter_shape={1,seq_len,inter_size};

        tensor_t gate=Tensor::create(inter_shape,dtype,device,0);
        tensor_t up=Tensor::create(inter_shape,dtype,device,0);

        ops::linear(gate, ffn_norm_out, layer.w_gate, nullptr);
        ops::linear(up, ffn_norm_out,layer.w_up,nullptr);

        tensor_t act=Tensor::create(inter_shape,dtype,device,0);
        ops::swiglu(act, gate, up);

        tensor_t mlp_out=Tensor::create(hidden_shape, dtype,device,0);
        ops::linear(mlp_out, act, layer.w_down, nullptr);

        ops::add(hidden_states, hidden_states, mlp_out);
    }

    tensor_t final_norm=Tensor::create(hidden_shape,dtype,device,0);
    ops::rms_norm(final_norm, hidden_states, model->norm, model->meta.epsilon);

    size_t vocab_size=model->output->shape()[0];
    std::vector<size_t> logits_shape={1,seq_len,vocab_size};
    tensor_t logits=Tensor::create(logits_shape, dtype,device,0);

    ops::linear(logits, final_norm, model->output, nullptr);

    return logits;

}
}