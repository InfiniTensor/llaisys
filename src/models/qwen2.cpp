#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/add/op.hpp" 
#include "../ops/argmax/op.hpp" 
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
    printf("DEBUG: Meta Check -> nlayer=%lu, hs=%lu, maxseq=%lu\n", 
        model->meta.nlayer, model->meta.hs, model->meta.maxseq);
    fflush(stdout);
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
void* llaisysQwen2ModelForward(
    LlaisysQwen2Model* model,
    int64_t*input_ids_ptr,
    size_t seq_len,
    size_t start_pos
){
    printf("DEBUG: Entered Forward | seq_len=%lu | start_pos=%lu\n", seq_len, start_pos); fflush(stdout);//debug
    auto device=model->device;
    auto dtype=model->tok_embeddings->dtype();
    size_t hs=model->meta.hs;
    size_t head_dim=hs/model->meta.nh;
    size_t kv_dim=head_dim*model->meta.nkvh;
    std::vector<size_t> q_shape={1,seq_len,hs};
    std::vector<size_t> kv_shape={1,seq_len,kv_dim};

    std::vector<size_t> input_shape={1,seq_len};
    
    tensor_t input_tensor=Tensor::create(input_shape,LLAISYS_DTYPE_I64,device,0);
    printf("DEBUG: Loading input ptr...\n"); fflush(stdout);
    if (!input_ids_ptr) { printf("ERROR: input_ids_ptr is NULL!\n"); exit(1); }//debug
    input_tensor->load(input_ids_ptr);
    
    std::vector<size_t> hidden_shape={1,seq_len,hs};

    tensor_t hidden_states=Tensor::create(hidden_shape,dtype,device,0);

    printf("DEBUG: Running Embedding...\n"); fflush(stdout);//debug
    ops::embedding(hidden_states, input_tensor, model->tok_embeddings);

    printf("DEBUG: Inside Forward. start_pos = %ld\n", start_pos); 
    fflush(stdout);
    std::vector<size_t> pos_shape={1,seq_len};
    tensor_t pos_ids=Tensor::create(pos_shape, LLAISYS_DTYPE_I64,device,0);
    printf("DEBUG: Creating Pos IDs...\n"); fflush(stdout);//debug
    std::vector<int64_t> pos_vec(seq_len);
    for(size_t i=0;i<seq_len;++i) pos_vec[i]=start_pos+i;
    pos_ids->load(pos_vec.data());

    for(size_t i=0;i<model->meta.nlayer;++i){
        printf("DEBUG: Layer %lu start\n", i); fflush(stdout);//debug
        auto&layer=model->layers[i];
        
        tensor_t norm_out=Tensor::create(hidden_shape,dtype,device,0);
        ops::rms_norm(norm_out, hidden_states, layer.attention_norm,model->meta.epsilon);

        tensor_t q=Tensor::create(q_shape,dtype,device,0);
        tensor_t k=Tensor::create(kv_shape,dtype,device,0);
        tensor_t v=Tensor::create(kv_shape,dtype,device,0);

        ops::linear(q, norm_out, layer.w_q, layer.b_q);
        ops::linear(k, norm_out, layer.w_k, layer.b_k);
        ops::linear(v, norm_out, layer.w_v, layer.b_v);

        ops::rope(q, q, pos_ids, 1000000.0f);
        ops::rope(k, k, pos_ids, 1000000.0f);
        // printf("DEBUG: Layer %lu RoPE done. Entering KV Cache...\n", i); fflush(stdout);//debug

        tensor_t k_slot=model->k_cache[i]->slice(1,start_pos,start_pos+seq_len);
        tensor_t v_slot=model->v_cache[i]->slice(1,start_pos,start_pos+seq_len);

        // printf("DEBUG: Layer %lu loading KV Cache...\n", i); fflush(stdout);//debug

        if (!k->data()) printf("ERROR: k->data() is NULL\n");
        if (!v->data()) printf("ERROR: v->data() is NULL\n");//debug

        k_slot->load(k->data());
        v_slot->load(v->data());

        tensor_t full_k=model->k_cache[i]->slice(1,0,start_pos+seq_len);
        tensor_t full_v=model->v_cache[i]->slice(1,0,start_pos+seq_len);

        // printf("DEBUG: Layer %lu KV Cache loaded. Running Attention...\n", i); fflush(stdout);//debug

        tensor_t attn_out=Tensor::create(hidden_shape,dtype,device,0);
        float scale = 0.0883883f;
        // printf("DEBUG: SelfAttn Check -> q[%lu,%lu,%lu], k[%lu,%lu,%lu], scale=%f\n", 
        //     q->shape()[0], q->shape()[1], q->shape()[2],
        //     full_k->shape()[0], full_k->shape()[1], full_k->shape()[2],
        //     scale);
        // fflush(stdout);

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
        // printf("DEBUG: Layer %lu FFN done.\n", i); fflush(stdout);//debug
    }

    tensor_t final_norm=Tensor::create(hidden_shape,dtype,device,0);
    ops::rms_norm(final_norm, hidden_states, model->norm, model->meta.epsilon);

    size_t vocab_size=model->output->shape()[0];
    std::vector<size_t> logits_shape={1,seq_len,vocab_size};
    tensor_t logits=Tensor::create(logits_shape,final_norm->dtype(),device,0);

    ops::linear(logits, final_norm, model->output, nullptr);
    printf("DEBUG: Final Norm & Head...\n"); fflush(stdout);

    tensor_t* heap_logits=new tensor_t(logits);
    return (void*)heap_logits;

}
int llaisysQwen2Sample(void* logits_void_ptr) {
    if (!logits_void_ptr) {
        printf("Error: logits ptr is NULL\n");
        return 0;
    }

    tensor_t* ptr_to_shared = (tensor_t*)logits_void_ptr;
    tensor_t logits = *ptr_to_shared;

    size_t seq_len = logits->shape()[1];

    tensor_t last_token_logits=logits->slice(1,seq_len-1,seq_len);

    tensor_t final_logits=last_token_logits->contiguous();

    if (final_logits->dtype() == LLAISYS_DTYPE_BF16) {
        uint16_t* debug_ptr = (uint16_t*)final_logits->data();
        float val0 = llaisys::utils::_bf16_to_f32(llaisys::bf16_t{debug_ptr[0]});
        // 确保你的 Logits[46055] 依然在合理范围内
        printf("[NUMERICAL] Logits[0]: %f\n", val0);
    }

    std::vector<size_t> out_shape = {1};
    tensor_t max_idx = Tensor::create(out_shape, LLAISYS_DTYPE_I64, logits->deviceType(), logits->deviceId());
    tensor_t max_val = Tensor::create(out_shape, logits->dtype(), logits->deviceType(), logits->deviceId());

    ops::argmax(max_idx, max_val, final_logits);

    // 6. 获取并返回结果
    int64_t result_index = *reinterpret_cast<int64_t*>(max_idx->data());

    printf("[DEBUG Cpp] Argmax Result (as I64): %ld\n", result_index);
    fflush(stdout);

    delete ptr_to_shared;
    return (int)result_index;
}
}