#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <cstring>


using namespace llaisys;

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;   // a.k.a. model.norm.weight
    std::vector<tensor_t> attn_norm_w; // a.k.a. input_layernorm.weight
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class debug {
public:
    debug& get() {
        static debug instance;
        return instance;
    }

    static void print_shape(tensor_t tensor, const std::string& tensor_name) {
        auto shape = tensor->shape();
        std::cout << tensor_name << " shape: [";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << shape[i];
            if (i != shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    template<typename... T>
    static void print(T... args) {
        ((std::cout << args << '\t'), ...);
        std::cout << std::endl;
    }

    debug(const debug&) = delete;
    debug& operator=(const debug&) = delete;
private:
    debug() {}
};

class Kv_cache {
public:
    Kv_cache(size_t nlayers, size_t dh, size_t nkvh, llaisysDataType_t dtype, llaisysDeviceType_t device_type)
    : nlayer_(nlayers), dh_(dh), nkvh_(nkvh), dtype_(dtype), device_type_(device_type) {
        k_cache_.resize(nlayer_);
        v_cache_.resize(nlayer_);
        total_len_.assign(nlayer_, 0);
        buf_size_.assign(nlayer_, 5);
        for(size_t i = 0; i < nlayer_; ++i) {
            k_cache_[i] = Tensor::create({buf_size_[i], nkvh_, dh_}, dtype_, device_type_);
            v_cache_[i] = Tensor::create({buf_size_[i], nkvh_, dh_}, dtype_, device_type_);
        }
    }

    void add(size_t layer_id, const tensor_t& k, const tensor_t& v, size_t seq_len) {//only support cpu for now
        if (layer_id >= nlayer_) {
            throw std::runtime_error("Layer id exceeds the number of layers in the model.");
        }
        auto& k_cache = k_cache_[layer_id];
        auto& v_cache = v_cache_[layer_id];
        if (total_len_[layer_id] + seq_len > buf_size_[layer_id]) {
            // If the total length exceeds the buffer size, we need to reallocate larger buffers and copy the existing data
            size_t new_buf_size = std::max(buf_size_[layer_id] * 2, total_len_[layer_id] + seq_len);
            auto new_k = Tensor::create({new_buf_size, nkvh_, dh_}, dtype_, device_type_);
            auto new_v = Tensor::create({new_buf_size, nkvh_, dh_}, dtype_, device_type_);
            llaisys::core::context().runtime().api()->memcpy_sync(
                new_k->data(),
                k_cache->data(),
                total_len_[layer_id] * nkvh_ * dh_ * k_cache->elementSize(),
                LLAISYS_MEMCPY_H2H
            );
            llaisys::core::context().runtime().api()->memcpy_sync(
                new_v->data(),
                v_cache->data(),
                total_len_[layer_id] * nkvh_ * dh_ * v_cache->elementSize(),
                LLAISYS_MEMCPY_H2H
            );
            k_cache_[layer_id] = new_k;
            v_cache_[layer_id] = new_v;
            buf_size_[layer_id] = new_buf_size;
        }
        // Copy the new k and v to the cache at the correct position.
        llaisys::core::context().runtime().api()->memcpy_sync(
            k_cache_[layer_id]->data() + total_len_[layer_id] * nkvh_ * dh_ * k->elementSize(),
            k->data(),
            k->numel() * k->elementSize(),
            LLAISYS_MEMCPY_H2H
        );
        llaisys::core::context().runtime().api()->memcpy_sync(
            v_cache_[layer_id]->data() + total_len_[layer_id] * nkvh_ * dh_ * v->elementSize(),
            v->data(),
            v->numel() * v->elementSize(),
            LLAISYS_MEMCPY_H2H
        );

        total_len_[layer_id] += seq_len;
    }

    tensor_t k(size_t layer_id) {
        if (layer_id >= nlayer_) {
            throw std::runtime_error("Layer id exceeds the number of layers in the model.");
        }
        return k_cache_[layer_id]->slice(0, 0, total_len_[layer_id]);
    }

    tensor_t v(size_t layer_id) {
        if (layer_id >= nlayer_) {
            throw std::runtime_error("Layer id exceeds the number of layers in the model.");
        }
        return v_cache_[layer_id]->slice(0, 0, total_len_[layer_id]);
    }

private:
    std::vector<tensor_t> k_cache_;
    std::vector<tensor_t> v_cache_;
    size_t nlayer_;
    std::vector<size_t> total_len_;
    std::vector<size_t> buf_size_;
    size_t dh_;
    size_t nkvh_;
    llaisysDataType_t dtype_;
    llaisysDeviceType_t device_type_;
};

class Qwen2ModelImpl {
public:
    Qwen2ModelImpl(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, const std::vector<int> &device_ids)
    : meta_(meta), device_(device), device_ids_(device_ids), kv_cache_(meta.nlayer, meta.dh, meta.nkvh, meta.dtype, device)
    {
        auto hs = meta_.hs;
        auto nh = meta_.nh;
        auto nkvh = meta_.nkvh;
        auto dh = meta_.dh;
        auto di = meta_.di;

        weights_.in_embed = Tensor::create({meta_.voc, hs}, meta_.dtype, device_, device_ids_[0]);
        weights_.out_embed = Tensor::create({meta_.voc, hs}, meta_.dtype, device_, device_ids_[0]);
        weights_.out_norm_w = Tensor::create({hs}, meta_.dtype, device_, device_ids_[0]);
        for (size_t i = 0; i < meta_.nlayer; ++i) {
            weights_.attn_norm_w.push_back(Tensor::create({hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_q_w.push_back(Tensor::create({nh * dh, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_q_b.push_back(Tensor::create({hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_k_w.push_back(Tensor::create({nkvh * dh, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_k_b.push_back(Tensor::create({nkvh * dh}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_v_w.push_back(Tensor::create({nkvh * dh, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_v_b.push_back(Tensor::create({nkvh * dh}, meta_.dtype, device_, device_ids_[0]));
            weights_.attn_o_w.push_back(Tensor::create({hs, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.mlp_norm_w.push_back(Tensor::create({hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.mlp_gate_w.push_back(Tensor::create({di, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.mlp_up_w.push_back(Tensor::create({di, hs}, meta_.dtype, device_, device_ids_[0]));
            weights_.mlp_down_w.push_back(Tensor::create({hs, di}, meta_.dtype, device_, device_ids_[0]));
        }
    }
    ~Qwen2ModelImpl() = default;

    void loadWeight(const void* src, std::string name) {
        if (name == "lm_head.weight") {
            weights_.out_embed->load(src);
        } else if (name.find("embed_tokens.weight") != std::string::npos) {
            weights_.in_embed->load(src);
        }else if (name == "model.norm.weight") {
            weights_.out_norm_w->load(src);
        } else {
            constexpr size_t prefix_len = 13; // "model.layers."
            name = name.substr(prefix_len);
            auto pos = name.find('.');
            auto layer_id = std::stoi(name.substr(0, pos));
            auto param_name = name.substr(pos + 1);
            if (param_name == "input_layernorm.weight") {
                weights_.attn_norm_w[layer_id]->load(src); 
            }else if (param_name == "self_attn.q_proj.weight") {
                weights_.attn_q_w[layer_id]->load(src);
            }else if (param_name == "self_attn.q_proj.bias") {
                weights_.attn_q_b[layer_id]->load(src);
            }else if (param_name == "self_attn.k_proj.weight") {
                weights_.attn_k_w[layer_id]->load(src);
            }else if (param_name == "self_attn.k_proj.bias") {
                weights_.attn_k_b[layer_id]->load(src);
            }else if (param_name == "self_attn.v_proj.weight") {
                weights_.attn_v_w[layer_id]->load(src);
            }else if (param_name == "self_attn.v_proj.bias") {
                weights_.attn_v_b[layer_id]->load(src);
            }else if (param_name == "self_attn.o_proj.weight") {
                weights_.attn_o_w[layer_id]->load(src);
            }else if (param_name == "post_attention_layernorm.weight") {
                weights_.mlp_norm_w[layer_id]->load(src);
            }else if (param_name == "mlp.gate_proj.weight") {
                weights_.mlp_gate_w[layer_id]->load(src);
            }else if (param_name == "mlp.up_proj.weight") {
                weights_.mlp_up_w[layer_id]->load(src);
            }else if (param_name == "mlp.down_proj.weight") {
                weights_.mlp_down_w[layer_id]->load(src);
            }else {
                throw std::runtime_error("Unknown weight name: " + name);
            }
        }
    }

    size_t forward_with_cache(const std::vector<int64_t> &input_ids) {
        using namespace ops;
        size_t seq_len = input_ids.size();
        //Allocate tensor
        auto cache = get_(seq_len);
        auto& tensor_input_ids  = cache.tensor_input_ids;
        auto& x                 = cache.x;
        auto& x_norm            = cache.x_norm;
        auto& q                 = cache.q;
        auto& k_                = cache.k_;
        auto& v_                = cache.v_;
        auto& q_rope            = cache.q_rope;
        auto& k_rope            = cache.k_rope;
        auto& pos               = cache.pos;
        auto& attn_val          = cache.attn_val;
        auto& attn_out          = cache.attn_out;
        auto& swiglu_out        = cache.swiglu_out;
        auto& gate_out          = cache.gate_out;
        auto& up_out            = cache.up_out;
        auto& norm_out          = cache.norm_out;
        auto& logits            = cache.logits;
        auto& next_token_id     = cache.next_token_id;
        auto& next_token_possibility = cache.next_token_possibility;
        tensor_t k;
        tensor_t v;
        // input embedding
        tensor_input_ids->load(input_ids.data());
        embedding(x, tensor_input_ids, weights_.in_embed);
        
        for (size_t i = 0; i < meta_.nlayer; i++) {
            rms_norm(x_norm, x, weights_.attn_norm_w[i], meta_.epsilon);
            //compute q, k, v
            linear(q, x_norm, weights_.attn_q_w[i], weights_.attn_q_b[i]);
            linear(k_, x_norm, weights_.attn_k_w[i], weights_.attn_k_b[i]);
            linear(v_, x_norm, weights_.attn_v_w[i], weights_.attn_v_b[i]);
            //rope
            rope(q_rope, q, pos, meta_.theta);
            rope(k_rope, k_, pos, meta_.theta);
            //cache and load k, v
            kv_cache_.add(i, k_rope, v_, seq_len);
            k = kv_cache_.k(i);
            v = kv_cache_.v(i);
            //attention
            self_attention(
                attn_val,
                q_rope, k, v,
                1.0f / std::sqrt(utils::cast<float>(meta_.dh))
            );
            linear(attn_out, attn_val, weights_.attn_o_w[i]);
            add(x, x, attn_out);
            rms_norm(x_norm, x, weights_.mlp_norm_w[i], meta_.epsilon);
            //FFN
            linear(gate_out, x_norm, weights_.mlp_gate_w[i]);
            linear(up_out, x_norm, weights_.mlp_up_w[i]);
            swiglu(swiglu_out, gate_out, up_out);
            linear(x_norm, swiglu_out, weights_.mlp_down_w[i]);
            add(x, x, x_norm);
        }
        rms_norm(norm_out, x->slice(0, seq_len - 1, seq_len), weights_.out_norm_w, meta_.epsilon);
        linear(logits, norm_out, weights_.out_embed);
        argmax(next_token_id, next_token_possibility, logits);
        auto ret = *reinterpret_cast<int64_t*>(next_token_id->data());

        //debug
        debug::print("Next token id: ", ret, " possibility: ", utils::cast<float>(*reinterpret_cast<llaisys::bf16_t*>(next_token_possibility->data())));

        return ret;
    }


    const Qwen2Weights& weights() const {
        return weights_;
    }
private:
    struct Infer_tensors_buf {
        tensor_t tensor_input_ids;
        tensor_t x;
        tensor_t x_norm;
        tensor_t q;
        tensor_t k_;
        tensor_t v_;
        tensor_t q_rope;
        tensor_t k_rope;
        tensor_t pos;
        tensor_t attn_val;
        tensor_t attn_out;
        tensor_t swiglu_out;
        tensor_t gate_out;
        tensor_t up_out;
        tensor_t norm_out;
        tensor_t logits;
        tensor_t next_token_id;
        tensor_t next_token_possibility;
        size_t seq_len = 0;
        size_t total_len = 0;
    };
    Infer_tensors_buf get_(size_t seq_len) {
        infer_buf_.total_len += seq_len;
        if (!infer_buf_.norm_out) {
            infer_buf_.norm_out               = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.logits                 = Tensor::create({meta_.voc}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.next_token_id          = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_ids_[0]);
            infer_buf_.next_token_possibility = Tensor::create({1}, meta_.dtype, device_, device_ids_[0]);
        }
        
        if (!infer_buf_.pos || infer_buf_.total_len > infer_buf_.pos->shape()[0]) {
            infer_buf_.pos = Tensor::create({infer_buf_.total_len * 2}, LLAISYS_DTYPE_I64, device_, device_ids_[0]);
            auto p = reinterpret_cast<int64_t*>(infer_buf_.pos->data());
            for (size_t i = 0; i < infer_buf_.pos->numel(); i++) {
                *p++ = i;
            }
        }
        Infer_tensors_buf cache;
        if (infer_buf_.seq_len < seq_len) {
            infer_buf_.tensor_input_ids  = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_, device_ids_[0]);
            infer_buf_.x                 = Tensor::create({seq_len, meta_.hs}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.x_norm            = Tensor::create({seq_len, meta_.hs}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.q                 = Tensor::create({seq_len, meta_.nh, meta_.dh}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.k_                = Tensor::create({seq_len, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.v_                = Tensor::create({seq_len, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.q_rope            = Tensor::create({seq_len, meta_.nh, meta_.dh}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.k_rope            = Tensor::create({seq_len, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.attn_val          = Tensor::create({seq_len, meta_.hs}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.attn_out          = Tensor::create({seq_len, meta_.hs}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.swiglu_out        = Tensor::create({seq_len, meta_.di}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.gate_out          = Tensor::create({seq_len, meta_.di}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.up_out            = Tensor::create({seq_len, meta_.di}, meta_.dtype, device_, device_ids_[0]);
            infer_buf_.seq_len = seq_len;
            cache = infer_buf_;
        }else {
            cache.tensor_input_ids = infer_buf_.tensor_input_ids->slice(0, 0, seq_len);
            cache.x = infer_buf_.x->slice(0, 0, seq_len);
            cache.x_norm = infer_buf_.x_norm->slice(0, 0, seq_len);
            cache.q = infer_buf_.q->slice(0, 0, seq_len);
            cache.k_ = infer_buf_.k_->slice(0, 0, seq_len);
            cache.v_ = infer_buf_.v_->slice(0, 0, seq_len);
            cache.q_rope = infer_buf_.q_rope->slice(0, 0, seq_len);
            cache.k_rope = infer_buf_.k_rope->slice(0, 0, seq_len);
            cache.attn_val = infer_buf_.attn_val->slice(0, 0, seq_len);
            cache.attn_out = infer_buf_.attn_out->slice(0, 0, seq_len);
            cache.swiglu_out = infer_buf_.swiglu_out->slice(0, 0, seq_len);
            cache.gate_out = infer_buf_.gate_out->slice(0, 0, seq_len);
            cache.up_out = infer_buf_.up_out->slice(0, 0, seq_len);
        }
        cache.pos = infer_buf_.pos->slice(0, infer_buf_.total_len - seq_len, infer_buf_.total_len);
        cache.norm_out = infer_buf_.norm_out;
        cache.logits = infer_buf_.logits;
        cache.next_token_id = infer_buf_.next_token_id;
        cache.next_token_possibility = infer_buf_.next_token_possibility;
        return cache;
    }
private:
    LlaisysQwen2Meta meta_;
    Qwen2Weights weights_;
    llaisysDeviceType_t device_;
    std::vector<int> device_ids_;
    Kv_cache kv_cache_;
    Infer_tensors_buf infer_buf_;
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        auto impl = new Qwen2ModelImpl(*meta, device, std::vector<int>(device_ids, device_ids + ndevice));
        auto model = new LlaisysQwen2Model;
        model->impl = impl;
        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        delete model->meta;
        delete model->weights;
        delete static_cast<Qwen2ModelImpl*>(model->impl);
        delete model;
    }

    void llaisysQwen2modelLoadWeight(LlaisysQwen2Model * model, const void *weight_data, const char *weight_name) {
        auto impl = static_cast<Qwen2ModelImpl*>(model->impl);
        impl->loadWeight(weight_data, weight_name);
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        delete model->weights; // Free previously allocated weights if any
        auto weights =  reinterpret_cast<Qwen2ModelImpl*>(model->impl)->weights();
        model->weights = new LlaisysQwen2Weights{
            reinterpret_cast<llaisysTensor_t>(weights.in_embed.get()),
            reinterpret_cast<llaisysTensor_t>(weights.out_embed.get()),
            reinterpret_cast<llaisysTensor_t>(weights.out_norm_w.get()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_norm_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_q_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_q_b.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_k_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_k_b.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_v_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_v_b.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.attn_o_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.mlp_norm_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.mlp_gate_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.mlp_up_w.data()),
            reinterpret_cast<llaisysTensor_t*>(weights.mlp_down_w.data())
        };
        return model->weights;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        auto impl = static_cast<Qwen2ModelImpl*>(model->impl);
        return impl->forward_with_cache(std::vector<int64_t>(token_ids,token_ids + ntoken));
    }
}