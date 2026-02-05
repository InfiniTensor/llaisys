#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "llaisys.h"
#include "llaisys/runtime.h"
#include "llaisys/ops.h"

#include <vector>
#include <cmath>
#include <string>

class LlaisysQwen2Model {
public:
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    std::vector<llaisysTensor_t> k_cache;
    std::vector<llaisysTensor_t> v_cache;
    const LlaisysRuntimeAPI *runtime_api;
    llaisysDeviceType_t device;
    int device_id;

    // 构造函数
    LlaisysQwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id)
        : meta(*meta), device(device), device_id(device_id) {
        this->runtime_api = llaisysGetRuntimeAPI(device);
        llaisysSetContextRuntime(device, this->device_id);
        createGlobalWeights(this->weights, meta, device, this->device_id);
        createLayerWeights(this->weights, meta, device, this->device_id);
        this->k_cache.resize(meta->nlayer);
        this->v_cache.resize(meta->nlayer);
        size_t shape_kv[3] = {meta->maxseq, meta->nkvh, meta->dh};
        for (size_t i = 0; i < meta->nlayer; ++i) {
            this->k_cache[i] = tensorCreate(shape_kv, 3, meta->dtype, device, this->device_id);
            this->v_cache[i] = tensorCreate(shape_kv, 3, meta->dtype, device, this->device_id);
        }
    }

    // 析构函数
    ~LlaisysQwen2Model() {
        // 释放全局权重
        tensorDestroy(weights.in_embed);
        tensorDestroy(weights.out_embed);
        tensorDestroy(weights.out_norm_w);

        // 释放层权重
        for (auto &t : weights.attn_norm_w) tensorDestroy(t);
        for (auto &t : weights.attn_q_w) tensorDestroy(t);
        for (auto &t : weights.attn_q_b) tensorDestroy(t);
        for (auto &t : weights.attn_k_w) tensorDestroy(t);
        for (auto &t : weights.attn_k_b) tensorDestroy(t);
        for (auto &t : weights.attn_v_w) tensorDestroy(t);
        for (auto &t : weights.attn_v_b) tensorDestroy(t);
        for (auto &t : weights.attn_o_w) tensorDestroy(t);
        for (auto &t : weights.mlp_norm_w) tensorDestroy(t);
        for (auto &t : weights.mlp_gate_w) tensorDestroy(t);
        for (auto &t : weights.mlp_up_w) tensorDestroy(t);
        for (auto &t : weights.mlp_down_w) tensorDestroy(t);

        // 释放 KV cache
        for (auto &t : k_cache) tensorDestroy(t);
        for (auto &t : v_cache) tensorDestroy(t);
    }

    // 推理方法
    int64_t infer(int64_t *token_ids, size_t ntoken, size_t start_pos = 0) {
        size_t hs = meta.hs;
        size_t head_dim = hs / meta.nh;
        size_t kv_dim = head_dim * meta.nkvh;

        // 创建输入张量
        size_t input_shape[1] = {ntoken};
        llaisysTensor_t input_tensor = tensorCreate(input_shape, 1, LLAISYS_DTYPE_I64, device, device_id);

        // 加载 token_ids
        tensorLoad(input_tensor, token_ids);

        size_t hidden_shape[2] = {ntoken, hs};
        llaisysTensor_t hidden_states = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);

        // Embedding
        llaisysEmbedding(hidden_states, input_tensor, weights.in_embed);

        // 位置 IDs
        size_t pos_shape[1] = {ntoken};
        llaisysTensor_t pos_ids = tensorCreate(pos_shape, 1, LLAISYS_DTYPE_I64, device, device_id);
        std::vector<int64_t> pos_vec(ntoken);
        for (size_t i = 0; i < ntoken; ++i) pos_vec[i] = start_pos + i;
        tensorLoad(pos_ids, pos_vec.data());

        for (size_t i = 0; i < meta.nlayer; ++i) {
            // RMS Norm
            llaisysTensor_t norm_out = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);
            llaisysRmsNorm(norm_out, hidden_states, weights.attn_norm_w[i], meta.epsilon);

            // Q, K, V
            size_t q_shape_2d[2] = {ntoken, hs};
            size_t kv_shape_2d[2] = {ntoken, kv_dim};
            llaisysTensor_t q2d = tensorCreate(q_shape_2d, 2, meta.dtype, device, device_id);
            llaisysTensor_t k2d = tensorCreate(kv_shape_2d, 2, meta.dtype, device, device_id);
            llaisysTensor_t v2d = tensorCreate(kv_shape_2d, 2, meta.dtype, device, device_id);

            llaisysLinear(q2d, norm_out, weights.attn_q_w[i], weights.attn_q_b[i]);
            llaisysLinear(k2d, norm_out, weights.attn_k_w[i], weights.attn_k_b[i]);
            llaisysLinear(v2d, norm_out, weights.attn_v_w[i], weights.attn_v_b[i]);

            size_t q_shape[3] = {ntoken, meta.nh, head_dim};
            size_t kv_shape[3] = {ntoken, meta.nkvh, head_dim};
            llaisysTensor_t q = tensorView(q2d, q_shape, 3);
            llaisysTensor_t k = tensorView(k2d, kv_shape, 3);
            llaisysTensor_t v = tensorView(v2d, kv_shape, 3);

            // RoPE
            llaisysROPE(q, q, pos_ids, meta.theta);
            llaisysROPE(k, k, pos_ids, meta.theta);

            // KV Cache
            llaisysTensor_t layer_k_cache = k_cache[i];
            llaisysTensor_t layer_v_cache = v_cache[i];

            llaisysTensor_t k_slot = tensorSlice(layer_k_cache, 0, start_pos, start_pos + ntoken);
            llaisysTensor_t v_slot = tensorSlice(layer_v_cache, 0, start_pos, start_pos + ntoken);
            tensorLoad(k_slot, tensorGetData(k2d));
            tensorLoad(v_slot, tensorGetData(v2d));

            llaisysTensor_t full_k = tensorSlice(layer_k_cache, 0, 0, start_pos + ntoken);
            llaisysTensor_t full_v = tensorSlice(layer_v_cache, 0, 0, start_pos + ntoken);

            // Self Attention
            size_t attn_shape[3] = {ntoken, meta.nh, head_dim};
            llaisysTensor_t attn_out = tensorCreate(attn_shape, 3, meta.dtype, device, device_id);
            float scale = 1.0f / sqrt(head_dim);
            llaisysSelfAttention(attn_out, q, full_k, full_v, scale);

            // Proj
            llaisysTensor_t attn_out_2d = tensorView(attn_out, hidden_shape, 2);
            llaisysTensor_t proj_out = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);
            llaisysLinear(proj_out, attn_out_2d, weights.attn_o_w[i], nullptr);

            llaisysAdd(hidden_states, hidden_states, proj_out);

            // FFN
            llaisysTensor_t ffn_norm_out = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);
            llaisysRmsNorm(ffn_norm_out, hidden_states, weights.mlp_norm_w[i], meta.epsilon);

            size_t inter_size = meta.di;
            size_t inter_shape[2] = {ntoken, inter_size};
            llaisysTensor_t gate = tensorCreate(inter_shape, 2, meta.dtype, device, device_id);
            llaisysTensor_t up = tensorCreate(inter_shape, 2, meta.dtype, device, device_id);

            llaisysLinear(gate, ffn_norm_out, weights.mlp_gate_w[i], nullptr);
            llaisysLinear(up, ffn_norm_out, weights.mlp_up_w[i], nullptr);

            llaisysTensor_t act = tensorCreate(inter_shape, 2, meta.dtype, device, device_id);
            llaisysSwiGLU(act, gate, up);

            llaisysTensor_t mlp_out = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);
            llaisysLinear(mlp_out, act, weights.mlp_down_w[i], nullptr);

            llaisysAdd(hidden_states, hidden_states, mlp_out);

            tensorDestroy(norm_out);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q);
            tensorDestroy(k);
            tensorDestroy(v);
            // layer_k_cache/layer_v_cache are owned by model
            tensorDestroy(k_slot);
            tensorDestroy(v_slot);
            tensorDestroy(full_k);
            tensorDestroy(full_v);
            tensorDestroy(attn_out);
            tensorDestroy(attn_out_2d);
            tensorDestroy(proj_out);
            tensorDestroy(ffn_norm_out);
            tensorDestroy(gate);
            tensorDestroy(up);
            tensorDestroy(act);
            tensorDestroy(mlp_out);
        }

        // Final Norm
        llaisysTensor_t final_norm = tensorCreate(hidden_shape, 2, meta.dtype, device, device_id);
        llaisysRmsNorm(final_norm, hidden_states, weights.out_norm_w, meta.epsilon);

        // Logits
        size_t logits_shape[2] = {ntoken, meta.voc};
        llaisysTensor_t logits = tensorCreate(logits_shape, 2, meta.dtype, device, device_id);
        llaisysLinear(logits, final_norm, weights.out_embed, nullptr);

        // Argmax on last token
        llaisysTensor_t last_token_logits = tensorSlice(logits, 0, ntoken - 1, ntoken);
        llaisysTensor_t final_logits = last_token_logits;
        size_t max_shape[1] = {1};
        llaisysTensor_t max_idx = tensorCreate(max_shape, 1, LLAISYS_DTYPE_I64, device, device_id);
        llaisysTensor_t max_val = tensorCreate(max_shape, 1, meta.dtype, device, device_id);
        llaisysArgmax(max_idx, max_val, final_logits);

        int64_t result = *reinterpret_cast<int64_t*>(tensorGetData(max_idx));

        // 释放临时张量
        tensorDestroy(input_tensor);
        tensorDestroy(hidden_states);
        tensorDestroy(pos_ids);
        tensorDestroy(final_norm);
        tensorDestroy(logits);
        tensorDestroy(last_token_logits);
        tensorDestroy(max_idx);
        tensorDestroy(max_val);

        return result;
    }

private:
    // 辅助函数：创建全局权重
    static void createGlobalWeights(LlaisysQwen2Weights &weights, const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id) {
        size_t shape_in_embed[2] = {meta->voc, meta->hs};
        weights.in_embed = tensorCreate(shape_in_embed, 2, meta->dtype, device, device_id);

        size_t shape_out_embed[2] = {meta->voc, meta->hs};
        weights.out_embed = tensorCreate(shape_out_embed, 2, meta->dtype, device, device_id);

        size_t shape_out_norm[1] = {meta->hs};
        weights.out_norm_w = tensorCreate(shape_out_norm, 1, meta->dtype, device, device_id);
    }

    // 辅助函数：创建层权重
    static void createLayerWeights(LlaisysQwen2Weights &weights, const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id) {
        size_t n = meta->nlayer;
        weights.attn_norm_w.resize(n);
        weights.attn_q_w.resize(n);
        weights.attn_q_b.resize(n);
        weights.attn_k_w.resize(n);
        weights.attn_k_b.resize(n);
        weights.attn_v_w.resize(n);
        weights.attn_v_b.resize(n);
        weights.attn_o_w.resize(n);
        weights.mlp_norm_w.resize(n);
        weights.mlp_gate_w.resize(n);
        weights.mlp_up_w.resize(n);
        weights.mlp_down_w.resize(n);

        for (size_t i = 0; i < n; ++i) {
            size_t shape_norm[1] = {meta->hs};
            weights.attn_norm_w[i] = tensorCreate(shape_norm, 1, meta->dtype, device, device_id);
            size_t shape_q[2] = {meta->nh * meta->dh, meta->hs};
            weights.attn_q_w[i] = tensorCreate(shape_q, 2, meta->dtype, device, device_id);
            size_t shape_qb[1] = {meta->nh * meta->dh};
            weights.attn_q_b[i] = tensorCreate(shape_qb, 1, meta->dtype, device, device_id);
            size_t shape_k[2] = {meta->nkvh * meta->dh, meta->hs};
            weights.attn_k_w[i] = tensorCreate(shape_k, 2, meta->dtype, device, device_id);
            size_t shape_kb[1] = {meta->nkvh * meta->dh};
            weights.attn_k_b[i] = tensorCreate(shape_kb, 1, meta->dtype, device, device_id);
            weights.attn_v_w[i] = tensorCreate(shape_k, 2, meta->dtype, device, device_id);
            weights.attn_v_b[i] = tensorCreate(shape_kb, 1, meta->dtype, device, device_id);
            size_t shape_o[2] = {meta->hs, meta->hs};
            weights.attn_o_w[i] = tensorCreate(shape_o, 2, meta->dtype, device, device_id);
            weights.mlp_norm_w[i] = tensorCreate(shape_norm, 1, meta->dtype, device, device_id);
            size_t shape_gate[2] = {meta->di, meta->hs};
            weights.mlp_gate_w[i] = tensorCreate(shape_gate, 2, meta->dtype, device, device_id);
            weights.mlp_up_w[i] = tensorCreate(shape_gate, 2, meta->dtype, device, device_id);
            size_t shape_down[2] = {meta->hs, meta->di};
            weights.mlp_down_w[i] = tensorCreate(shape_down, 2, meta->dtype, device, device_id);
        }
    }
};

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    return new LlaisysQwen2Model(meta, device, device_ids[0]);
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

static llaisysTensor_t resolve_weight_tensor(LlaisysQwen2Model *model, const std::string &name) {
    if (name == "model.embed_tokens.weight") return model->weights.in_embed;
    if (name == "lm_head.weight") return model->weights.out_embed;
    if (name == "model.norm.weight") return model->weights.out_norm_w;

    const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) == 0) {
        size_t idx_start = prefix.size();
        size_t idx_end = name.find('.', idx_start);
        if (idx_end == std::string::npos) return nullptr;
        size_t layer = static_cast<size_t>(std::stoul(name.substr(idx_start, idx_end - idx_start)));
        if (layer >= model->meta.nlayer) return nullptr;
        std::string suffix = name.substr(idx_end + 1);

        if (suffix == "input_layernorm.weight") return model->weights.attn_norm_w[layer];
        if (suffix == "post_attention_layernorm.weight") return model->weights.mlp_norm_w[layer];

        if (suffix == "self_attn.q_proj.weight") return model->weights.attn_q_w[layer];
        if (suffix == "self_attn.q_proj.bias") return model->weights.attn_q_b[layer];
        if (suffix == "self_attn.k_proj.weight") return model->weights.attn_k_w[layer];
        if (suffix == "self_attn.k_proj.bias") return model->weights.attn_k_b[layer];
        if (suffix == "self_attn.v_proj.weight") return model->weights.attn_v_w[layer];
        if (suffix == "self_attn.v_proj.bias") return model->weights.attn_v_b[layer];
        if (suffix == "self_attn.o_proj.weight") return model->weights.attn_o_w[layer];

        if (suffix == "mlp.gate_proj.weight") return model->weights.mlp_gate_w[layer];
        if (suffix == "mlp.up_proj.weight") return model->weights.mlp_up_w[layer];
        if (suffix == "mlp.down_proj.weight") return model->weights.mlp_down_w[layer];
    }

    return nullptr;
}

__export void llaisysQwen2LoadWeight(
    struct LlaisysQwen2Model *model,
    const char *name,
    const void *data,
    size_t *shape,
    size_t ndim,
    llaisysDataType_t dtype) {
    if (!model || !name || !data) return;
    llaisysTensor_t tensor = resolve_weight_tensor(model, std::string(name));
    if (!tensor) return;
    if (tensorGetDataType(tensor) != dtype) {
        return;
    }
    tensorLoad(tensor, data);
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, size_t start_pos) {
    return model->infer(token_ids, ntoken, start_pos);
}
