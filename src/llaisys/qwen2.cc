#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "llaisys_tensor.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <regex>
#include <vector>
#include <iostream>
#include "llaisys/ops.h"
#include "../core/llaisys_core.hpp"

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    LlaisysQwen2Weights weights;
    // store arbitrary named tensors provided from python
    std::unordered_map<std::string, llaisysTensor_t> weight_map;
    // simple KV cache pointer (optional)
    // not owning tensors, just storing handles
    struct KVCache *kv_cache = nullptr;
};

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta) return nullptr;
    LlaisysQwen2Model *m = new LlaisysQwen2Model();
    m->meta = *meta;
    m->device_type = device;
    m->device_id = (device_ids != nullptr && ndevice > 0) ? device_ids[0] : 0;
    memset(&m->weights, 0, sizeof(m->weights));
    return m;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;
    if (model->kv_cache) {
        llaisysQwen2KVDestroy((void *)model->kv_cache);
        model->kv_cache = nullptr;
    }
    model->weight_map.clear();
    if (model->weights.attn_norm_w) delete[] model->weights.attn_norm_w;
    if (model->weights.attn_q_w) delete[] model->weights.attn_q_w;
    if (model->weights.attn_q_b) delete[] model->weights.attn_q_b;
    if (model->weights.attn_k_w) delete[] model->weights.attn_k_w;
    if (model->weights.attn_k_b) delete[] model->weights.attn_k_b;
    if (model->weights.attn_v_w) delete[] model->weights.attn_v_w;
    if (model->weights.attn_v_b) delete[] model->weights.attn_v_b;
    if (model->weights.attn_o_w) delete[] model->weights.attn_o_w;
    if (model->weights.mlp_norm_w) delete[] model->weights.mlp_norm_w;
    if (model->weights.mlp_gate_w) delete[] model->weights.mlp_gate_w;
    if (model->weights.mlp_up_w) delete[] model->weights.mlp_up_w;
    if (model->weights.mlp_down_w) delete[] model->weights.mlp_down_w;
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return &model->weights;
}

static bool str_contains_any(const std::string &s, const std::vector<std::string> &subs) {
    for (auto &p : subs) if (s.find(p) != std::string::npos) return true;
    return false;
}

static bool tensor_matches_shape_and_dtype(llaisysTensor_t t, const std::vector<size_t> &expected_shape, llaisysDataType_t dtype) {
    if (!t) return false;
    try {
        auto &s = t->tensor->shape();
        if (s.size() != expected_shape.size()) return false;
        for (size_t i = 0; i < s.size(); ++i) if (s[i] != expected_shape[i]) return false;
        if (t->tensor->dtype() != dtype) return false;
        return true;
    } catch (...) {
        return false;
    }
}

static void ensure_layer_weight_arrays(LlaisysQwen2Model *model) {
    if (!model || model->meta.nlayer == 0) {
        return;
    }

    size_t n = model->meta.nlayer;
    if (!model->weights.attn_norm_w) model->weights.attn_norm_w = new llaisysTensor_t[n]();
    if (!model->weights.attn_q_w) model->weights.attn_q_w = new llaisysTensor_t[n]();
    if (!model->weights.attn_q_b) model->weights.attn_q_b = new llaisysTensor_t[n]();
    if (!model->weights.attn_k_w) model->weights.attn_k_w = new llaisysTensor_t[n]();
    if (!model->weights.attn_k_b) model->weights.attn_k_b = new llaisysTensor_t[n]();
    if (!model->weights.attn_v_w) model->weights.attn_v_w = new llaisysTensor_t[n]();
    if (!model->weights.attn_v_b) model->weights.attn_v_b = new llaisysTensor_t[n]();
    if (!model->weights.attn_o_w) model->weights.attn_o_w = new llaisysTensor_t[n]();
    if (!model->weights.mlp_norm_w) model->weights.mlp_norm_w = new llaisysTensor_t[n]();
    if (!model->weights.mlp_gate_w) model->weights.mlp_gate_w = new llaisysTensor_t[n]();
    if (!model->weights.mlp_up_w) model->weights.mlp_up_w = new llaisysTensor_t[n]();
    if (!model->weights.mlp_down_w) model->weights.mlp_down_w = new llaisysTensor_t[n]();
}

__export int llaisysQwen2ModelSetWeight(struct LlaisysQwen2Model * model, const char * name, llaisysTensor_t tensor) {
    if (!model || !name) return -1;
    ensure_layer_weight_arrays(model);
    std::string sname(name);
    model->weight_map[sname] = tensor;

    // top-level mapping heuristics
    if (str_contains_any(sname, {"embed_tokens", "word_embeddings", "tok_embeddings", "token_embedding", "embed."})) {
        std::vector<size_t> exp = {model->meta.voc, model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.in_embed = tensor;
        else std::cerr << "[llaisys qwen2] Warning: in_embed shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }
    if (str_contains_any(sname, {"lm_head", "output_projection", "out_proj", "out_embed", "head"})) {
        std::vector<size_t> exp = {model->meta.voc, model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.out_embed = tensor;
        else std::cerr << "[llaisys qwen2] Warning: out_embed shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }
    if (sname == "norm.weight" || str_contains_any(sname, {"model.norm.weight", "ln_f.weight", "final_layernorm.weight", "out_norm.weight"})) {
        // avoid matching layer-norm inside layers by checking common patterns
        std::vector<size_t> exp = {model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.out_norm_w = tensor;
        else std::cerr << "[llaisys qwen2] Warning: out_norm_w shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }

    // per-layer: try to find layer index using several common patterns
    std::smatch m;
    std::vector<std::regex> layer_patterns = {
        std::regex("layers\\.(\\d+)"),
        std::regex("blocks\\.(\\d+)"),
        std::regex("h\\.(\\d+)")
    };
    int idx = -1;
    for (auto &rp : layer_patterns) {
        if (std::regex_search(sname, m, rp)) { idx = std::stoi(m[1].str()); break; }
    }
    if (idx >= 0 && (size_t)idx < model->meta.nlayer) {
        // per-layer mapping with shape/dtype validation
        size_t hs = model->meta.hs;
        size_t nh = model->meta.nh;
        size_t nkv = model->meta.nkvh;
        size_t dh = model->meta.dh;
        size_t di = model->meta.di;

        // post-attention norm
        if (str_contains_any(sname, {"post_attention_layernorm.weight", "ln_2.weight", "layernorm_after.weight"})) {
            std::vector<size_t> exp = {hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_norm_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_norm_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // input layer norm weight [hs]
        if (str_contains_any(sname, {"input_layernorm.weight", "ln_1.weight", "layernorm_before.weight"}) ||
            (sname.find("attention_layernorm.weight") != std::string::npos && sname.find("post_attention_layernorm.weight") == std::string::npos)) {
            std::vector<size_t> exp = {hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_norm_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_norm_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // q projection [nh*dh, hs], bias [nh*dh]
        if (str_contains_any(sname, {"q_proj.weight", "q.weight", "qkv.q.weight", "wq.weight", "attn.q.weight"})) {
            std::vector<size_t> exp = {nh * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_q_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_q_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"q_proj.bias", "q.bias", "wq.bias"})) {
            std::vector<size_t> exp = {nh * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_q_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_q_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // k projection [nkv*dh, hs], bias [nkv*dh]
        if (str_contains_any(sname, {"k_proj.weight", "k.weight", "wk.weight", "attn.k.weight"})) {
            std::vector<size_t> exp = {nkv * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_k_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_k_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"k_proj.bias", "k.bias", "wk.bias"})) {
            std::vector<size_t> exp = {nkv * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_k_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_k_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // v projection [nkv*dh, hs], bias [nkv*dh]
        if (str_contains_any(sname, {"v_proj.weight", "v.weight", "wv.weight", "attn.v.weight"})) {
            std::vector<size_t> exp = {nkv * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_v_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_v_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"v_proj.bias", "v.bias", "wv.bias"})) {
            std::vector<size_t> exp = {nkv * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_v_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_v_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // out projection (attention) [hs, nh*dh]
        if (str_contains_any(sname, {"o_proj.weight", "o.weight", "wo.weight", "attn.o.weight"})) {
            std::vector<size_t> exp = {hs, nh * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_o_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_o_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // MLP gate/up/down
        if (str_contains_any(sname, {"mlp.gate_proj.weight", "gate_proj.weight", "dense_h_to_4h.weight", "gate.weight"})) {
            std::vector<size_t> exp = {di, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_gate_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_gate_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"mlp.up_proj.weight", "up_proj.weight", "up.weight"})) {
            std::vector<size_t> exp = {di, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_up_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_up_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"mlp.down_proj.weight", "down_proj.weight", "down.weight"})) {
            std::vector<size_t> exp = {hs, di};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_down_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_down_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
    }

    return 0;
}

__export int llaisysQwen2ModelFinalize(struct LlaisysQwen2Model * model) {
    if (!model) return -1;
    size_t n = model->meta.nlayer;
    if (n > 0) {
        if (!model->weights.attn_norm_w) model->weights.attn_norm_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_q_w) model->weights.attn_q_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_q_b) model->weights.attn_q_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_k_w) model->weights.attn_k_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_k_b) model->weights.attn_k_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_v_w) model->weights.attn_v_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_v_b) model->weights.attn_v_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_o_w) model->weights.attn_o_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_norm_w) model->weights.mlp_norm_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_gate_w) model->weights.mlp_gate_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_up_w) model->weights.mlp_up_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_down_w) model->weights.mlp_down_w = new llaisysTensor_t[n]();
    }

    // quick validation: warn about missing critical weights
    std::vector<std::string> missing;
    if (!model->weights.in_embed) missing.push_back("in_embed");
    if (!model->weights.out_norm_w) missing.push_back("out_norm_w");
    if (!model->weights.out_embed) missing.push_back("out_embed");
    if (n > 0) {
        for (size_t i = 0; i < n; ++i) {
            size_t hs = model->meta.hs;
            size_t nh = model->meta.nh;
            size_t nkv = model->meta.nkvh;
            size_t dh = model->meta.dh;
            size_t di = model->meta.di;

            std::vector<size_t> qw = {nh * dh, hs};
            std::vector<size_t> kw = {nkv * dh, hs};
            std::vector<size_t> vw = {nkv * dh, hs};
            std::vector<size_t> ow = {hs, nh * dh};
            std::vector<size_t> normv = {hs};
            std::vector<size_t> gatew = {di, hs};
            std::vector<size_t> downw = {hs, di};

            if (!model->weights.attn_q_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_q_w[i], qw, model->meta.dtype)) missing.push_back("attn_q_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_k_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_k_w[i], kw, model->meta.dtype)) missing.push_back("attn_k_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_v_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_v_w[i], vw, model->meta.dtype)) missing.push_back("attn_v_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_o_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_o_w[i], ow, model->meta.dtype)) missing.push_back("attn_o_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_norm_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_norm_w[i], normv, model->meta.dtype)) missing.push_back("mlp_norm_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_gate_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_gate_w[i], gatew, model->meta.dtype)) missing.push_back("mlp_gate_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_down_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_down_w[i], downw, model->meta.dtype)) missing.push_back("mlp_down_w[" + std::to_string(i) + "]");
        }
    }
    if (!missing.empty()) {
        std::cerr << "[llaisys qwen2] Warning: missing weights:";
        for (auto &s : missing) std::cerr << " " << s;
        std::cerr << std::endl;
    }

    return 0;
}

// Simple KV cache structure and APIs
struct KVCache {
    LlaisysQwen2Model *owner = nullptr;
    size_t max_tokens = 0;
    size_t nlayer = 0;
    std::vector<std::vector<llaisysTensor_t>> keys;
    std::vector<std::vector<llaisysTensor_t>> vals;
};

static llaisysTensor_t make_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    return new LlaisysTensor{llaisys::Tensor::create(shape, dtype, device, device_id)};
}

static void clear_layer_cache(std::vector<llaisysTensor_t> &cache) {
    for (auto *t : cache) {
        delete t;
    }
    cache.clear();
}

static void kv_append_for_layer(KVCache *kv, size_t layer_idx, llaisysTensor_t k, llaisysTensor_t v) {
    if (!kv || layer_idx >= kv->nlayer || !k || !v) {
        return;
    }
    auto &ks = kv->keys[layer_idx];
    auto &vs = kv->vals[layer_idx];

    if (kv->max_tokens > 0 && ks.size() >= kv->max_tokens) {
        delete ks.front();
        ks.erase(ks.begin());
        delete vs.front();
        vs.erase(vs.begin());
    }

    ks.push_back(k);
    vs.push_back(v);
}

__export void *llaisysQwen2KVCreat(struct LlaisysQwen2Model * model, size_t max_tokens) {
    if (!model) return nullptr;

    if (model->kv_cache) {
        llaisysQwen2KVDestroy((void *)model->kv_cache);
    }

    KVCache *kv = new KVCache();
    kv->owner = model;
    kv->max_tokens = max_tokens;
    kv->nlayer = model->meta.nlayer;
    kv->keys.resize(kv->nlayer);
    kv->vals.resize(kv->nlayer);
    for (size_t i = 0; i < kv->nlayer; ++i) {
        kv->keys[i].reserve(max_tokens);
        kv->vals[i].reserve(max_tokens);
    }
    model->kv_cache = kv;
    return (void *)kv;
}

__export void llaisysQwen2KVDestroy(void *kv) {
    if (!kv) return;
    KVCache *c = (KVCache *)kv;
    for (size_t i = 0; i < c->nlayer; ++i) {
        clear_layer_cache(c->keys[i]);
        clear_layer_cache(c->vals[i]);
    }
    c->keys.clear();
    c->vals.clear();
    if (c->owner && c->owner->kv_cache == c) {
        c->owner->kv_cache = nullptr;
    }
    delete c;
}

__export int llaisysQwen2KVAppend(void *kv, llaisysTensor_t k, llaisysTensor_t v) {
    if (!kv) return -1;
    KVCache *c = (KVCache *)kv;
    if (c->nlayer == 0) return -1;
    kv_append_for_layer(c, 0, k, v);
    return 0;
}

__export size_t llaisysQwen2KVLen(void *kv) {
    if (!kv) return 0;
    KVCache *c = (KVCache *)kv;
    if (c->nlayer == 0) return 0;
    return c->keys[0].size();
}

__export uint8_t llaisysQwen2ModelHasWeight(struct LlaisysQwen2Model * model, const char * name) {
    if (!model || !name) return 0;
    std::string sname(name);
    auto it = model->weight_map.find(sname);
    return it != model->weight_map.end() ? 1 : 0;
}

static int64_t read_scalar_i64(llaisysTensor_t tensor, const LlaisysRuntimeAPI *runtime_api, int64_t fallback) {
    if (!tensor) {
        return fallback;
    }
    if (tensor->tensor->deviceType() == LLAISYS_DEVICE_CPU) {
        int64_t *ptr = reinterpret_cast<int64_t *>(tensor->tensor->data());
        return ptr ? ptr[0] : fallback;
    }
    int64_t v = fallback;
    runtime_api->memcpy_sync(&v, tensor->tensor->data(), sizeof(v), LLAISYS_MEMCPY_D2H);
    return v;
}

static int64_t infer_one_token(
    struct LlaisysQwen2Model *model,
    int64_t token_id,
    bool do_sample,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    using namespace llaisys;

    if (!model->weights.in_embed || !model->weights.out_embed) {
        return token_id;
    }

    llaisys::core::context().setDevice(model->device_type, model->device_id);
    const LlaisysRuntimeAPI *runtime_api = llaisys::core::context().runtime().api();

    const llaisysDataType_t dtype = model->meta.dtype;
    const size_t hs = model->meta.hs;
    const size_t nh = model->meta.nh;
    const size_t nkvh = model->meta.nkvh;
    const size_t dh = model->meta.dh;
    const size_t di = model->meta.di;
    const std::vector<size_t> one_shape = {1};
    const std::vector<size_t> hidden_shape = {1, hs};
    const std::vector<size_t> q_shape = {1, nh, dh};
    const std::vector<size_t> kv_shape = {1, nkvh, dh};

    auto idx = make_tensor(one_shape, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    idx->tensor->load(&token_id);

    auto x = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
    llaisysEmbedding(x, idx, model->weights.in_embed);

    KVCache *kv = model->kv_cache;

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        bool has_attn = model->weights.attn_norm_w && model->weights.attn_q_w && model->weights.attn_k_w && model->weights.attn_v_w &&
                        model->weights.attn_o_w && model->weights.attn_norm_w[i] && model->weights.attn_q_w[i] &&
                        model->weights.attn_k_w[i] && model->weights.attn_v_w[i] && model->weights.attn_o_w[i];

        if (has_attn) {
            auto norm1 = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
            llaisysRmsNorm(norm1, x, model->weights.attn_norm_w[i], model->meta.epsilon);

            auto qflat = make_tensor({1, nh * dh}, dtype, model->device_type, model->device_id);
            auto kflat = make_tensor({1, nkvh * dh}, dtype, model->device_type, model->device_id);
            auto vflat = make_tensor({1, nkvh * dh}, dtype, model->device_type, model->device_id);

            llaisysLinear(qflat, norm1, model->weights.attn_q_w[i], model->weights.attn_q_b ? model->weights.attn_q_b[i] : nullptr);
            llaisysLinear(kflat, norm1, model->weights.attn_k_w[i], model->weights.attn_k_b ? model->weights.attn_k_b[i] : nullptr);
            llaisysLinear(vflat, norm1, model->weights.attn_v_w[i], model->weights.attn_v_b ? model->weights.attn_v_b[i] : nullptr);

            auto q = new LlaisysTensor{qflat->tensor->view(q_shape)};
            auto k = new LlaisysTensor{kflat->tensor->view(kv_shape)};
            auto v = new LlaisysTensor{vflat->tensor->view(kv_shape)};

            int64_t pos = 0;
            if (kv && i < kv->nlayer) {
                pos = static_cast<int64_t>(kv->keys[i].size());
            }

            auto pos_ids = make_tensor(one_shape, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
            pos_ids->tensor->load(&pos);

            auto q_rope = make_tensor(q_shape, dtype, model->device_type, model->device_id);
            auto k_rope = make_tensor(kv_shape, dtype, model->device_type, model->device_id);
            llaisysROPE(q_rope, q, pos_ids, model->meta.theta);
            llaisysROPE(k_rope, k, pos_ids, model->meta.theta);

            bool saved_in_cache = false;
            if (kv && i < kv->nlayer) {
                kv_append_for_layer(kv, i, k_rope, v);
                saved_in_cache = true;
            }

            std::vector<llaisysTensor_t> local_keys;
            std::vector<llaisysTensor_t> local_vals;
            const std::vector<llaisysTensor_t> *k_src = nullptr;
            const std::vector<llaisysTensor_t> *v_src = nullptr;
            if (saved_in_cache) {
                k_src = &kv->keys[i];
                v_src = &kv->vals[i];
            } else {
                local_keys.push_back(k_rope);
                local_vals.push_back(v);
                k_src = &local_keys;
                v_src = &local_vals;
            }

            const size_t total_len = k_src->size();
            auto k_all = make_tensor({total_len, nkvh, dh}, dtype, model->device_type, model->device_id);
            auto v_all = make_tensor({total_len, nkvh, dh}, dtype, model->device_type, model->device_id);
            const size_t bytes_per_token = nkvh * dh * k_all->tensor->elementSize();
            for (size_t j = 0; j < total_len; ++j) {
                runtime_api->memcpy_sync(k_all->tensor->data() + j * bytes_per_token, (*k_src)[j]->tensor->data(), bytes_per_token, LLAISYS_MEMCPY_D2D);
                runtime_api->memcpy_sync(v_all->tensor->data() + j * bytes_per_token, (*v_src)[j]->tensor->data(), bytes_per_token, LLAISYS_MEMCPY_D2D);
            }

            auto attn_val = make_tensor(q_shape, dtype, model->device_type, model->device_id);
            const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
            llaisysSelfAttention(attn_val, q_rope, k_all, v_all, scale);

            auto attn_flat = new LlaisysTensor{attn_val->tensor->view(hidden_shape)};
            auto attn_out = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
            llaisysLinear(attn_out, attn_flat, model->weights.attn_o_w[i], nullptr);

            auto x_next = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
            llaisysAdd(x_next, x, attn_out);
            delete x;
            x = x_next;

            delete norm1;
            delete qflat;
            delete kflat;
            delete vflat;
            delete q;
            delete k;
            delete pos_ids;
            delete q_rope;
            delete k_all;
            delete v_all;
            delete attn_val;
            delete attn_flat;
            delete attn_out;

            if (!saved_in_cache) {
                delete k_rope;
                delete v;
            }
        }

        bool has_mlp = model->weights.mlp_norm_w && model->weights.mlp_gate_w && model->weights.mlp_up_w && model->weights.mlp_down_w &&
                       model->weights.mlp_norm_w[i] && model->weights.mlp_gate_w[i] && model->weights.mlp_up_w[i] &&
                       model->weights.mlp_down_w[i];

        if (has_mlp) {
            auto norm2 = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
            llaisysRmsNorm(norm2, x, model->weights.mlp_norm_w[i], model->meta.epsilon);

            auto gate = make_tensor({1, di}, dtype, model->device_type, model->device_id);
            auto up = make_tensor({1, di}, dtype, model->device_type, model->device_id);
            auto act = make_tensor({1, di}, dtype, model->device_type, model->device_id);
            auto down = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);

            llaisysLinear(gate, norm2, model->weights.mlp_gate_w[i], nullptr);
            llaisysLinear(up, norm2, model->weights.mlp_up_w[i], nullptr);
            llaisysSwiGLU(act, gate, up);
            llaisysLinear(down, act, model->weights.mlp_down_w[i], nullptr);

            auto x_next = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
            llaisysAdd(x_next, x, down);
            delete x;
            x = x_next;

            delete norm2;
            delete gate;
            delete up;
            delete act;
            delete down;
        }
    }

    llaisysTensor_t logits_in = x;
    llaisysTensor_t out_norm = nullptr;
    if (model->weights.out_norm_w) {
        out_norm = make_tensor(hidden_shape, dtype, model->device_type, model->device_id);
        llaisysRmsNorm(out_norm, x, model->weights.out_norm_w, model->meta.epsilon);
        logits_in = out_norm;
    }

    auto logits = make_tensor({1, model->meta.voc}, dtype, model->device_type, model->device_id);
    llaisysLinear(logits, logits_in, model->weights.out_embed, nullptr);

    int64_t next = model->meta.end_token;
    if (do_sample) {
        auto sample_idx = make_tensor(one_shape, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        llaisysRandomSample(sample_idx, logits, temperature, top_k, top_p, seed);
        next = read_scalar_i64(sample_idx, runtime_api, model->meta.end_token);
        delete sample_idx;
    } else {
        auto max_idx = make_tensor(one_shape, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        auto max_val = make_tensor(one_shape, dtype, model->device_type, model->device_id);
        llaisysArgmax(max_idx, max_val, logits);
        next = read_scalar_i64(max_idx, runtime_api, model->meta.end_token);
        delete max_idx;
        delete max_val;
    }

    delete idx;
    delete x;
    if (out_norm) delete out_norm;
    delete logits;

    return next;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model) return -1;
    if (ntoken == 0 || token_ids == nullptr) return model->meta.end_token;

    int64_t next = model->meta.end_token;
    for (size_t i = 0; i < ntoken; ++i) {
        next = infer_one_token(model, token_ids[i], false, 1.0f, 1, 1.0f, 0);
    }
    return next;
}

__export int64_t llaisysQwen2ModelInferSample(
    struct LlaisysQwen2Model * model,
    int64_t * token_ids,
    size_t ntoken,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    if (!model) return -1;
    if (ntoken == 0 || token_ids == nullptr) return model->meta.end_token;

    int64_t next = model->meta.end_token;
    for (size_t i = 0; i < ntoken; ++i) {
        const bool do_sample = (i + 1 == ntoken);
        const uint64_t step_seed = seed + static_cast<uint64_t>(i);
        next = infer_one_token(model, token_ids[i], do_sample, temperature, top_k, top_p, step_seed);
    }
    return next;
}

} // extern "C"

// end of file
