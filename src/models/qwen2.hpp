#pragma once

#include "../tensor/tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"

#include <vector>

namespace llaisys::models {

struct Qwen2Config {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2LayerWeights {
    tensor_t attn_norm_w;
    tensor_t attn_q_w, attn_q_b;
    tensor_t attn_k_w, attn_k_b;
    tensor_t attn_v_w, attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w, mlp_up_w, mlp_down_w;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<Qwen2LayerWeights> layers;
};

struct KVCache {
    tensor_t k; // [maxseq, nkvh, dh]
    tensor_t v; // [maxseq, nkvh, dh]
    size_t len;
};

struct Qwen2Workspace {
    size_t seqlen = 0;
    tensor_t input_ids, pos_ids;
    tensor_t hidden, normed;
    tensor_t q_proj, k_proj, v_proj;
    tensor_t attn_out_flat, attn_projected;
    tensor_t gate_buf, up_buf, swiglu_out, mlp_out;
    tensor_t residual;
    tensor_t q_rope, k_rope, attn_val;
    tensor_t logits;
    tensor_t max_idx, max_val, sampled_idx;
};

class Qwen2Model {
private:
    Qwen2Config _config;
    Qwen2Weights _weights;
    std::vector<KVCache> _kvcache;
    llaisysDeviceType_t _device_type;
    int _device_id;
    Qwen2Workspace _ws;

    tensor_t _alloc(const std::vector<size_t> &shape);
    tensor_t _alloc(const std::vector<size_t> &shape, llaisysDataType_t dtype);

    void _copy_into(tensor_t dst, size_t dst_offset_elems, tensor_t src);
    void _ensure_workspace(size_t seqlen);

    tensor_t forward(const int64_t *token_ids, size_t ntoken);

public:
    Qwen2Model(const Qwen2Config &config, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;

    Qwen2Weights &weights() { return _weights; }

    int64_t infer(const int64_t *token_ids, size_t ntoken);
    int64_t infer_sample(const int64_t *token_ids, size_t ntoken,
                         float temperature, int top_k, float top_p);
    void reset_kvcache();
};

} // namespace llaisys::models
