#include "qwen2.hpp"
#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config &config, llaisysDeviceType_t device_type, int device_id)
    : _config(config), _device_type(device_type), _device_id(device_id) {

    core::context().setDevice(_device_type, _device_id);

    _weights.layers.resize(config.nlayer);

    _kvcache.resize(config.nlayer);
    for (size_t i = 0; i < config.nlayer; i++) {
        _kvcache[i].k = _alloc({config.maxseq, config.nkvh, config.dh});
        _kvcache[i].v = _alloc({config.maxseq, config.nkvh, config.dh});
        _kvcache[i].len = 0;
    }
}

tensor_t Qwen2Model::_alloc(const std::vector<size_t> &shape) {
    return Tensor::create(shape, _config.dtype, _device_type, _device_id);
}

tensor_t Qwen2Model::_alloc(const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    return Tensor::create(shape, dtype, _device_type, _device_id);
}

void Qwen2Model::_copy_into(tensor_t dst, size_t dst_offset_elems, tensor_t src) {
    size_t bytes = src->numel() * src->elementSize();
    size_t offset_bytes = dst_offset_elems * dst->elementSize();
    auto &rt = core::context().runtime();
    rt.api()->memcpy_async(
        dst->data() + offset_bytes, src->data(), bytes, LLAISYS_MEMCPY_D2D, rt.stream());
}

void Qwen2Model::_ensure_workspace(size_t seqlen) {
    if (_ws.seqlen == seqlen) return;
    _ws.seqlen = seqlen;

    auto &c = _config;
    _ws.input_ids      = _alloc({seqlen}, LLAISYS_DTYPE_I64);
    _ws.pos_ids        = _alloc({seqlen}, LLAISYS_DTYPE_I64);
    _ws.hidden         = _alloc({seqlen, c.hs});
    _ws.normed         = _alloc({seqlen, c.hs});
    _ws.q_proj         = _alloc({seqlen, c.nh * c.dh});
    _ws.k_proj         = _alloc({seqlen, c.nkvh * c.dh});
    _ws.v_proj         = _alloc({seqlen, c.nkvh * c.dh});
    _ws.attn_out_flat  = _alloc({seqlen, c.nh * c.dh});
    _ws.attn_projected = _alloc({seqlen, c.hs});
    _ws.gate_buf       = _alloc({seqlen, c.di});
    _ws.up_buf         = _alloc({seqlen, c.di});
    _ws.swiglu_out     = _alloc({seqlen, c.di});
    _ws.mlp_out        = _alloc({seqlen, c.hs});
    _ws.residual       = _alloc({seqlen, c.hs});
    _ws.q_rope         = _alloc({seqlen, c.nh, c.dh});
    _ws.k_rope         = _alloc({seqlen, c.nkvh, c.dh});
    _ws.attn_val       = _alloc({seqlen, c.nh, c.dh});
    _ws.logits         = _alloc({1, c.voc});
    _ws.max_idx        = _alloc({1}, LLAISYS_DTYPE_I64);
    _ws.max_val        = _alloc({1});
    _ws.sampled_idx    = _alloc({1}, LLAISYS_DTYPE_I64);
}

void Qwen2Model::reset_kvcache() {
    for (auto &kv : _kvcache) {
        kv.len = 0;
    }
}

tensor_t Qwen2Model::forward(const int64_t *token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);

    auto &cfg = _config;
    size_t seqlen = ntoken;
    size_t nh = cfg.nh;
    size_t nkvh = cfg.nkvh;
    size_t dh = cfg.dh;

    _ensure_workspace(seqlen);

    _ws.input_ids->load(token_ids);

    size_t start_pos = _kvcache[0].len;
    std::vector<int64_t> pos_data(seqlen);
    for (size_t i = 0; i < seqlen; i++) {
        pos_data[i] = static_cast<int64_t>(start_pos + i);
    }
    _ws.pos_ids->load(pos_data.data());

    ops::embedding(_ws.hidden, _ws.input_ids, _weights.in_embed);

    for (size_t layer = 0; layer < cfg.nlayer; layer++) {
        auto &lw = _weights.layers[layer];
        auto &kv = _kvcache[layer];

        ops::rms_norm(_ws.normed, _ws.hidden, lw.attn_norm_w, cfg.epsilon);

        ops::linear(_ws.q_proj, _ws.normed, lw.attn_q_w, lw.attn_q_b);
        ops::linear(_ws.k_proj, _ws.normed, lw.attn_k_w, lw.attn_k_b);
        ops::linear(_ws.v_proj, _ws.normed, lw.attn_v_w, lw.attn_v_b);

        auto q = _ws.q_proj->view({seqlen, nh, dh});
        auto k_new = _ws.k_proj->view({seqlen, nkvh, dh});
        auto v_new = _ws.v_proj->view({seqlen, nkvh, dh});

        ops::rope(_ws.q_rope, q, _ws.pos_ids, cfg.theta);
        ops::rope(_ws.k_rope, k_new, _ws.pos_ids, cfg.theta);

        size_t kv_offset = kv.len * nkvh * dh;
        _copy_into(kv.k, kv_offset, _ws.k_rope);
        _copy_into(kv.v, kv_offset, v_new);

        size_t total_len = kv.len + seqlen;

        auto k_full = kv.k->slice(0, 0, total_len);
        auto v_full = kv.v->slice(0, 0, total_len);

        float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        ops::self_attention(_ws.attn_val, _ws.q_rope, k_full, v_full, scale);

        auto attn_flat = _ws.attn_val->view({seqlen, nh * dh});
        ops::linear(_ws.attn_projected, attn_flat, lw.attn_o_w, nullptr);

        ops::add(_ws.residual, _ws.hidden, _ws.attn_projected);

        ops::rms_norm(_ws.normed, _ws.residual, lw.mlp_norm_w, cfg.epsilon);

        ops::linear(_ws.gate_buf, _ws.normed, lw.mlp_gate_w, nullptr);
        ops::linear(_ws.up_buf, _ws.normed, lw.mlp_up_w, nullptr);
        ops::swiglu(_ws.swiglu_out, _ws.gate_buf, _ws.up_buf);
        ops::linear(_ws.mlp_out, _ws.swiglu_out, lw.mlp_down_w, nullptr);

        ops::add(_ws.hidden, _ws.residual, _ws.mlp_out);

        kv.len = total_len;
    }

    ops::rms_norm(_ws.normed, _ws.hidden, _weights.out_norm_w, cfg.epsilon);

    auto last_hidden = _ws.normed->slice(0, seqlen - 1, seqlen);

    ops::linear(_ws.logits, last_hidden, _weights.out_embed, nullptr);

    return _ws.logits;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    auto logits = forward(token_ids, ntoken);

    _ensure_workspace(ntoken);
    ops::argmax(_ws.max_idx, _ws.max_val, logits->view({_config.voc}));

    int64_t result = 0;
    core::context().runtime().api()->memcpy_sync(
        &result, _ws.max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    return result;
}

int64_t Qwen2Model::infer_sample(const int64_t *token_ids, size_t ntoken,
                                  float temperature, int top_k, float top_p) {
    auto logits = forward(token_ids, ntoken);

    _ensure_workspace(ntoken);
    ops::sample(_ws.sampled_idx, logits->view({_config.voc}), temperature, top_k, top_p);

    int64_t result = 0;
    core::context().runtime().api()->memcpy_sync(
        &result, _ws.sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    return result;
}

} // namespace llaisys::models
