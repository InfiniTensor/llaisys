#include "qwen2_model.hpp"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>

namespace llaisys::models::qwen2 {
namespace {
inline tensor_t unwrap(llaisysTensor_t handle) {
    if (!handle) {
        return nullptr;
    }
    return handle->tensor;
}

void bind_layer_list(std::vector<tensor_t> &dst, llaisysTensor_t *src, size_t nlayer) {
    dst.resize(nlayer);
    if (!src) {
        for (size_t i = 0; i < nlayer; ++i) {
            dst[i] = nullptr;
        }
        return;
    }
    for (size_t i = 0; i < nlayer; ++i) {
        dst[i] = unwrap(src[i]);
    }
}
} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    CHECK_ARGUMENT(_meta.nlayer > 0, "nlayer must be > 0");
    CHECK_ARGUMENT(_meta.hs > 0, "hidden_size must be > 0");
    CHECK_ARGUMENT(_meta.nh > 0, "num_attention_heads must be > 0");
    CHECK_ARGUMENT(_meta.nkvh > 0, "num_key_value_heads must be > 0");
    CHECK_ARGUMENT(_meta.dh > 0, "head_dim must be > 0");
    CHECK_ARGUMENT(_meta.di > 0, "intermediate_size must be > 0");
    CHECK_ARGUMENT(_meta.voc > 0, "vocab_size must be > 0");
    CHECK_ARGUMENT(_meta.nh * _meta.dh == _meta.hs, "nh * dh must equal hidden_size");

    _attn_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

    _kv_cache.reserve(
        _meta.nlayer,
        _meta.maxseq,
        _meta.nkvh,
        _meta.dh,
        _meta.dtype,
        _device,
        _device_id);

    _token_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    _pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    _hidden = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _attn_norm = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _q_proj = Tensor::create({1, _meta.nh * _meta.dh}, _meta.dtype, _device, _device_id);
    _k_proj = Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id);
    _v_proj = Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id);
    _q_view = _q_proj->view({1, _meta.nh, _meta.dh});
    _k_view = _k_proj->view({1, _meta.nkvh, _meta.dh});
    _v_view = _v_proj->view({1, _meta.nkvh, _meta.dh});
    _q_rope = Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
    _k_rope = Tensor::create({1, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
    _attn_out = Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
    _attn_out_flat = _attn_out->view({1, _meta.hs});
    _attn_proj = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _mlp_norm = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _mlp_gate = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
    _mlp_up = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
    _mlp_act = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
    _mlp_down = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _final_norm = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    _logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device, _device_id);
    _logits_flat = _logits->view({_meta.voc});
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    _max_val = Tensor::create({1}, _meta.dtype, _device, _device_id);
}

void Qwen2Model::bind_weights(const LlaisysQwen2Weights &weights) {
    _weights.in_embed = unwrap(weights.in_embed);
    _weights.out_embed = unwrap(weights.out_embed);
    _weights.out_norm_w = unwrap(weights.out_norm_w);

    bind_layer_list(_weights.attn_norm_w, weights.attn_norm_w, _meta.nlayer);
    bind_layer_list(_weights.attn_q_w, weights.attn_q_w, _meta.nlayer);
    bind_layer_list(_weights.attn_q_b, weights.attn_q_b, _meta.nlayer);
    bind_layer_list(_weights.attn_k_w, weights.attn_k_w, _meta.nlayer);
    bind_layer_list(_weights.attn_k_b, weights.attn_k_b, _meta.nlayer);
    bind_layer_list(_weights.attn_v_w, weights.attn_v_w, _meta.nlayer);
    bind_layer_list(_weights.attn_v_b, weights.attn_v_b, _meta.nlayer);
    bind_layer_list(_weights.attn_o_w, weights.attn_o_w, _meta.nlayer);

    bind_layer_list(_weights.mlp_norm_w, weights.mlp_norm_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_gate_w, weights.mlp_gate_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_up_w, weights.mlp_up_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_down_w, weights.mlp_down_w, _meta.nlayer);

    CHECK_ARGUMENT(_weights.in_embed != nullptr, "missing in_embed weight");
    CHECK_ARGUMENT(_weights.out_embed != nullptr, "missing out_embed weight");
    CHECK_ARGUMENT(_weights.out_norm_w != nullptr, "missing out_norm_w weight");

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        CHECK_ARGUMENT(_weights.attn_norm_w[i] != nullptr, "missing attn_norm_w");
        CHECK_ARGUMENT(_weights.attn_q_w[i] != nullptr, "missing attn_q_w");
        CHECK_ARGUMENT(_weights.attn_k_w[i] != nullptr, "missing attn_k_w");
        CHECK_ARGUMENT(_weights.attn_v_w[i] != nullptr, "missing attn_v_w");
        CHECK_ARGUMENT(_weights.attn_o_w[i] != nullptr, "missing attn_o_w");
        CHECK_ARGUMENT(_weights.mlp_norm_w[i] != nullptr, "missing mlp_norm_w");
        CHECK_ARGUMENT(_weights.mlp_gate_w[i] != nullptr, "missing mlp_gate_w");
        CHECK_ARGUMENT(_weights.mlp_up_w[i] != nullptr, "missing mlp_up_w");
        CHECK_ARGUMENT(_weights.mlp_down_w[i] != nullptr, "missing mlp_down_w");
    }

    _weights_bound = true;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids || ntoken == 0, "token_ids is null");
    CHECK_ARGUMENT(_weights_bound, "Model weights are not bound");
    if (ntoken == 0) {
        return _meta.end_token;
    }
    CHECK_ARGUMENT(ntoken <= _meta.maxseq, "ntoken exceeds maxseq");

    if (_kv_cache.seq_len() >= _meta.maxseq) {
        _kv_cache.reset();
    }

    if (ntoken <= _kv_cache.seq_len()) {
        _kv_cache.reset();
    }

    for (size_t i = _kv_cache.seq_len(); i < ntoken; ++i) {
        process_token(token_ids[i]);
    }

    llaisys::ops::rms_norm(_final_norm, _hidden, _weights.out_norm_w, _meta.epsilon);
    llaisys::ops::linear(_logits, _final_norm, _weights.out_embed, nullptr);
    llaisys::ops::argmax(_max_idx, _max_val, _logits_flat);

    auto *idx_ptr = reinterpret_cast<const int64_t *>(_max_idx->data());
    return idx_ptr[0];
}

void Qwen2Model::process_token(int64_t token_id) {
    int64_t pos = static_cast<int64_t>(_kv_cache.seq_len());

    _token_ids->load(&token_id);
    _pos_ids->load(&pos);

    llaisys::ops::embedding(_hidden, _token_ids, _weights.in_embed);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        llaisys::ops::rms_norm(_attn_norm, _hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        llaisys::ops::linear(_q_proj, _attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        llaisys::ops::linear(_k_proj, _attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        llaisys::ops::linear(_v_proj, _attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        llaisys::ops::rope(_q_rope, _q_view, _pos_ids, _meta.theta);
        llaisys::ops::rope(_k_rope, _k_view, _pos_ids, _meta.theta);

        auto k_cache_slice = _kv_cache.k(layer)->slice(0, pos, pos + 1);
        auto v_cache_slice = _kv_cache.v(layer)->slice(0, pos, pos + 1);
        llaisys::ops::rearrange(k_cache_slice, _k_rope);
        llaisys::ops::rearrange(v_cache_slice, _v_view);

        auto k_cache = _kv_cache.k(layer)->slice(0, 0, pos + 1);
        auto v_cache = _kv_cache.v(layer)->slice(0, 0, pos + 1);

        llaisys::ops::self_attention(_attn_out, _q_rope, k_cache, v_cache, _attn_scale);
        llaisys::ops::linear(_attn_proj, _attn_out_flat, _weights.attn_o_w[layer], nullptr);
        llaisys::ops::add(_hidden, _hidden, _attn_proj);

        llaisys::ops::rms_norm(_mlp_norm, _hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        llaisys::ops::linear(_mlp_gate, _mlp_norm, _weights.mlp_gate_w[layer], nullptr);
        llaisys::ops::linear(_mlp_up, _mlp_norm, _weights.mlp_up_w[layer], nullptr);
        llaisys::ops::swiglu(_mlp_act, _mlp_gate, _mlp_up);
        llaisys::ops::linear(_mlp_down, _mlp_act, _weights.mlp_down_w[layer], nullptr);
        llaisys::ops::add(_hidden, _hidden, _mlp_down);
    }

    _kv_cache.advance(1);
}

} // namespace llaisys::models::qwen2
