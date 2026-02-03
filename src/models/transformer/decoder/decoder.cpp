#include "decoder.hpp"

#include "llaisys/ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace llaisys::models::transformer {
namespace {
bool trace_enabled() {
    static bool enabled = false;
    static bool inited = false;
    if (!inited) {
#if defined(_WIN32)
        char *value = nullptr;
        size_t len = 0;
        if (_dupenv_s(&value, &len, "LLAISYS_QWEN2_TRACE") == 0 && value) {
            if (value[0] != '\0' && value[0] != '0') enabled = true;
            free(value);
        }
#else
        const char *value = std::getenv("LLAISYS_QWEN2_TRACE");
        if (value && value[0] != '\0' && value[0] != '0') enabled = true;
#endif
        inited = true;
    }
    return enabled;
}

void trace(const char *stage) {
    if (trace_enabled()) {
        std::cerr << "[TRACE] Decoder forward: " << stage << std::endl;
    }
}

bool require_tensor(llaisysTensor_t t, const char *stage) {
    if (t) return true;
    std::cerr << "[ERROR] Decoder: tensorCreate failed at " << stage << std::endl;
    return false;
}

bool ensure_data(llaisysTensor_t t, const char *stage) {
    if (!t) {
        std::cerr << "[ERROR] Decoder: null tensor at " << stage << std::endl;
        return false;
    }
    if (!tensorGetData(t)) {
        std::cerr << "[ERROR] Decoder: null data at " << stage << std::endl;
        return false;
    }
    return true;
}
} // namespace

Decoder::Decoder(const DecoderConfig &config,
                 const LlaisysQwen2Weights *weights,
                 llaisysDeviceType_t device,
                 const std::vector<int> &device_ids)
    : _config(config),
      _weights(weights),
      _device(device),
      _device_ids(device_ids) {}

Decoder::~Decoder() {
    releaseCache();
}

void Decoder::ensureCache() {
    if (!_kv_cache_enabled || _cache_inited || _config.maxseq == 0 || _config.nlayer == 0) return;
    _k_cache.assign(_config.nlayer, nullptr);
    _v_cache.assign(_config.nlayer, nullptr);

    size_t kv_shape[3] = {_config.maxseq, _config.nkvh, _config.dh};
    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    for (size_t i = 0; i < _config.nlayer; ++i) {
        _k_cache[i] = tensorCreate(kv_shape, 3, _config.dtype, _device, device_id);
        _v_cache[i] = tensorCreate(kv_shape, 3, _config.dtype, _device, device_id);
    }
    _past_len = 0;
    _cache_inited = true;
}

void Decoder::releaseCache() {
    for (auto &t : _k_cache) {
        if (t) tensorDestroy(t);
        t = nullptr;
    }
    for (auto &t : _v_cache) {
        if (t) tensorDestroy(t);
        t = nullptr;
    }
    _k_cache.clear();
    _v_cache.clear();
    _past_len = 0;
    _cache_inited = false;
}

void Decoder::resetKVCache() {
    if (!_cache_inited) return;
    _past_len = 0;
}

void Decoder::setKVCacheEnabled(bool enabled) {
    if (_kv_cache_enabled == enabled) return;
    _kv_cache_enabled = enabled;
    if (!enabled) {
        releaseCache();
    }
}

bool Decoder::runHidden(const int64_t *token_ids,
                        size_t ntoken,
                        bool append_only,
                        size_t &past_len,
                        size_t &cur_len,
                        llaisysTensor_t &idx,
                        llaisysTensor_t &pos_ids,
                        llaisysTensor_t &hidden) {
    idx = nullptr;
    pos_ids = nullptr;
    hidden = nullptr;
    if (!token_ids || ntoken == 0) return false;
    if (!_weights || !_weights->in_embed) return false;

    ensureCache();
    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    const bool can_cache = _cache_inited && _config.maxseq > 0;
    if (can_cache && ntoken > _config.maxseq) return false;

    past_len = can_cache ? _past_len : 0;
    if (append_only && !can_cache) {
        return false;
    }
    if (!append_only) {
        if (!can_cache || ntoken <= past_len) {
            past_len = 0;
            if (can_cache) _past_len = 0;
        }
        cur_len = ntoken - past_len;
    } else {
        cur_len = ntoken;
    }
    if (cur_len == 0) return false;
    if (trace_enabled()) {
        std::cerr << "[TRACE] Decoder cache: enabled=" << (_kv_cache_enabled ? 1 : 0)
                  << " inited=" << (_cache_inited ? 1 : 0)
                  << " can_cache=" << (can_cache ? 1 : 0)
                  << " past_len=" << past_len
                  << " cur_len=" << cur_len
                  << " ntoken=" << ntoken << std::endl;
    }
    const int64_t *new_tokens = append_only ? token_ids : (token_ids + past_len);
    if (can_cache) {
        if (_k_cache.size() != _config.nlayer || _v_cache.size() != _config.nlayer) return false;
        if (past_len + cur_len > _config.maxseq) return false;
    }

    trace("begin");
    // 1) token ids -> embedding
    size_t idx_shape[1] = {cur_len};
    idx = tensorCreate(idx_shape, 1, LLAISYS_DTYPE_I64, _device, device_id);
    if (!require_tensor(idx, "idx")) return false;
    tensorLoad(idx, new_tokens);

    size_t hidden_shape[2] = {cur_len, _config.hs};
    hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
    if (!require_tensor(hidden, "hidden")) {
        tensorDestroy(idx);
        idx = nullptr;
        return false;
    }

    trace("embedding");
    ::llaisysEmbedding(hidden, idx, _weights->in_embed);

    // 2) position ids for RoPE
    std::vector<int64_t> pos_buf(cur_len);
    for (size_t i = 0; i < cur_len; ++i) pos_buf[i] = static_cast<int64_t>(past_len + i);
    trace("pos_ids");
    pos_ids = tensorCreate(idx_shape, 1, LLAISYS_DTYPE_I64, _device, device_id);
    if (!require_tensor(pos_ids, "pos_ids")) {
        tensorDestroy(hidden);
        tensorDestroy(idx);
        hidden = nullptr;
        idx = nullptr;
        return false;
    }
    tensorLoad(pos_ids, pos_buf.data());

    // 3) Attention + MLP blocks
    const float scale = 1.0f / std::sqrt(static_cast<float>(_config.dh));
    for (size_t layer = 0; layer < _config.nlayer; ++layer) {
        trace("attn.weights.check");
        if (!_weights->attn_norm_w || !_weights->attn_q_w || !_weights->attn_k_w || !_weights->attn_v_w ||
            !_weights->attn_o_w || !_weights->mlp_norm_w || !_weights->mlp_gate_w || !_weights->mlp_up_w ||
            !_weights->mlp_down_w) {
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        if (!_weights->attn_norm_w[layer] || !_weights->attn_q_w[layer] || !_weights->attn_k_w[layer] ||
            !_weights->attn_v_w[layer] || !_weights->attn_o_w[layer] || !_weights->mlp_norm_w[layer] ||
            !_weights->mlp_gate_w[layer] || !_weights->mlp_up_w[layer] || !_weights->mlp_down_w[layer]) {
            std::cerr << "[ERROR] Decoder: missing weights at layer " << layer << std::endl;
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }

        trace("attn.norm");
        llaisysTensor_t norm = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(norm, "attn.norm")) {
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysRmsNorm(norm, hidden, _weights->attn_norm_w[layer], _config.epsilon);

        trace("attn.qkv");
        size_t q2d_shape[2] = {cur_len, _config.nh * _config.dh};
        size_t kv2d_shape[2] = {cur_len, _config.nkvh * _config.dh};
        llaisysTensor_t q2d = tensorCreate(q2d_shape, 2, _config.dtype, _device, device_id);
        llaisysTensor_t k2d = tensorCreate(kv2d_shape, 2, _config.dtype, _device, device_id);
        llaisysTensor_t v2d = tensorCreate(kv2d_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(q2d, "attn.q2d") || !require_tensor(k2d, "attn.k2d") ||
            !require_tensor(v2d, "attn.v2d")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            if (q2d) tensorDestroy(q2d);
            if (k2d) tensorDestroy(k2d);
            if (v2d) tensorDestroy(v2d);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }

        llaisysTensor_t q_bias = (_weights->attn_q_b && _weights->attn_q_b[layer]) ? _weights->attn_q_b[layer] : nullptr;
        llaisysTensor_t k_bias = (_weights->attn_k_b && _weights->attn_k_b[layer]) ? _weights->attn_k_b[layer] : nullptr;
        llaisysTensor_t v_bias = (_weights->attn_v_b && _weights->attn_v_b[layer]) ? _weights->attn_v_b[layer] : nullptr;

        ::llaisysLinear(q2d, norm, _weights->attn_q_w[layer], q_bias);
        ::llaisysLinear(k2d, norm, _weights->attn_k_w[layer], k_bias);
        ::llaisysLinear(v2d, norm, _weights->attn_v_w[layer], v_bias);

        trace("attn.view");
        size_t q3d_shape[3] = {cur_len, _config.nh, _config.dh};
        size_t k3d_shape[3] = {cur_len, _config.nkvh, _config.dh};
        llaisysTensor_t q3d = tensorView(q2d, q3d_shape, 3);
        llaisysTensor_t k3d = tensorView(k2d, k3d_shape, 3);
        llaisysTensor_t v3d = tensorView(v2d, k3d_shape, 3);
        if (!require_tensor(q3d, "attn.q3d") || !require_tensor(k3d, "attn.k3d") ||
            !require_tensor(v3d, "attn.v3d")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            if (q3d) tensorDestroy(q3d);
            if (k3d) tensorDestroy(k3d);
            if (v3d) tensorDestroy(v3d);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }

        trace("attn.rope");
        llaisysTensor_t q_rope = tensorCreate(q3d_shape, 3, _config.dtype, _device, device_id);
        llaisysTensor_t k_rope = tensorCreate(k3d_shape, 3, _config.dtype, _device, device_id);
        if (!require_tensor(q_rope, "attn.q_rope") || !require_tensor(k_rope, "attn.k_rope")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q3d);
            tensorDestroy(k3d);
            tensorDestroy(v3d);
            if (q_rope) tensorDestroy(q_rope);
            if (k_rope) tensorDestroy(k_rope);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysROPE(q_rope, q3d, pos_ids, _config.theta);
        ::llaisysROPE(k_rope, k3d, pos_ids, _config.theta);

        if (can_cache) {
            trace("attn.cache.write");
            llaisysTensor_t k_slot = tensorSlice(_k_cache[layer], 0, past_len, past_len + cur_len);
            llaisysTensor_t v_slot = tensorSlice(_v_cache[layer], 0, past_len, past_len + cur_len);
            ::llaisysRearrange(k_slot, k_rope);
            ::llaisysRearrange(v_slot, v3d);
            tensorDestroy(k_slot);
            tensorDestroy(v_slot);
        }

        llaisysTensor_t k_attn = k_rope;
        llaisysTensor_t v_attn = v3d;
        llaisysTensor_t k_cache_view = nullptr;
        llaisysTensor_t v_cache_view = nullptr;
        if (can_cache) {
            trace("attn.cache.read");
            size_t total_len = past_len + cur_len;
            k_cache_view = tensorSlice(_k_cache[layer], 0, 0, total_len);
            v_cache_view = tensorSlice(_v_cache[layer], 0, 0, total_len);
            k_attn = k_cache_view;
            v_attn = v_cache_view;
        }

        trace("attn.softmax");
        llaisysTensor_t attn_out3d = tensorCreate(q3d_shape, 3, _config.dtype, _device, device_id);
        if (!require_tensor(attn_out3d, "attn.out3d")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q3d);
            tensorDestroy(k3d);
            tensorDestroy(v3d);
            tensorDestroy(q_rope);
            tensorDestroy(k_rope);
            if (k_cache_view) tensorDestroy(k_cache_view);
            if (v_cache_view) tensorDestroy(v_cache_view);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysSelfAttention(attn_out3d, q_rope, k_attn, v_attn, scale);
        if (k_cache_view) tensorDestroy(k_cache_view);
        if (v_cache_view) tensorDestroy(v_cache_view);

        trace("attn.proj");
        llaisysTensor_t attn_out2d = tensorView(attn_out3d, hidden_shape, 2);
        llaisysTensor_t proj_out = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(attn_out2d, "attn.out2d") || !require_tensor(proj_out, "attn.proj_out")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q3d);
            tensorDestroy(k3d);
            tensorDestroy(v3d);
            tensorDestroy(q_rope);
            tensorDestroy(k_rope);
            tensorDestroy(attn_out3d);
            if (attn_out2d) tensorDestroy(attn_out2d);
            if (proj_out) tensorDestroy(proj_out);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        if (!ensure_data(attn_out2d, "attn.proj.in") || !ensure_data(proj_out, "attn.proj.out") ||
            !ensure_data(_weights->attn_o_w[layer], "attn.proj.w")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q3d);
            tensorDestroy(k3d);
            tensorDestroy(v3d);
            tensorDestroy(q_rope);
            tensorDestroy(k_rope);
            tensorDestroy(attn_out3d);
            tensorDestroy(attn_out2d);
            tensorDestroy(proj_out);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysLinear(proj_out, attn_out2d, _weights->attn_o_w[layer], nullptr);

        trace("attn.residual");
        llaisysTensor_t new_hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(new_hidden, "attn.residual")) {
            tensorDestroy(norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            tensorDestroy(q2d);
            tensorDestroy(k2d);
            tensorDestroy(v2d);
            tensorDestroy(q3d);
            tensorDestroy(k3d);
            tensorDestroy(v3d);
            tensorDestroy(q_rope);
            tensorDestroy(k_rope);
            tensorDestroy(attn_out3d);
            tensorDestroy(attn_out2d);
            tensorDestroy(proj_out);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysAdd(new_hidden, hidden, proj_out);

        tensorDestroy(hidden);
        hidden = new_hidden;

        tensorDestroy(norm);
        tensorDestroy(q2d);
        tensorDestroy(k2d);
        tensorDestroy(v2d);
        tensorDestroy(q3d);
        tensorDestroy(k3d);
        tensorDestroy(v3d);
        tensorDestroy(q_rope);
        tensorDestroy(k_rope);
        tensorDestroy(attn_out3d);
        tensorDestroy(attn_out2d);
        tensorDestroy(proj_out);

        // 4) MLP
        trace("mlp.norm");
        llaisysTensor_t mlp_norm = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(mlp_norm, "mlp.norm")) {
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysRmsNorm(mlp_norm, hidden, _weights->mlp_norm_w[layer], _config.epsilon);

        trace("mlp.gate_up");
        size_t mlp_shape[2] = {cur_len, _config.di};
        llaisysTensor_t gate = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
        llaisysTensor_t up = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(gate, "mlp.gate") || !require_tensor(up, "mlp.up")) {
            tensorDestroy(mlp_norm);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            if (gate) tensorDestroy(gate);
            if (up) tensorDestroy(up);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysLinear(gate, mlp_norm, _weights->mlp_gate_w[layer], nullptr);
        ::llaisysLinear(up, mlp_norm, _weights->mlp_up_w[layer], nullptr);

        trace("mlp.swiglu");
        llaisysTensor_t swiglu = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(swiglu, "mlp.swiglu")) {
            tensorDestroy(mlp_norm);
            tensorDestroy(gate);
            tensorDestroy(up);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysSwiGLU(swiglu, gate, up);

        trace("mlp.down");
        llaisysTensor_t mlp_out = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(mlp_out, "mlp.down")) {
            tensorDestroy(mlp_norm);
            tensorDestroy(gate);
            tensorDestroy(up);
            tensorDestroy(swiglu);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysLinear(mlp_out, swiglu, _weights->mlp_down_w[layer], nullptr);

        trace("mlp.residual");
        llaisysTensor_t mlp_hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        if (!require_tensor(mlp_hidden, "mlp.residual")) {
            tensorDestroy(mlp_norm);
            tensorDestroy(gate);
            tensorDestroy(up);
            tensorDestroy(swiglu);
            tensorDestroy(mlp_out);
            tensorDestroy(pos_ids);
            tensorDestroy(hidden);
            tensorDestroy(idx);
            pos_ids = nullptr;
            hidden = nullptr;
            idx = nullptr;
            return false;
        }
        ::llaisysAdd(mlp_hidden, hidden, mlp_out);

        tensorDestroy(hidden);
        hidden = mlp_hidden;

        tensorDestroy(mlp_norm);
        tensorDestroy(gate);
        tensorDestroy(up);
        tensorDestroy(swiglu);
        tensorDestroy(mlp_out);
    }

    if (can_cache) {
        _past_len = past_len + cur_len;
    }

    return true;
}

bool Decoder::prefill(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits) {
    if (!out_last_logits) return false;
    if (!ensure_data(out_last_logits, "head.logits.out")) return false;

    size_t past_len = 0;
    size_t cur_len = 0;
    llaisysTensor_t idx = nullptr;
    llaisysTensor_t pos_ids = nullptr;
    llaisysTensor_t hidden = nullptr;
    if (!runHidden(token_ids, ntoken, false, past_len, cur_len, idx, pos_ids, hidden)) return false;

    if (!_weights || !_weights->out_norm_w || !_weights->out_embed) {
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }

    trace("head.slice");
    llaisysTensor_t last_hidden = tensorSlice(hidden, 0, cur_len - 1, cur_len);
    if (!require_tensor(last_hidden, "head.last_hidden")) {
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }

    size_t last_shape[2] = {1, _config.hs};
    trace("head.norm");
    llaisysTensor_t final_norm = tensorCreate(last_shape, 2, _config.dtype, _device, _device_ids.empty() ? 0 : _device_ids[0]);
    if (!require_tensor(final_norm, "head.norm")) {
        tensorDestroy(last_hidden);
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }
    ::llaisysRmsNorm(final_norm, last_hidden, _weights->out_norm_w, _config.epsilon);

    trace("head.logits");
    ::llaisysLinear(out_last_logits, final_norm, _weights->out_embed, nullptr);

    tensorDestroy(last_hidden);
    tensorDestroy(final_norm);
    tensorDestroy(idx);
    tensorDestroy(pos_ids);
    tensorDestroy(hidden);
    return true;
}

bool Decoder::decodeStep(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits) {
    if (!out_last_logits) return false;
    if (!ensure_data(out_last_logits, "head.logits.out")) return false;

    size_t past_len = 0;
    size_t cur_len = 0;
    llaisysTensor_t idx = nullptr;
    llaisysTensor_t pos_ids = nullptr;
    llaisysTensor_t hidden = nullptr;
    if (!runHidden(token_ids, ntoken, true, past_len, cur_len, idx, pos_ids, hidden)) return false;

    if (!_weights || !_weights->out_norm_w || !_weights->out_embed) {
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }

    trace("head.slice");
    llaisysTensor_t last_hidden = tensorSlice(hidden, 0, cur_len - 1, cur_len);
    if (!require_tensor(last_hidden, "head.last_hidden")) {
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }

    size_t last_shape[2] = {1, _config.hs};
    trace("head.norm");
    llaisysTensor_t final_norm = tensorCreate(last_shape, 2, _config.dtype, _device, _device_ids.empty() ? 0 : _device_ids[0]);
    if (!require_tensor(final_norm, "head.norm")) {
        tensorDestroy(last_hidden);
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }
    ::llaisysRmsNorm(final_norm, last_hidden, _weights->out_norm_w, _config.epsilon);

    trace("head.logits");
    ::llaisysLinear(out_last_logits, final_norm, _weights->out_embed, nullptr);

    tensorDestroy(last_hidden);
    tensorDestroy(final_norm);
    tensorDestroy(idx);
    tensorDestroy(pos_ids);
    tensorDestroy(hidden);
    return true;
}

} // namespace llaisys::models::transformer
