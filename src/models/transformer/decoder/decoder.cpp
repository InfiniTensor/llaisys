#include "decoder.hpp"
#include "../../../llaisys/models/qwen2_kv_internal.hpp"
#include "../../../device/comm_api.hpp"

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

void destroy_if_not_null(llaisysTensor_t t) {
    if (t) tensorDestroy(t);
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

void Decoder::setTensorParallel(llaisysComm_t comm, llaisysStream_t stream, int tp_size) {
    _comm = comm;
    _comm_stream = stream;
    _tp_size = tp_size > 0 ? tp_size : 1;
}

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

void Decoder::bindExternalKVContext(void *ctx, size_t past_len_tokens) {
    _external_kv_ctx = ctx;
    _external_past_len = past_len_tokens;
    _external_cache_ready = false;
    if (ctx) {
        releaseCache();
    }
}

void Decoder::clearExternalKVContext() {
    _external_kv_ctx = nullptr;
    _external_past_len = 0;
    _external_cache_ready = false;
}

bool Decoder::hasExternalKVContext() const {
    return _external_kv_ctx != nullptr;
}

int Decoder::exportKVContext(void *ctx_ptr, size_t block_tokens) {
    if (!ctx_ptr) return -1;
    if (!_kv_cache_enabled) return -2;
    ensureCache();
    if (!_cache_inited) return -3;

    auto *ctx = reinterpret_cast<LlaisysQwen2KVContext *>(ctx_ptr);
    if (!ctx) return -4;
    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    if (ctx->dtype != _config.dtype || ctx->device != _device || ctx->device_id != device_id) return -5;
    if (ctx->nlayer != _config.nlayer || ctx->nkvh != _config.nkvh || ctx->dh != _config.dh) return -6;

    llaisysQwen2KVContextDetachAll(ctx);
    if (_past_len == 0) return 0;

    const size_t chunk_size = block_tokens > 0 ? block_tokens : _past_len;
    size_t offset = 0;
    while (offset < _past_len) {
        const size_t used = std::min(chunk_size, _past_len - offset);
        LlaisysQwen2KVBlockMeta meta{};
        meta.dtype = _config.dtype;
        meta.nlayer = _config.nlayer;
        meta.nh = _config.nh;
        meta.nkvh = _config.nkvh;
        meta.dh = _config.dh;
        meta.max_tokens = used;
        auto *block = llaisysQwen2KVBlockCreate(&meta, _device, device_id);
        if (!block) {
            llaisysQwen2KVContextDetachAll(ctx);
            return -7;
        }
        if (llaisysQwen2KVBlockSetTokenCount(block, used) != 0) {
            llaisysQwen2KVBlockRelease(block);
            llaisysQwen2KVContextDetachAll(ctx);
            return -8;
        }

        bool copy_ok = true;
        for (size_t layer = 0; layer < _config.nlayer && copy_ok; ++layer) {
            llaisysTensor_t src_k = tensorSlice(_k_cache[layer], 0, offset, offset + used);
            llaisysTensor_t src_v = tensorSlice(_v_cache[layer], 0, offset, offset + used);
            llaisysTensor_t dst_k_full = llaisysQwen2KVBlockKeyTensor(block, layer);
            llaisysTensor_t dst_v_full = llaisysQwen2KVBlockValueTensor(block, layer);
            llaisysTensor_t dst_k = dst_k_full ? tensorSlice(dst_k_full, 0, 0, used) : nullptr;
            llaisysTensor_t dst_v = dst_v_full ? tensorSlice(dst_v_full, 0, 0, used) : nullptr;
            if (!src_k || !src_v || !dst_k || !dst_v) {
                copy_ok = false;
            } else {
                ::llaisysRearrange(dst_k, src_k);
                ::llaisysRearrange(dst_v, src_v);
            }
            destroy_if_not_null(src_k);
            destroy_if_not_null(src_v);
            destroy_if_not_null(dst_k);
            destroy_if_not_null(dst_v);
        }

        if (!copy_ok || llaisysQwen2KVContextAttachBlock(ctx, block) != 0) {
            llaisysQwen2KVBlockRelease(block);
            llaisysQwen2KVContextDetachAll(ctx);
            return -9;
        }
        llaisysQwen2KVBlockRelease(block);
        offset += used;
    }
    return 0;
}

bool Decoder::recoverExternalCache() {
    if (!_external_kv_ctx || _external_cache_ready) return true;
    if (!_kv_cache_enabled) return false;
    ensureCache();
    if (!_cache_inited) return false;

    auto *ctx = reinterpret_cast<LlaisysQwen2KVContext *>(_external_kv_ctx);
    if (!ctx) return false;
    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    if (ctx->dtype != _config.dtype || ctx->device != _device || ctx->device_id != device_id) return false;
    if (ctx->nlayer != _config.nlayer || ctx->nkvh != _config.nkvh || ctx->dh != _config.dh) return false;

    size_t total_tokens = 0;
    for (auto *blk : ctx->chain) {
        if (!blk) return false;
        if (blk->meta.dtype != _config.dtype || blk->device != _device || blk->device_id != device_id) return false;
        if (blk->meta.nlayer != _config.nlayer || blk->meta.nkvh != _config.nkvh || blk->meta.dh != _config.dh) return false;
        if (blk->used_tokens > blk->meta.max_tokens) return false;
        total_tokens += blk->used_tokens;
        if (total_tokens > _config.maxseq) return false;
    }

    _past_len = 0;
    for (size_t layer = 0; layer < _config.nlayer; ++layer) {
        size_t offset = 0;
        for (auto *blk : ctx->chain) {
            const size_t used = blk->used_tokens;
            if (used == 0) continue;
            if (layer >= blk->k_layers.size() || layer >= blk->v_layers.size()) return false;
            auto *k_block = blk->k_layers[layer];
            auto *v_block = blk->v_layers[layer];
            if (!k_block || !v_block) return false;

            llaisysTensor_t src_k = tensorSlice(k_block, 0, 0, used);
            llaisysTensor_t src_v = tensorSlice(v_block, 0, 0, used);
            llaisysTensor_t dst_k = tensorSlice(_k_cache[layer], 0, offset, offset + used);
            llaisysTensor_t dst_v = tensorSlice(_v_cache[layer], 0, offset, offset + used);
            if (!src_k || !src_v || !dst_k || !dst_v) {
                destroy_if_not_null(src_k);
                destroy_if_not_null(src_v);
                destroy_if_not_null(dst_k);
                destroy_if_not_null(dst_v);
                return false;
            }
            ::llaisysRearrange(dst_k, src_k);
            ::llaisysRearrange(dst_v, src_v);
            tensorDestroy(src_k);
            tensorDestroy(src_v);
            tensorDestroy(dst_k);
            tensorDestroy(dst_v);
            offset += used;
        }
    }

    _past_len = total_tokens;
    _external_past_len = total_tokens;
    _external_cache_ready = true;
    return true;
}

bool Decoder::runHidden(const int64_t *token_ids,
                        size_t ntoken,
                        bool append_only,
                        const int64_t *segment_offsets,
                        size_t nseg,
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
    const bool segmented = (segment_offsets != nullptr && nseg > 0);
    if (segmented && append_only) return false;

    if (!segmented) {
        ensureCache();
        if (_external_kv_ctx && !_external_cache_ready) {
            if (!recoverExternalCache()) {
                clearExternalKVContext();
                _past_len = 0;
            }
        }
    }
    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    // Segmented packed prefill treats each call as an independent packed forward.
    // Reusing decoder KV cache here breaks offset domains (past_len vs packed offsets).
    const bool can_cache = (!segmented) && _cache_inited && _config.maxseq > 0;
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
        if (segmented) {
            ::llaisysSelfAttentionSegmented(
                attn_out3d,
                q_rope,
                k_attn,
                v_attn,
                scale,
                segment_offsets,
                segment_offsets,
                nseg);
        } else {
            ::llaisysSelfAttention(attn_out3d, q_rope, k_attn, v_attn, scale);
        }
        if (k_cache_view) tensorDestroy(k_cache_view);
        if (v_cache_view) tensorDestroy(v_cache_view);

        trace("attn.proj");
        llaisysTensor_t attn_out2d = tensorView(attn_out3d, q2d_shape, 2);
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

        // Tensor parallel: allreduce after attn_o projection
        if (_tp_size > 1 && _comm) {
            size_t ndim = tensorGetNdim(proj_out);
            size_t shape[4];
            tensorGetShape(proj_out, shape);
            size_t count = 1;
            for (size_t d = 0; d < ndim; ++d) count *= shape[d];
            auto backend = (_device == LLAISYS_DEVICE_ILUVATAR) ? LLAISYS_COMM_IXCCL : LLAISYS_COMM_NCCL;
            auto *api = llaisys::device::getCommAPI(backend);
            api->allreduce(tensorGetData(proj_out), tensorGetData(proj_out),
                           count, _config.dtype, LLAISYS_REDUCE_SUM, _comm, _comm_stream);
        }

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

        // Tensor parallel: allreduce after mlp_down projection
        if (_tp_size > 1 && _comm) {
            size_t ndim = tensorGetNdim(mlp_out);
            size_t shape[4];
            tensorGetShape(mlp_out, shape);
            size_t count = 1;
            for (size_t d = 0; d < ndim; ++d) count *= shape[d];
            auto backend = (_device == LLAISYS_DEVICE_ILUVATAR) ? LLAISYS_COMM_IXCCL : LLAISYS_COMM_NCCL;
            auto *api = llaisys::device::getCommAPI(backend);
            api->allreduce(tensorGetData(mlp_out), tensorGetData(mlp_out),
                           count, _config.dtype, LLAISYS_REDUCE_SUM, _comm, _comm_stream);
        }

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
    if (!runHidden(token_ids, ntoken, false, nullptr, 0, past_len, cur_len, idx, pos_ids, hidden)) return false;

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

bool Decoder::prefillPacked(const int64_t *token_ids,
                            size_t ntoken,
                            const int64_t *token_offsets,
                            size_t nseq,
                            llaisysTensor_t out_last_logits) {
    if (!out_last_logits || !token_ids || !token_offsets || nseq == 0 || ntoken == 0) return false;
    if (!ensure_data(out_last_logits, "head.packed.logits.out")) return false;
    if (tensorGetNdim(out_last_logits) != 2) return false;
    size_t out_shape[2] = {0, 0};
    tensorGetShape(out_last_logits, out_shape);
    if (out_shape[0] != nseq || out_shape[1] != _config.voc) return false;
    if (token_offsets[0] != 0 || static_cast<size_t>(token_offsets[nseq]) != ntoken) return false;
    for (size_t i = 0; i < nseq; ++i) {
        if (token_offsets[i] > token_offsets[i + 1]) return false;
        if (token_offsets[i] == token_offsets[i + 1]) return false;
    }

    size_t past_len = 0;
    size_t cur_len = 0;
    llaisysTensor_t idx = nullptr;
    llaisysTensor_t pos_ids = nullptr;
    llaisysTensor_t hidden = nullptr;
    if (!runHidden(token_ids, ntoken, false, token_offsets, nseq, past_len, cur_len, idx, pos_ids, hidden)) return false;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    if (!_weights || !_weights->out_norm_w || !_weights->out_embed) {
        tensorDestroy(idx);
        tensorDestroy(pos_ids);
        tensorDestroy(hidden);
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < nseq && ok; ++i) {
        const size_t seg_end = static_cast<size_t>(token_offsets[i + 1]);
        const size_t last_pos = seg_end - 1;
        llaisysTensor_t last_hidden = tensorSlice(hidden, 0, last_pos, last_pos + 1);
        llaisysTensor_t row_logits = tensorSlice(out_last_logits, 0, i, i + 1);
        size_t last_shape[2] = {1, _config.hs};
        llaisysTensor_t final_norm = tensorCreate(last_shape, 2, _config.dtype, _device, device_id);
        if (!last_hidden || !row_logits || !final_norm) {
            ok = false;
        } else {
            ::llaisysRmsNorm(final_norm, last_hidden, _weights->out_norm_w, _config.epsilon);
            ::llaisysLinear(row_logits, final_norm, _weights->out_embed, nullptr);
        }
        destroy_if_not_null(last_hidden);
        destroy_if_not_null(row_logits);
        destroy_if_not_null(final_norm);
    }

    tensorDestroy(idx);
    tensorDestroy(pos_ids);
    tensorDestroy(hidden);
    return ok;
}

bool Decoder::decodePacked(const int64_t *token_ids,
                           size_t nseq,
                           const std::vector<LlaisysQwen2KVContext *> &contexts,
                           llaisysTensor_t out_last_logits,
                           size_t block_tokens_hint) {
    if (!token_ids || nseq == 0 || contexts.size() != nseq || !out_last_logits) return false;
    if (!ensure_data(out_last_logits, "head.decode_packed.logits.out")) return false;
    if (tensorGetNdim(out_last_logits) != 2) return false;
    size_t out_shape[2] = {0, 0};
    tensorGetShape(out_last_logits, out_shape);
    if (out_shape[0] != nseq || out_shape[1] != _config.voc) return false;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    std::vector<size_t> past_lens(nseq, 0);
    std::vector<int64_t> q_offsets(nseq + 1, 0);
    std::vector<int64_t> kv_offsets(nseq + 1, 0);
    std::vector<LlaisysQwen2KVBlock *> append_blocks(nseq, nullptr);
    std::vector<size_t> append_pos(nseq, 0);
    size_t kv_total = 0;

    for (size_t i = 0; i < nseq; ++i) {
        auto *ctx = contexts[i];
        if (!ctx) return false;
        if (ctx->dtype != _config.dtype || ctx->device != _device || ctx->device_id != device_id) return false;
        if (ctx->nlayer != _config.nlayer || ctx->nkvh != _config.nkvh || ctx->dh != _config.dh) return false;
        const size_t past = llaisysQwen2KVContextTokenCount(ctx);
        if (past + 1 > _config.maxseq) return false;
        past_lens[i] = past;
        q_offsets[i + 1] = static_cast<int64_t>(i + 1);
        kv_total += past + 1;
        kv_offsets[i + 1] = static_cast<int64_t>(kv_total);

        LlaisysQwen2KVBlock *target = nullptr;
        size_t pos = 0;
        if (!ctx->chain.empty()) {
            auto *last = ctx->chain.back();
            if (last && last->used_tokens < last->meta.max_tokens) {
                target = last;
                pos = last->used_tokens;
            }
        }
        if (!target) {
            const size_t max_tokens = block_tokens_hint > 0 ? block_tokens_hint : 64;
            LlaisysQwen2KVBlockMeta meta{};
            meta.dtype = _config.dtype;
            meta.nlayer = _config.nlayer;
            meta.nh = _config.nh;
            meta.nkvh = _config.nkvh;
            meta.dh = _config.dh;
            meta.max_tokens = max_tokens;
            auto *blk = llaisysQwen2KVBlockCreate(&meta, _device, device_id);
            if (!blk) return false;
            if (llaisysQwen2KVContextAttachBlock(ctx, blk) != 0) {
                llaisysQwen2KVBlockRelease(blk);
                return false;
            }
            llaisysQwen2KVBlockRelease(blk);
            if (ctx->chain.empty() || !ctx->chain.back()) return false;
            target = ctx->chain.back();
            pos = target->used_tokens;
        }
        append_blocks[i] = target;
        append_pos[i] = pos;
    }

    size_t idx_shape[1] = {nseq};
    size_t hidden_shape[2] = {nseq, _config.hs};
    llaisysTensor_t idx = tensorCreate(idx_shape, 1, LLAISYS_DTYPE_I64, _device, device_id);
    llaisysTensor_t pos_ids = tensorCreate(idx_shape, 1, LLAISYS_DTYPE_I64, _device, device_id);
    llaisysTensor_t hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
    if (!idx || !pos_ids || !hidden) {
        destroy_if_not_null(idx);
        destroy_if_not_null(pos_ids);
        destroy_if_not_null(hidden);
        return false;
    }
    tensorLoad(idx, token_ids);
    std::vector<int64_t> pos_buf(nseq, 0);
    for (size_t i = 0; i < nseq; ++i) pos_buf[i] = static_cast<int64_t>(past_lens[i]);
    tensorLoad(pos_ids, pos_buf.data());
    ::llaisysEmbedding(hidden, idx, _weights->in_embed);

    const float scale = 1.0f / std::sqrt(static_cast<float>(_config.dh));
    bool ok = true;
    for (size_t layer = 0; layer < _config.nlayer && ok; ++layer) {
        llaisysTensor_t norm = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
        size_t q2d_shape[2] = {nseq, _config.nh * _config.dh};
        size_t kv2d_shape[2] = {nseq, _config.nkvh * _config.dh};
        llaisysTensor_t q2d = tensorCreate(q2d_shape, 2, _config.dtype, _device, device_id);
        llaisysTensor_t k2d = tensorCreate(kv2d_shape, 2, _config.dtype, _device, device_id);
        llaisysTensor_t v2d = tensorCreate(kv2d_shape, 2, _config.dtype, _device, device_id);
        if (!norm || !q2d || !k2d || !v2d) {
            destroy_if_not_null(norm);
            destroy_if_not_null(q2d);
            destroy_if_not_null(k2d);
            destroy_if_not_null(v2d);
            ok = false;
            break;
        }
        ::llaisysRmsNorm(norm, hidden, _weights->attn_norm_w[layer], _config.epsilon);
        llaisysTensor_t q_bias = (_weights->attn_q_b && _weights->attn_q_b[layer]) ? _weights->attn_q_b[layer] : nullptr;
        llaisysTensor_t k_bias = (_weights->attn_k_b && _weights->attn_k_b[layer]) ? _weights->attn_k_b[layer] : nullptr;
        llaisysTensor_t v_bias = (_weights->attn_v_b && _weights->attn_v_b[layer]) ? _weights->attn_v_b[layer] : nullptr;
        ::llaisysLinear(q2d, norm, _weights->attn_q_w[layer], q_bias);
        ::llaisysLinear(k2d, norm, _weights->attn_k_w[layer], k_bias);
        ::llaisysLinear(v2d, norm, _weights->attn_v_w[layer], v_bias);

        size_t q3d_shape[3] = {nseq, _config.nh, _config.dh};
        size_t k3d_shape[3] = {nseq, _config.nkvh, _config.dh};
        llaisysTensor_t q3d = tensorView(q2d, q3d_shape, 3);
        llaisysTensor_t k3d = tensorView(k2d, k3d_shape, 3);
        llaisysTensor_t v3d = tensorView(v2d, k3d_shape, 3);
        llaisysTensor_t q_rope = tensorCreate(q3d_shape, 3, _config.dtype, _device, device_id);
        llaisysTensor_t k_rope = tensorCreate(k3d_shape, 3, _config.dtype, _device, device_id);
        if (!q3d || !k3d || !v3d || !q_rope || !k_rope) {
            destroy_if_not_null(norm);
            destroy_if_not_null(q2d);
            destroy_if_not_null(k2d);
            destroy_if_not_null(v2d);
            destroy_if_not_null(q3d);
            destroy_if_not_null(k3d);
            destroy_if_not_null(v3d);
            destroy_if_not_null(q_rope);
            destroy_if_not_null(k_rope);
            ok = false;
            break;
        }
        ::llaisysROPE(q_rope, q3d, pos_ids, _config.theta);
        ::llaisysROPE(k_rope, k3d, pos_ids, _config.theta);

        size_t kv_all_shape[3] = {kv_total, _config.nkvh, _config.dh};
        llaisysTensor_t k_all = tensorCreate(kv_all_shape, 3, _config.dtype, _device, device_id);
        llaisysTensor_t v_all = tensorCreate(kv_all_shape, 3, _config.dtype, _device, device_id);
        if (!k_all || !v_all) {
            destroy_if_not_null(norm);
            destroy_if_not_null(q2d);
            destroy_if_not_null(k2d);
            destroy_if_not_null(v2d);
            destroy_if_not_null(q3d);
            destroy_if_not_null(k3d);
            destroy_if_not_null(v3d);
            destroy_if_not_null(q_rope);
            destroy_if_not_null(k_rope);
            destroy_if_not_null(k_all);
            destroy_if_not_null(v_all);
            ok = false;
            break;
        }
        for (size_t i = 0; i < nseq && ok; ++i) {
            auto *ctx = contexts[i];
            const size_t kv_begin = static_cast<size_t>(kv_offsets[i]);
            const size_t past = past_lens[i];
            size_t copied = 0;
            for (auto *blk : ctx->chain) {
                if (!blk) {
                    ok = false;
                    break;
                }
                const size_t used = blk->used_tokens;
                if (used == 0) continue;
                llaisysTensor_t src_k = tensorSlice(blk->k_layers[layer], 0, 0, used);
                llaisysTensor_t src_v = tensorSlice(blk->v_layers[layer], 0, 0, used);
                llaisysTensor_t dst_k = tensorSlice(k_all, 0, kv_begin + copied, kv_begin + copied + used);
                llaisysTensor_t dst_v = tensorSlice(v_all, 0, kv_begin + copied, kv_begin + copied + used);
                if (!src_k || !src_v || !dst_k || !dst_v) {
                    destroy_if_not_null(src_k);
                    destroy_if_not_null(src_v);
                    destroy_if_not_null(dst_k);
                    destroy_if_not_null(dst_v);
                    ok = false;
                    break;
                }
                ::llaisysRearrange(dst_k, src_k);
                ::llaisysRearrange(dst_v, src_v);
                tensorDestroy(src_k);
                tensorDestroy(src_v);
                tensorDestroy(dst_k);
                tensorDestroy(dst_v);
                copied += used;
            }
            if (!ok || copied != past) {
                ok = false;
                break;
            }

            const size_t kv_new_pos = kv_begin + past;
            llaisysTensor_t src_new_k = tensorSlice(k_rope, 0, i, i + 1);
            llaisysTensor_t src_new_v = tensorSlice(v3d, 0, i, i + 1);
            llaisysTensor_t dst_new_k = tensorSlice(k_all, 0, kv_new_pos, kv_new_pos + 1);
            llaisysTensor_t dst_new_v = tensorSlice(v_all, 0, kv_new_pos, kv_new_pos + 1);
            llaisysTensor_t dst_ctx_k = tensorSlice(append_blocks[i]->k_layers[layer], 0, append_pos[i], append_pos[i] + 1);
            llaisysTensor_t dst_ctx_v = tensorSlice(append_blocks[i]->v_layers[layer], 0, append_pos[i], append_pos[i] + 1);
            if (!src_new_k || !src_new_v || !dst_new_k || !dst_new_v || !dst_ctx_k || !dst_ctx_v) {
                destroy_if_not_null(src_new_k);
                destroy_if_not_null(src_new_v);
                destroy_if_not_null(dst_new_k);
                destroy_if_not_null(dst_new_v);
                destroy_if_not_null(dst_ctx_k);
                destroy_if_not_null(dst_ctx_v);
                ok = false;
                break;
            }
            ::llaisysRearrange(dst_new_k, src_new_k);
            ::llaisysRearrange(dst_new_v, src_new_v);
            ::llaisysRearrange(dst_ctx_k, src_new_k);
            ::llaisysRearrange(dst_ctx_v, src_new_v);
            tensorDestroy(src_new_k);
            tensorDestroy(src_new_v);
            tensorDestroy(dst_new_k);
            tensorDestroy(dst_new_v);
            tensorDestroy(dst_ctx_k);
            tensorDestroy(dst_ctx_v);
        }

        llaisysTensor_t attn_out3d = nullptr;
        llaisysTensor_t attn_out2d = nullptr;
        llaisysTensor_t proj_out = nullptr;
        llaisysTensor_t attn_hidden = nullptr;
        llaisysTensor_t mlp_norm = nullptr;
        llaisysTensor_t gate = nullptr;
        llaisysTensor_t up = nullptr;
        llaisysTensor_t swiglu = nullptr;
        llaisysTensor_t mlp_out = nullptr;
        llaisysTensor_t mlp_hidden = nullptr;

        if (ok) {
            attn_out3d = tensorCreate(q3d_shape, 3, _config.dtype, _device, device_id);
            if (!attn_out3d) ok = false;
        }
        if (ok) {
            ::llaisysSelfAttentionSegmented(
                attn_out3d, q_rope, k_all, v_all, scale, q_offsets.data(), kv_offsets.data(), nseq);
            attn_out2d = tensorView(attn_out3d, q2d_shape, 2);
            proj_out = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);            attn_hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
            if (!attn_out2d || !proj_out || !attn_hidden) ok = false;
        }
        if (ok) {
            ::llaisysLinear(proj_out, attn_out2d, _weights->attn_o_w[layer], nullptr);
            ::llaisysAdd(attn_hidden, hidden, proj_out);
        }

        if (ok) {
            mlp_norm = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
            size_t mlp_shape[2] = {nseq, _config.di};
            gate = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
            up = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
            swiglu = tensorCreate(mlp_shape, 2, _config.dtype, _device, device_id);
            mlp_out = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
            mlp_hidden = tensorCreate(hidden_shape, 2, _config.dtype, _device, device_id);
            if (!mlp_norm || !gate || !up || !swiglu || !mlp_out || !mlp_hidden) {
                ok = false;
            }
        }
        if (ok) {
            ::llaisysRmsNorm(mlp_norm, attn_hidden, _weights->mlp_norm_w[layer], _config.epsilon);
            ::llaisysLinear(gate, mlp_norm, _weights->mlp_gate_w[layer], nullptr);
            ::llaisysLinear(up, mlp_norm, _weights->mlp_up_w[layer], nullptr);
            ::llaisysSwiGLU(swiglu, gate, up);
            ::llaisysLinear(mlp_out, swiglu, _weights->mlp_down_w[layer], nullptr);
            ::llaisysAdd(mlp_hidden, attn_hidden, mlp_out);
        }

        if (ok) {
            tensorDestroy(hidden);
            hidden = mlp_hidden;
            mlp_hidden = nullptr;
        }

        destroy_if_not_null(norm);
        destroy_if_not_null(q2d);
        destroy_if_not_null(k2d);
        destroy_if_not_null(v2d);
        destroy_if_not_null(q3d);
        destroy_if_not_null(k3d);
        destroy_if_not_null(v3d);
        destroy_if_not_null(q_rope);
        destroy_if_not_null(k_rope);
        destroy_if_not_null(k_all);
        destroy_if_not_null(v_all);
        destroy_if_not_null(attn_out3d);
        destroy_if_not_null(attn_out2d);
        destroy_if_not_null(proj_out);
        destroy_if_not_null(attn_hidden);
        destroy_if_not_null(mlp_norm);
        destroy_if_not_null(gate);
        destroy_if_not_null(up);
        destroy_if_not_null(swiglu);
        destroy_if_not_null(mlp_out);
        destroy_if_not_null(mlp_hidden);
    }

    if (ok) {
        for (size_t i = 0; i < nseq; ++i) {
            if (append_blocks[i] && append_blocks[i]->used_tokens < append_pos[i] + 1) {
                append_blocks[i]->used_tokens = append_pos[i] + 1;
            }
        }
        for (size_t i = 0; i < nseq && ok; ++i) {
            llaisysTensor_t last_hidden = tensorSlice(hidden, 0, i, i + 1);
            llaisysTensor_t row_logits = tensorSlice(out_last_logits, 0, i, i + 1);
            size_t last_shape[2] = {1, _config.hs};
            llaisysTensor_t final_norm = tensorCreate(last_shape, 2, _config.dtype, _device, device_id);
            if (!last_hidden || !row_logits || !final_norm) {
                ok = false;
            } else {
                ::llaisysRmsNorm(final_norm, last_hidden, _weights->out_norm_w, _config.epsilon);
                ::llaisysLinear(row_logits, final_norm, _weights->out_embed, nullptr);
            }
            destroy_if_not_null(last_hidden);
            destroy_if_not_null(row_logits);
            destroy_if_not_null(final_norm);
        }
    }

    tensorDestroy(idx);
    tensorDestroy(pos_ids);
    tensorDestroy(hidden);
    return ok;
}

bool Decoder::decodeStep(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits) {
    if (!out_last_logits) return false;
    if (!ensure_data(out_last_logits, "head.logits.out")) return false;

    size_t past_len = 0;
    size_t cur_len = 0;
    llaisysTensor_t idx = nullptr;
    llaisysTensor_t pos_ids = nullptr;
    llaisysTensor_t hidden = nullptr;
    if (!runHidden(token_ids, ntoken, true, nullptr, 0, past_len, cur_len, idx, pos_ids, hidden)) return false;

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
