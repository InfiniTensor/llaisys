#include "qwen2.hpp"
#include <cstring>
#include <cmath>

namespace llaisys {
namespace models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta* meta_, llaisysDeviceType_t device, int dev_id)
    : device_type(device), device_id(dev_id), cache_len(0) {
    std::memcpy(&meta, meta_, sizeof(LlaisysQwen2Meta));
    allocateWeights();
    allocateCache();
    allocateBuffers(meta.maxseq);
}

void Qwen2Model::allocateWeights() {
    auto dtype = meta.dtype;
    size_t nlayer = meta.nlayer;
    size_t hs = meta.hs;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t di = meta.di;
    size_t voc = meta.voc;

    // Embedding weights
    in_embed = Tensor::create({voc, hs}, dtype, device_type, device_id);
    out_embed = Tensor::create({voc, hs}, dtype, device_type, device_id);
    out_norm_w = Tensor::create({hs}, dtype, device_type, device_id);

    // Per-layer weights
    attn_norm_w.resize(nlayer);
    attn_q_w.resize(nlayer);
    attn_q_b.resize(nlayer);
    attn_k_w.resize(nlayer);
    attn_k_b.resize(nlayer);
    attn_v_w.resize(nlayer);
    attn_v_b.resize(nlayer);
    attn_o_w.resize(nlayer);
    mlp_norm_w.resize(nlayer);
    mlp_gate_w.resize(nlayer);
    mlp_up_w.resize(nlayer);
    mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        attn_norm_w[i] = Tensor::create({hs}, dtype, device_type, device_id);
        attn_q_w[i] = Tensor::create({nh * dh, hs}, dtype, device_type, device_id);
        attn_q_b[i] = Tensor::create({nh * dh}, dtype, device_type, device_id);
        attn_k_w[i] = Tensor::create({nkvh * dh, hs}, dtype, device_type, device_id);
        attn_k_b[i] = Tensor::create({nkvh * dh}, dtype, device_type, device_id);
        attn_v_w[i] = Tensor::create({nkvh * dh, hs}, dtype, device_type, device_id);
        attn_v_b[i] = Tensor::create({nkvh * dh}, dtype, device_type, device_id);
        attn_o_w[i] = Tensor::create({hs, nh * dh}, dtype, device_type, device_id);
        mlp_norm_w[i] = Tensor::create({hs}, dtype, device_type, device_id);
        mlp_gate_w[i] = Tensor::create({di, hs}, dtype, device_type, device_id);
        mlp_up_w[i] = Tensor::create({di, hs}, dtype, device_type, device_id);
        mlp_down_w[i] = Tensor::create({hs, di}, dtype, device_type, device_id);
    }
}

void Qwen2Model::allocateCache() {
    size_t nlayer = meta.nlayer;
    size_t maxseq = meta.maxseq;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    auto dtype = meta.dtype;

    k_cache.resize(nlayer);
    v_cache.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        k_cache[i] = Tensor::create({maxseq, nkvh, dh}, dtype, device_type, device_id);
        v_cache[i] = Tensor::create({maxseq, nkvh, dh}, dtype, device_type, device_id);
    }
}

void Qwen2Model::allocateBuffers(size_t max_seqlen) {
    auto dtype = meta.dtype;
    size_t hs = meta.hs;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t di = meta.di;
    size_t voc = meta.voc;

    hidden = Tensor::create({max_seqlen, hs}, dtype, device_type, device_id);
    hidden_norm = Tensor::create({max_seqlen, hs}, dtype, device_type, device_id);
    q = Tensor::create({max_seqlen, nh * dh}, dtype, device_type, device_id);
    k = Tensor::create({max_seqlen, nkvh * dh}, dtype, device_type, device_id);
    v = Tensor::create({max_seqlen, nkvh * dh}, dtype, device_type, device_id);
    q_rope = Tensor::create({max_seqlen, nh, dh}, dtype, device_type, device_id);
    k_rope = Tensor::create({max_seqlen, nkvh, dh}, dtype, device_type, device_id);
    attn_out = Tensor::create({max_seqlen, nh, dh}, dtype, device_type, device_id);
    attn_proj = Tensor::create({max_seqlen, hs}, dtype, device_type, device_id);
    gate = Tensor::create({max_seqlen, di}, dtype, device_type, device_id);
    up = Tensor::create({max_seqlen, di}, dtype, device_type, device_id);
    mlp_out = Tensor::create({max_seqlen, hs}, dtype, device_type, device_id);
    logits = Tensor::create({1, voc}, dtype, device_type, device_id);
    max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
    max_val = Tensor::create({1}, dtype, device_type, device_id);
    pos_ids = Tensor::create({max_seqlen}, LLAISYS_DTYPE_I64, device_type, device_id);
}

int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    // Create input tensor for embedding lookup
    tensor_t input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device_type, device_id);
    input_ids->load(token_ids);

    // Get hidden states view for current sequence
    tensor_t hidden_view = hidden->slice(0, 0, ntoken);
    tensor_t in_embed_view = in_embed;

    // Embedding lookup
    ops::embedding(hidden_view, input_ids, in_embed_view);

    // Forward pass through all layers
    forward(ntoken, cache_len);

    // Update cache length
    cache_len += ntoken;

    // Final layer norm on last token
    tensor_t last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    tensor_t last_norm = hidden_norm->slice(0, 0, 1);
    ops::rms_norm(last_norm, last_hidden, out_norm_w, meta.epsilon);

    // Compute logits for last token
    ops::linear(logits, last_norm, out_embed, nullptr);

    // Get last row of logits
    tensor_t last_logits = logits->view({meta.voc});

    // Argmax to get predicted token
    ops::argmax(max_idx, max_val, last_logits);

    // Read result
    int64_t result;
    if (device_type == LLAISYS_DEVICE_CPU) {
        result = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        // Copy from device to host
        core::context().setDevice(device_type, device_id);
        core::context().runtime().api()->memcpy_sync(
            &result, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    }

    return result;
}

void Qwen2Model::forward(size_t seqlen, size_t start_pos) {
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        forwardLayer(layer, seqlen, start_pos);
    }
}

void Qwen2Model::forwardLayer(size_t layer, size_t seqlen, size_t start_pos) {
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t total_len = start_pos + seqlen;

    // Get views for current sequence length
    tensor_t hidden_view = hidden->slice(0, 0, seqlen);
    tensor_t norm_view = hidden_norm->slice(0, 0, seqlen);
    tensor_t q_view = q->slice(0, 0, seqlen);
    tensor_t k_view = k->slice(0, 0, seqlen);
    tensor_t v_view = v->slice(0, 0, seqlen);
    tensor_t q_rope_view = q_rope->slice(0, 0, seqlen);
    tensor_t k_rope_view = k_rope->slice(0, 0, seqlen);
    tensor_t attn_out_view = attn_out->slice(0, 0, seqlen);
    tensor_t attn_proj_view = attn_proj->slice(0, 0, seqlen);
    tensor_t gate_view = gate->slice(0, 0, seqlen);
    tensor_t up_view = up->slice(0, 0, seqlen);
    tensor_t mlp_out_view = mlp_out->slice(0, 0, seqlen);

    // 1. Pre-attention layer norm
    ops::rms_norm(norm_view, hidden_view, attn_norm_w[layer], meta.epsilon);

    // 2. Compute Q, K, V projections
    ops::linear(q_view, norm_view, attn_q_w[layer], attn_q_b[layer]);
    ops::linear(k_view, norm_view, attn_k_w[layer], attn_k_b[layer]);
    ops::linear(v_view, norm_view, attn_v_w[layer], attn_v_b[layer]);

    // 3. Reshape Q, K, V to [seqlen, nhead, dh]
    tensor_t q_reshaped = q_view->view({seqlen, nh, dh});
    tensor_t k_reshaped = k_view->view({seqlen, nkvh, dh});
    tensor_t v_reshaped = v_view->view({seqlen, nkvh, dh});

    // 4. Set up position ids
    tensor_t pos_view = pos_ids->slice(0, 0, seqlen);
    std::vector<int64_t> pos_data(seqlen);
    for (size_t i = 0; i < seqlen; ++i) {
        pos_data[i] = static_cast<int64_t>(start_pos + i);
    }
    pos_view->load(pos_data.data());

    // 5. Apply RoPE
    ops::rope(q_rope_view, q_reshaped, pos_view, meta.theta);
    ops::rope(k_rope_view, k_reshaped, pos_view, meta.theta);

    // 6. Update KV cache
    tensor_t k_cache_slice = k_cache[layer]->slice(0, start_pos, total_len);
    tensor_t v_cache_slice = v_cache[layer]->slice(0, start_pos, total_len);

    // Copy new K, V to cache
    size_t kv_bytes = seqlen * nkvh * dh * k_rope_view->elementSize();
    if (device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(k_cache_slice->data(), k_rope_view->data(), kv_bytes);
        std::memcpy(v_cache_slice->data(), v_reshaped->data(), kv_bytes);
    } else {
        core::context().setDevice(device_type, device_id);
        auto api = core::context().runtime().api();
        api->memcpy_sync(k_cache_slice->data(), k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
        api->memcpy_sync(v_cache_slice->data(), v_reshaped->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    }

    // 7. Self-attention with full KV cache
    tensor_t k_full = k_cache[layer]->slice(0, 0, total_len);
    tensor_t v_full = v_cache[layer]->slice(0, 0, total_len);

    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    ops::self_attention(attn_out_view, q_rope_view, k_full, v_full, scale);

    // 8. Reshape attention output and project
    tensor_t attn_out_flat = attn_out_view->view({seqlen, nh * dh});
    ops::linear(attn_proj_view, attn_out_flat, attn_o_w[layer], nullptr);

    // 9. Residual connection
    ops::add(hidden_view, hidden_view, attn_proj_view);

    // 10. Post-attention layer norm
    ops::rms_norm(norm_view, hidden_view, mlp_norm_w[layer], meta.epsilon);

    // 11. MLP: gate and up projections
    ops::linear(gate_view, norm_view, mlp_gate_w[layer], nullptr);
    ops::linear(up_view, norm_view, mlp_up_w[layer], nullptr);

    // 12. SwiGLU activation
    ops::swiglu(gate_view, gate_view, up_view);

    // 13. Down projection (reuse attn_proj_view as output buffer, shape [seqlen, hs])
    ops::linear(attn_proj_view, gate_view, mlp_down_w[layer], nullptr);

    // 14. Residual connection
    ops::add(hidden_view, hidden_view, attn_proj_view);
}

}  // namespace models
}  // namespace llaisys
