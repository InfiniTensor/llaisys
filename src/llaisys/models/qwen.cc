extern "C" {

struct LlaisysQwen2Model {
    Qwen2Impl* impl;
};

__export struct LlaisysQwen2Model *
llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                        llaisysDeviceType_t device,
                        int* /*device_ids*/, int /*ndevice*/) noexcept {
    try {
        CHECK_ARGUMENT(meta != nullptr, "meta is null");

        auto* model = new LlaisysQwen2Model;
        model->impl = new Qwen2Impl;
        auto* m = model->impl;

        m->meta = *meta;
        m->device = device;
        m->device_id = 0;
        m->cur_pos = 0;

        llaisys::core::context().setDevice(device, 0);

        alloc_weights(m);
        alloc_kv_tmp(m, meta->maxseq);

        return model;
    } catch (...) {
        return nullptr;
    }
}

__export void
llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) noexcept {
    try {
        if (!model) return;
        if (model->impl) {
            free_weights(model->impl);
            delete model->impl;
        }
        delete model;
    } catch (...) {
        // ignore
    }
}

__export struct LlaisysQwen2Weights *
llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) noexcept {
    return model ? &model->impl->weights : nullptr;
}

__export int64_t
llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) noexcept {
    try {
        CHECK_ARGUMENT(model && model->impl, "model is null");
        CHECK_ARGUMENT(token_ids != nullptr, "token_ids is null");
        CHECK_ARGUMENT(ntoken > 0, "ntoken must be > 0");

        auto* m = model->impl;
        auto& meta = m->meta;
        auto& w = m->weights;
        auto& t = m->tmp;

        CHECK_ARGUMENT(m->cur_pos + ntoken < meta.maxseq, "sequence length exceeds maxseq");

        const size_t dh = meta.dh;
        const float scale = 1.0f / std::sqrt((float)dh);

        // ---- 写 tok/pos ----
        std::memcpy(t.tok_i64->data(), token_ids, ntoken * sizeof(int64_t));
        {
            int64_t* p = reinterpret_cast<int64_t*>(t.pos_i64->data());
            for (size_t i = 0; i < ntoken; ++i) p[i] = (int64_t)(m->cur_pos + i);
        }

        tensor_t tok_L = t.tok_i64->slice(0, 0, ntoken);
        tensor_t pos_L = t.pos_i64->slice(0, 0, ntoken);

        tensor_t x_L  = t.x_a->slice(0, 0, ntoken);
        tensor_t xb_L = t.x_b->slice(0, 0, ntoken);
        tensor_t h_L  = t.h->slice(0, 0, ntoken);

        require_contiguous(tok_L, "tok_L must be contiguous");
        require_contiguous(pos_L, "pos_L must be contiguous");
        require_contiguous(x_L,   "x_L must be contiguous");

        // ---- embedding ----
        llaisys::ops::embedding(x_L, tok_L, w.in_embed->tensor);

        DBG_TENSOR("x after embedding (first token)", x_L->slice(0, 0, 1));

        // ---- layers ----
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            // attn rmsnorm
            llaisys::ops::rms_norm(h_L, x_L, w.attn_norm_w[layer]->tensor, meta.epsilon);

            // qkv
            tensor_t q1_L = t.q1->slice(0, 0, ntoken);
            tensor_t k1_L = t.k1->slice(0, 0, ntoken);
            tensor_t v1_L = t.v1->slice(0, 0, ntoken);

            llaisys::ops::linear(q1_L, h_L, w.attn_q_w[layer]->tensor, w.attn_q_b[layer]->tensor);
            llaisys::ops::linear(k1_L, h_L, w.attn_k_w[layer]->tensor, w.attn_k_b[layer]->tensor);
            llaisys::ops::linear(v1_L, h_L, w.attn_v_w[layer]->tensor, w.attn_v_b[layer]->tensor);

            // rearrange to [L, head, dh]
            tensor_t q_L = t.q->slice(0, 0, ntoken);
            tensor_t k_L = t.k->slice(0, 0, ntoken);
            tensor_t v_L = t.v->slice(0, 0, ntoken);

            llaisys::ops::rearrange(q_L, q1_L);
            llaisys::ops::rearrange(k_L, k1_L);
            llaisys::ops::rearrange(v_L, v1_L);

            // rope
            tensor_t q_rope_L = t.q_rope->slice(0, 0, ntoken);
            tensor_t k_rope_L = t.k_rope->slice(0, 0, ntoken);
            llaisys::ops::rope(q_rope_L, q_L, pos_L, meta.theta);
            llaisys::ops::rope(k_rope_L, k_L, pos_L, meta.theta);

            if (LLAISYS_QWEN2_DEBUG && layer == 0) {
                DBG_TENSOR("layer0 q_rope (first token)", q_rope_L->slice(0, 0, 1));
                DBG_TENSOR("layer0 k_rope (first token)", k_rope_L->slice(0, 0, 1));
            }

            // write KV cache
            tensor_t kc_win = m->kv[layer].k->slice(0, m->cur_pos, m->cur_pos + ntoken);
            tensor_t vc_win = m->kv[layer].v->slice(0, m->cur_pos, m->cur_pos + ntoken);
            llaisys::ops::rearrange(kc_win, k_rope_L);
            llaisys::ops::rearrange(vc_win, v_L);

            // history KV for attention
            const size_t total_len = m->cur_pos + ntoken;
            tensor_t k_hist = m->kv[layer].k->slice(0, 0, total_len);
            tensor_t v_hist = m->kv[layer].v->slice(0, 0, total_len);

            tensor_t attn_val_L = t.attn_val->slice(0, 0, ntoken);
            llaisys::ops::self_attention(attn_val_L, q_rope_L, k_hist, v_hist, scale);

            // merge heads
            tensor_t attn_merge_L = t.attn_merge->slice(0, 0, ntoken);
            llaisys::ops::rearrange(attn_merge_L, attn_val_L);

            // o proj
            tensor_t attn_out_L = t.attn_out->slice(0, 0, ntoken);
            llaisys::ops::linear(attn_out_L, attn_merge_L, w.attn_o_w[layer]->tensor, nullptr);

            // residual
            llaisys::ops::add(xb_L, x_L, attn_out_L);
            std::swap(t.x_a, t.x_b);
            x_L  = t.x_a->slice(0, 0, ntoken);
            xb_L = t.x_b->slice(0, 0, ntoken);

            // MLP
            tensor_t mlp_in_L = t.mlp_in->slice(0, 0, ntoken);
            llaisys::ops::rms_norm(mlp_in_L, x_L, w.mlp_norm_w[layer]->tensor, meta.epsilon);

            tensor_t gate_L = t.gate->slice(0, 0, ntoken);
            tensor_t up_L   = t.up->slice(0, 0, ntoken);
            llaisys::ops::linear(gate_L, mlp_in_L, w.mlp_gate_w[layer]->tensor, nullptr);
            llaisys::ops::linear(up_L,   mlp_in_L, w.mlp_up_w[layer]->tensor,   nullptr);

            tensor_t act_L = t.act->slice(0, 0, ntoken);
            llaisys::ops::swiglu(act_L, gate_L, up_L);

            tensor_t mlp_out_L = t.mlp_out->slice(0, 0, ntoken);
            llaisys::ops::linear(mlp_out_L, act_L, w.mlp_down_w[layer]->tensor, nullptr);

            // residual
            llaisys::ops::add(xb_L, x_L, mlp_out_L);
            std::swap(t.x_a, t.x_b);
            x_L  = t.x_a->slice(0, 0, ntoken);
            xb_L = t.x_b->slice(0, 0, ntoken);
        }

        // final norm
        tensor_t y_L = t.y->slice(0, 0, ntoken);
        llaisys::ops::rms_norm(y_L, x_L, w.out_norm_w->tensor, meta.epsilon);

        // last token hidden
        tensor_t y_last = y_L->slice(0, ntoken - 1, ntoken);

        // logits
        llaisys::ops::linear(t.logits, y_last, w.out_embed->tensor, nullptr);
        DBG_TENSOR("logits (shape [1, voc])", t.logits);

        // argmax
        llaisys::ops::argmax(t.max_idx, t.max_val, t.logits);

        int64_t next = *reinterpret_cast<const int64_t*>(t.max_idx->data());

        m->cur_pos += ntoken;
        return next;
    } catch (...) {
        return -1;  // 错误返回 -1
    }
}

} // extern "C"
