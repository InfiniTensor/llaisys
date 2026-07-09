#include "qwen2.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : _meta(meta), _device(device), _device_id(0), _past_len(0) {
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU, "Qwen2Model: only CPU is supported in this implementation");
    if (device_ids != nullptr && ndevice > 0) {
        _device_id = device_ids[0];
    }
    CHECK_ARGUMENT(_meta.nlayer > 0, "Qwen2Model: nlayer must be > 0");
    CHECK_ARGUMENT(_meta.hs > 0, "Qwen2Model: hidden size must be > 0");
    CHECK_ARGUMENT(_meta.nh > 0, "Qwen2Model: nhead must be > 0");
    CHECK_ARGUMENT(_meta.dh > 0, "Qwen2Model: head dim must be > 0");
    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh, "Qwen2Model: hidden size must equal nhead * head_dim");

    _weights.in_embed = create_weight_tensor({_meta.voc, _meta.hs});
    _weights.out_embed = create_weight_tensor({_meta.voc, _meta.hs});
    _weights.out_norm_w = create_weight_tensor({_meta.hs});

    _weights.attn_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_q_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_q_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_k_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_k_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_v_w = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_v_b = new llaisysTensor_t[_meta.nlayer];
    _weights.attn_o_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_gate_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_up_w = new llaisysTensor_t[_meta.nlayer];
    _weights.mlp_down_w = new llaisysTensor_t[_meta.nlayer];

    const size_t qkv_dim = _meta.nh * _meta.dh;
    const size_t kv_dim = _meta.nkvh * _meta.dh;

    for (size_t i = 0; i < _meta.nlayer; i++) {
        _weights.attn_norm_w[i] = create_weight_tensor({_meta.hs});
        _weights.attn_q_w[i] = create_weight_tensor({qkv_dim, _meta.hs});
        _weights.attn_q_b[i] = create_weight_tensor({qkv_dim});
        _weights.attn_k_w[i] = create_weight_tensor({kv_dim, _meta.hs});
        _weights.attn_k_b[i] = create_weight_tensor({kv_dim});
        _weights.attn_v_w[i] = create_weight_tensor({kv_dim, _meta.hs});
        _weights.attn_v_b[i] = create_weight_tensor({kv_dim});
        _weights.attn_o_w[i] = create_weight_tensor({_meta.hs, qkv_dim});
        _weights.mlp_norm_w[i] = create_weight_tensor({_meta.hs});
        _weights.mlp_gate_w[i] = create_weight_tensor({_meta.di, _meta.hs});
        _weights.mlp_up_w[i] = create_weight_tensor({_meta.di, _meta.hs});
        _weights.mlp_down_w[i] = create_weight_tensor({_meta.hs, _meta.di});

        zero_tensor(_weights.attn_q_b[i]);
        zero_tensor(_weights.attn_k_b[i]);
        zero_tensor(_weights.attn_v_b[i]);
    }

    _zero_bias_hs = llaisys::Tensor::create({_meta.hs}, _meta.dtype, _device, _device_id);
    _zero_bias_di = llaisys::Tensor::create({_meta.di}, _meta.dtype, _device, _device_id);
    _zero_bias_voc = llaisys::Tensor::create({_meta.voc}, _meta.dtype, _device, _device_id);

    zero_tensor_data(_zero_bias_hs);
    zero_tensor_data(_zero_bias_di);
    zero_tensor_data(_zero_bias_voc);

    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _k_cache[i] = llaisys::Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        _v_cache[i] = llaisys::Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
    }
}

Qwen2Model::~Qwen2Model() {
    delete[] _weights.attn_norm_w;
    delete[] _weights.attn_q_w;
    delete[] _weights.attn_q_b;
    delete[] _weights.attn_k_w;
    delete[] _weights.attn_k_b;
    delete[] _weights.attn_v_w;
    delete[] _weights.attn_v_b;
    delete[] _weights.attn_o_w;
    delete[] _weights.mlp_norm_w;
    delete[] _weights.mlp_gate_w;
    delete[] _weights.mlp_up_w;
    delete[] _weights.mlp_down_w;

    for (auto *tensor : _owned_tensors) {
        delete tensor;
    }
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights;
}

llaisysTensor_t Qwen2Model::create_weight_tensor(const std::vector<size_t> &shape) {
    auto tensor = llaisys::Tensor::create(shape, _meta.dtype, _device, _device_id);
    auto *wrapped = new LlaisysTensor{tensor};
    _owned_tensors.push_back(wrapped);
    return wrapped;
}

llaisys::tensor_t Qwen2Model::unwrap(llaisysTensor_t tensor) const {
    return tensor->tensor;
}

void Qwen2Model::zero_tensor(llaisysTensor_t tensor) {
    zero_tensor_data(tensor->tensor);
}

void Qwen2Model::zero_tensor_data(const llaisys::tensor_t &tensor) {
    CHECK_ARGUMENT(tensor->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2Model: only CPU is supported for zeroing");
    const size_t bytes = tensor->numel() * tensor->elementSize();
    if (bytes == 0) {
        return;
    }
    std::memset(tensor->data(), 0, bytes);
}

void Qwen2Model::write_kv_cache(size_t layer, size_t pos, const llaisys::tensor_t &k, const llaisys::tensor_t &v) {
    CHECK_ARGUMENT(layer < _meta.nlayer, "Qwen2Model: layer out of range");
    CHECK_ARGUMENT(pos < _meta.maxseq, "Qwen2Model: position exceeds max sequence length");

    const size_t elem_bytes = k->elementSize();
    const size_t row_elems = _meta.nkvh * _meta.dh;
    const size_t row_bytes = row_elems * elem_bytes;

    std::byte *k_dst = _k_cache[layer]->data() + pos * row_bytes;
    std::byte *v_dst = _v_cache[layer]->data() + pos * row_bytes;

    std::memcpy(k_dst, k->data(), row_bytes);
    std::memcpy(v_dst, v->data(), row_bytes);
}

void Qwen2Model::reset_state() {
    _past_len = 0;
}

llaisys::tensor_t Qwen2Model::forward_token(int64_t token_id, size_t pos) {
    CHECK_ARGUMENT(pos < _meta.maxseq, "Qwen2Model: position exceeds max sequence length");

    auto token_tensor = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    token_tensor->load(&token_id);

	// x->(1, hs)
    auto x = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    llaisys::ops::embedding(x, token_tensor, unwrap(_weights.in_embed));

    auto pos_ids = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    int64_t pos_id = static_cast<int64_t>(pos);
    pos_ids->load(&pos_id);

    const float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

    for (size_t layer = 0; layer < _meta.nlayer; layer++) {
		// x_norm->(1, hs)
        auto x_norm = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
        llaisys::ops::rms_norm(x_norm, x, unwrap(_weights.attn_norm_w[layer]), _meta.epsilon);

		// q->(1, nh*dh), k->(1, nkvh*dh), v->(1, nkvh*dh)
		// for qwen2 q->(1, 1536), k,v->(1, 256)
        auto q = llaisys::Tensor::create({1, _meta.nh * _meta.dh}, _meta.dtype, _device, _device_id);
        auto k = llaisys::Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id);
        auto v = llaisys::Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id);

        // Wq->(1536, 1536), Wk->(256, 1536), Wv->(256, 1536)
        // x_norm->(1, 1536), q = x_norm * Wq^T + bq
        // q->(1, 1536), k,v->(1, 256)
        llaisys::ops::linear(q, x_norm, unwrap(_weights.attn_q_w[layer]), unwrap(_weights.attn_q_b[layer]));
        llaisys::ops::linear(k, x_norm, unwrap(_weights.attn_k_w[layer]), unwrap(_weights.attn_k_b[layer]));
        llaisys::ops::linear(v, x_norm, unwrap(_weights.attn_v_w[layer]), unwrap(_weights.attn_v_b[layer]));

		// q_view->(1, 12, 128), k_view,v_view->(1, 2, 128)
        auto q_view = q->view({1, _meta.nh, _meta.dh});
        auto k_view = k->view({1, _meta.nkvh, _meta.dh});
        auto v_view = v->view({1, _meta.nkvh, _meta.dh});

        auto q_rope = llaisys::Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        auto k_rope = llaisys::Tensor::create({1, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);

		// q_rope->(1, 12, 128), k_rope->(1, 2, 128)
        llaisys::ops::rope(q_rope, q_view, pos_ids, _meta.theta);
        llaisys::ops::rope(k_rope, k_view, pos_ids, _meta.theta);

        // kv_cache[layer]->(maxseq, nkvh, dh) = (131072, 2, 128)
        write_kv_cache(layer, pos, k_rope, v_view);

		// k_total->(n, 2, 128), v_total->(n, 2, 128)
        auto k_total = _k_cache[layer]->slice(0, 0, pos + 1);
        auto v_total = _v_cache[layer]->slice(0, 0, pos + 1);

		// attn_val->(1, nh, dh) = (1, 12, 128)
        auto attn_val = llaisys::Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
		// attn_val = q @ k^T / sqrt(dh) @v = (1, 12, 128) @ (n, 2, 128)^T @ (n, 2, 128) = (1, 12, 128)
        llaisys::ops::self_attention(attn_val, q_rope, k_total, v_total, scale);

		// attn_val_2d->(1, 1536)
        auto attn_val_2d = attn_val->view({1, _meta.hs});
		// attn_out = (1, 1536) attn_o_w->(1536, 1536)
        auto attn_out = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
        llaisys::ops::linear(attn_out, attn_val_2d, unwrap(_weights.attn_o_w[layer]), _zero_bias_hs);

		// x = x + attn_out ->(1, 1536)
        auto x_attn = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
        llaisys::ops::add(x_attn, x, attn_out);
        x = x_attn;

		// mlp_norm_w->(1, 1536)
        auto mlp_norm = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
        llaisys::ops::rms_norm(mlp_norm, x, unwrap(_weights.mlp_norm_w[layer]), _meta.epsilon);

		// mlp_gate_w->(8960, 1536), mlp_up_w->(8960, 1536)
        auto gate = llaisys::Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
        auto up = llaisys::Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
		// gate, up->(1, 8960)
        llaisys::ops::linear(gate, mlp_norm, unwrap(_weights.mlp_gate_w[layer]), _zero_bias_di);
        llaisys::ops::linear(up, mlp_norm, unwrap(_weights.mlp_up_w[layer]), _zero_bias_di);

		// swiglu_out->(1, 8960)
        auto swiglu_out = llaisys::Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_id);
        llaisys::ops::swiglu(swiglu_out, gate, up);

		// mlp_down_w->(1536, 8960)
        auto mlp_out = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
		// mlp_out->(1, 1536)
        llaisys::ops::linear(mlp_out, swiglu_out, unwrap(_weights.mlp_down_w[layer]), _zero_bias_hs);

        auto x_mlp = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
        llaisys::ops::add(x_mlp, x, mlp_out);
        x = x_mlp;
    }

    auto out_norm = llaisys::Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    llaisys::ops::rms_norm(out_norm, x, unwrap(_weights.out_norm_w), _meta.epsilon);

    auto logits_2d = llaisys::Tensor::create({1, _meta.voc}, _meta.dtype, _device, _device_id);
    llaisys::ops::linear(logits_2d, out_norm, unwrap(_weights.out_embed), _zero_bias_voc);

    return logits_2d->view({_meta.voc});
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken, float temperature, int top_k, float top_p) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2Model: token_ids is null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2Model: ntoken must be > 0");
    CHECK_ARGUMENT(ntoken <= _meta.maxseq, "Qwen2Model: ntoken exceeds max sequence length");

    if (ntoken <= _past_len) {
        reset_state();
    }

    llaisys::tensor_t logits;
    for (size_t i = _past_len; i < ntoken; i++) {
        logits = forward_token(token_ids[i], i);
    }

    _past_len = ntoken;

    auto sampled = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    llaisys::ops::sample(sampled, logits, temperature, top_k, top_p);

    return *reinterpret_cast<int64_t *>(sampled->data());
}

} // namespace llaisys::models
