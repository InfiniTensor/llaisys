from __future__ import annotations

from .decoder_only import DecoderOnlyModel
from ..libllaisys.qwen2 import LlaisysQwen2Meta


class Qwen2(DecoderOnlyModel):
    create_api = "llaisysQwen2ModelCreate"
    destroy_api = "llaisysQwen2ModelDestroy"
    weights_api = "llaisysQwen2ModelWeights"
    reset_api = "llaisysQwen2ModelReset"
    infer_api = "llaisysQwen2ModelInfer"
    infer_sample_api = "llaisysQwen2ModelInferSample"
    meta_cls = LlaisysQwen2Meta
    default_rope_theta = 1000000.0

    def _load_weights(self):
        state_dict = self._load_hf_state_dict()
        weights = self._weights.contents
        num_layers = int(self.meta.nlayer)

        def get_required(name: str):
            if name not in state_dict:
                raise KeyError(f"Missing required weight: {name}")
            return state_dict[name]

        weights.in_embed = self._make_tensor(get_required("model.embed_tokens.weight")).lib_tensor()
        weights.out_norm_w = self._make_tensor(get_required("model.norm.weight")).lib_tensor()
        lm_head = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        weights.out_embed = self._make_tensor(lm_head).lib_tensor()

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}."
            self._set_optional_array_tensor(
                weights.attn_norm_w,
                layer,
                get_required(prefix + "input_layernorm.weight"),
            )
            self._set_optional_array_tensor(
                weights.attn_q_w,
                layer,
                get_required(prefix + "self_attn.q_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.attn_q_b,
                layer,
                state_dict.get(prefix + "self_attn.q_proj.bias"),
            )
            self._set_optional_array_tensor(
                weights.attn_k_w,
                layer,
                get_required(prefix + "self_attn.k_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.attn_k_b,
                layer,
                state_dict.get(prefix + "self_attn.k_proj.bias"),
            )
            self._set_optional_array_tensor(
                weights.attn_v_w,
                layer,
                get_required(prefix + "self_attn.v_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.attn_v_b,
                layer,
                state_dict.get(prefix + "self_attn.v_proj.bias"),
            )
            self._set_optional_array_tensor(
                weights.attn_o_w,
                layer,
                get_required(prefix + "self_attn.o_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.mlp_norm_w,
                layer,
                get_required(prefix + "post_attention_layernorm.weight"),
            )
            self._set_optional_array_tensor(
                weights.mlp_gate_w,
                layer,
                get_required(prefix + "mlp.gate_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.mlp_up_w,
                layer,
                get_required(prefix + "mlp.up_proj.weight"),
            )
            self._set_optional_array_tensor(
                weights.mlp_down_w,
                layer,
                get_required(prefix + "mlp.down_proj.weight"),
            )
