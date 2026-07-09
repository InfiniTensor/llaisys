from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import Qwen2Meta, Qwen2Weights
from pathlib import Path
import safetensors
import json
from ctypes import c_int64, c_int

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        model_path = Path(model_path)
        
        with open(model_path / "config.json") as f:
            config = json.load(f)
        
        meta = Qwen2Meta()
        meta.dtype = DataType.F32
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = config["hidden_size"] // config["num_attention_heads"]
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 32768)
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(meta, device, None, 0)
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        self.weights = self.weights_ptr.contents
        self.nlayer = meta.nlayer
        self.end_token = meta.end_token
        
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                tensor_data = data_.get_tensor(name_)
                if tensor_data.dtype.is_floating_point and tensor_data.dtype != tensor_data.float().dtype:
                    tensor_data = tensor_data.float()
                tensor_data = tensor_data.numpy()
                self._load_weight(name_, tensor_data)

    def _load_weight(self, name, data):
        if name == "model.embed_tokens.weight":
            LIB_LLAISYS.tensorLoad(self.weights.in_embed, data.ctypes.data)
        elif name == "lm_head.weight":
            LIB_LLAISYS.tensorLoad(self.weights.out_embed, data.ctypes.data)
        elif name == "model.norm.weight":
            LIB_LLAISYS.tensorLoad(self.weights.out_norm_w, data.ctypes.data)
        elif "layers" in name:
            parts = name.split(".")
            layer_idx = int(parts[2])
            if "input_layernorm.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_norm_w[layer_idx], data.ctypes.data)
            elif "self_attn.q_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_q_w[layer_idx], data.ctypes.data)
            elif "self_attn.q_proj.bias" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_q_b[layer_idx], data.ctypes.data)
            elif "self_attn.k_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_k_w[layer_idx], data.ctypes.data)
            elif "self_attn.k_proj.bias" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_k_b[layer_idx], data.ctypes.data)
            elif "self_attn.v_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_v_w[layer_idx], data.ctypes.data)
            elif "self_attn.v_proj.bias" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_v_b[layer_idx], data.ctypes.data)
            elif "self_attn.o_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.attn_o_w[layer_idx], data.ctypes.data)
            elif "post_attention_layernorm.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.mlp_norm_w[layer_idx], data.ctypes.data)
            elif "mlp.gate_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.mlp_gate_w[layer_idx], data.ctypes.data)
            elif "mlp.up_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.mlp_up_w[layer_idx], data.ctypes.data)
            elif "mlp.down_proj.weight" in name:
                LIB_LLAISYS.tensorLoad(self.weights.mlp_down_w[layer_idx], data.ctypes.data)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # TODO: Implement generate function
        tokens = list(inputs)
        for _ in range(max_new_tokens or 128):
            token_array = (c_int64 * len(tokens))(*tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, token_array, len(tokens))
            tokens.append(next_token)
            if next_token == self.end_token:
                break
        return tokens
    
    def __del__(self):
        if hasattr(self, 'model'):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
