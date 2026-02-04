from typing import Sequence, Optional
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import json
import safetensors

from ..libllaisys.llaisys_types import DataType
from ..libllaisys.tensor import llaisysTensor_t
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
import ctypes

from ..tensor import Tensor

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # - parse model config -> meta
        # - create backend model
        # - load safetensors -> create llaisys tensors -> fill weights struct
        self._device = device
        self._model = None
        self._weights = None
        self._meta = None
        self._model_path = None
        # Keep Python Tensor objects alive; the backend only stores raw handles.
        self._owned_tensors = []

        model_path = Path(model_path)
        self._model_path = model_path

        meta = self._load_meta(model_path)
        self._meta = meta
        self._model, self._weights_ptr, self._weights = self._create_backend(meta, device)

        for file in sorted(model_path.glob("*.safetensors")):
            # NOTE: weights are often bf16, NumPy may not support it well.
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                t = data_.get_tensor(name_)
                self._load_weight(name_, t)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        _ = (top_k, top_p, temperature)
        if max_new_tokens is None:
            max_new_tokens = 128

        # loop:
        #   next_id = LIB_LLAISYS.llaisysQwen2ModelInfer(...)
        #   append -> stop on eos
        token_ids = list(inputs)
        eos_id = int(self._meta.end_token) if self._meta is not None else -1
        for _ in range(max_new_tokens):
            arr = (ctypes.c_int64 * len(token_ids))(*token_ids)
            next_id = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(token_ids))
            )
            token_ids.append(next_id)
            if eos_id != -1 and next_id == eos_id:
                break
            print(next_id)
        return token_ids

    def _create_backend(self, meta: LlaisysQwen2Meta, device: DeviceType):
        device_ids = (ctypes.c_int * 1)(0)
        m = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(meta), int(device), device_ids, 1)
        if not m:
            raise RuntimeError("llaisysQwen2ModelCreate failed (returned null)")
        w_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(m)
        return m, w_ptr, w_ptr.contents

    def _load_meta(self, model_path: Path) -> LlaisysQwen2Meta:
        cfg_path = model_path / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json under: {model_path}")

        cfg = json.loads(cfg_path.read_text())

        hs = int(cfg["hidden_size"])
        nh = int(cfg["num_attention_heads"])
        nkvh = int(cfg.get("num_key_value_heads", nh))
        nlayer = int(cfg["num_hidden_layers"])
        di = int(cfg["intermediate_size"])
        voc = int(cfg["vocab_size"])
        maxseq = int(cfg.get("max_position_embeddings", cfg.get("seq_length", 0)))
        if maxseq <= 0:
            maxseq = 4096

        # Model-specific constants.
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        end_token = int(cfg.get("eos_token_id", -1))
        # DType: keep bf16 by default (matching test/test_infer.py).
        dtype = DataType.BF16

        dh = hs // nh
        return LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=eps,
            theta=theta,
            end_token=end_token,
        )

    def _load_weight(self, name: str, t):
        # t is a torch.Tensor when loaded via safetensors (framework="pt").
        # We only use torch here as a container for the raw bytes.
        if not hasattr(t, "dtype"):
            raise TypeError(f"Unexpected tensor type for {name}: {type(t)}")

        # Map torch dtype -> llaisys dtype
        if str(t.dtype) == "torch.bfloat16":
            dtype = DataType.BF16
        elif str(t.dtype) == "torch.float16":
            dtype = DataType.F16
        elif str(t.dtype) == "torch.float32":
            dtype = DataType.F32
        else:
            raise ValueError(f"Unsupported dtype for {name}: {t.dtype}")

        # Always load from contiguous CPU tensor for now.
        t_contig = t.contiguous().cpu()
        shape = tuple(int(s) for s in t_contig.shape)
        w = Tensor(shape=shape, dtype=dtype, device=self._device)
        w.load(ctypes.c_void_p(int(t_contig.data_ptr())))

        # IMPORTANT: keep it alive to avoid freeing the underlying llaisysTensor_t.
        self._owned_tensors.append(w)
        self._route_weight(name, w.lib_tensor())

    def _route_weight(self, name: str, handle: llaisysTensor_t):
        # Minimal mapping skeleton (Assignment #3).
        if name == "model.embed_tokens.weight":
            self._weights.in_embed = handle
            return
        if name == "lm_head.weight":
            self._weights.out_embed = handle
            return
        if name == "model.norm.weight":
            self._weights.out_norm_w = handle
            return

        # Per-layer mappings
        if name.startswith("model.layers."):
            parts = name.split(".")
            # Expect: model.layers.{i}.<...>
            try:
                layer = int(parts[2])
            except Exception:
                print(f"TODO unmapped (bad layer parse): {name}")
                return

            if name.endswith("input_layernorm.weight"):
                self._weights.attn_norm_w[layer] = handle
                return
            if name.endswith("post_attention_layernorm.weight"):
                self._weights.mlp_norm_w[layer] = handle
                return

            # Attention projections
            if "self_attn.q_proj.weight" in name:
                self._weights.attn_q_w[layer] = handle
                return
            if "self_attn.q_proj.bias" in name:
                self._weights.attn_q_b[layer] = handle
                return
            if "self_attn.k_proj.weight" in name:
                self._weights.attn_k_w[layer] = handle
                return
            if "self_attn.k_proj.bias" in name:
                self._weights.attn_k_b[layer] = handle
                return
            if "self_attn.v_proj.weight" in name:
                self._weights.attn_v_w[layer] = handle
                return
            if "self_attn.v_proj.bias" in name:
                self._weights.attn_v_b[layer] = handle
                return
            if "self_attn.o_proj.weight" in name:
                self._weights.attn_o_w[layer] = handle
                return

            # MLP projections
            if "mlp.gate_proj.weight" in name:
                self._weights.mlp_gate_w[layer] = handle
                return
            if "mlp.up_proj.weight" in name:
                self._weights.mlp_up_w[layer] = handle
                return
            if "mlp.down_proj.weight" in name:
                self._weights.mlp_down_w[layer] = handle
                return

        print(f"TODO unmapped: {name}")
