from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from .. import Tensor
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from pathlib import Path
import random

import safetensors

try:
    import torch
except Exception:
    torch = None


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._backend_model = None
        self._backend_kv = None
        self._device = DeviceType.CPU if device == DeviceType.CPU else DeviceType.NVIDIA
        self._device_id = 0
        self._weight_tensors = []
        self._maxseq = 2048
        self._end_token = -1
        self._dtype = DataType.F32

        # attempt to create backend model
        try:
            # try to read model config if present to populate meta
            cfg = {}
            cfg_path = model_path / "config.json"
            if cfg_path.exists():
                import json

                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

            meta = LlaisysQwen2Meta()
            dtype_str = str(cfg.get("torch_dtype", "bfloat16")).lower()
            if "bfloat16" in dtype_str:
                self._dtype = DataType.BF16
            elif "float16" in dtype_str or "half" in dtype_str:
                self._dtype = DataType.F16
            else:
                self._dtype = DataType.F32
            meta.dtype = self._dtype
            # robustly extract fields from config
            meta.nlayer = int(cfg.get("num_hidden_layers", cfg.get("n_layer", cfg.get("num_layers", 0))))
            meta.hs = int(cfg.get("hidden_size", cfg.get("d_model", 0)))
            meta.nh = int(cfg.get("num_attention_heads", cfg.get("n_head", 0)))
            meta.nkvh = int(cfg.get("num_key_value_heads", meta.nh if meta.nh else 0))
            if meta.nkvh == 0:
                meta.nkvh = meta.nh
            meta.dh = int(cfg.get("head_dim", int(meta.hs / meta.nh) if meta.nh else 0))
            meta.di = int(cfg.get("intermediate_size", cfg.get("ffn_dim", meta.hs * 4 if meta.hs else 0)))
            meta.maxseq = int(cfg.get("max_position_embeddings", cfg.get("max_seq_len", 2048)))
            meta.voc = int(cfg.get("vocab_size", cfg.get("vocab_size", 0)))
            meta.epsilon = float(cfg.get("layer_norm_eps", cfg.get("eps", 1e-5)))
            meta.theta = float(cfg.get("rope_theta", cfg.get("theta", 10000.0)))
            meta.end_token = int(cfg.get("eos_token_id", cfg.get("end_token", -1)))
            self._maxseq = int(meta.maxseq)
            self._end_token = int(meta.end_token)

            import ctypes

            self._backend_model = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(meta), self._device, None, 0)

            if torch is None:
                raise RuntimeError("PyTorch is required to load safetensors weights")

            if self._dtype == DataType.BF16:
                target_torch_dtype = torch.bfloat16
            elif self._dtype == DataType.F16:
                target_torch_dtype = torch.float16
            else:
                target_torch_dtype = torch.float32

            for file in sorted(model_path.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                for name_ in data_.keys():
                    ten = data_.get_tensor(name_).detach().cpu().contiguous()
                    if ten.dtype != target_torch_dtype:
                        ten = ten.to(target_torch_dtype)

                    t = Tensor(shape=ten.shape, dtype=self._dtype, device=self._device, device_id=self._device_id)
                    t.load(ten.data_ptr())
                    LIB_LLAISYS.llaisysQwen2ModelSetWeight(self._backend_model, name_.encode("utf-8"), t.lib_tensor())
                    self._weight_tensors.append(t)

            LIB_LLAISYS.llaisysQwen2ModelFinalize(self._backend_model)
            self._backend_kv = LIB_LLAISYS.llaisysQwen2KVCreat(self._backend_model, self._maxseq)

            # verify some required weights exist (best-effort)
            required_groups = [
                [b"model.norm.weight", b"norm.weight"],
                [b"model.embed_tokens.weight", b"embed_tokens.weight"],
                [b"lm_head.weight"],
            ]
            missing = []
            for group in required_groups:
                found = False
                for r in group:
                    try:
                        has = LIB_LLAISYS.llaisysQwen2ModelHasWeight(self._backend_model, r)
                    except Exception:
                        has = 0
                    if has:
                        found = True
                        break
                if not found:
                    missing.append(group[0].decode("utf-8"))
            if missing:
                print("[llaisys qwen2] Warning: missing weights:", missing)
        except Exception as e:
            raise RuntimeError(f"[llaisys qwen2] backend load failed: {e}") from e

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = None,
    ):
        if self._backend_model is None:
            raise RuntimeError("Qwen2 backend is not initialized")

        import ctypes
        from ctypes import c_float, c_int64, c_size_t, c_uint64

        input_ids = [int(t) for t in inputs]
        if not input_ids:
            return []

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return input_ids

        kv_cap = len(input_ids) + max_new_tokens
        if self._maxseq > 0:
            kv_cap = min(kv_cap, self._maxseq)
        kv_cap = max(kv_cap, 1)

        if self._backend_kv is not None:
            LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
            self._backend_kv = None
        self._backend_kv = LIB_LLAISYS.llaisysQwen2KVCreat(self._backend_model, c_size_t(kv_cap))

        sample_api_available = hasattr(LIB_LLAISYS, "llaisysQwen2ModelInferSample")
        sample_top_k = max(0, int(top_k))
        sample_top_p = float(top_p)
        sample_temperature = float(temperature)
        rng = random.Random() if seed is None else random.Random(int(seed))

        def infer_with_sampling(token_buffer, n_token):
            if sample_api_available:
                step_seed = rng.getrandbits(64)
                return int(
                    LIB_LLAISYS.llaisysQwen2ModelInferSample(
                        self._backend_model,
                        token_buffer,
                        c_size_t(n_token),
                        c_float(sample_temperature),
                        c_size_t(sample_top_k),
                        c_float(sample_top_p),
                        c_uint64(step_seed),
                    )
                )
            return int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._backend_model,
                    token_buffer,
                    c_size_t(n_token),
                )
            )

        arr = (c_int64 * len(input_ids))(*input_ids)
        next_token = infer_with_sampling(arr, len(input_ids))

        output_ids = list(input_ids)
        for _ in range(max_new_tokens):
            if next_token is None:
                break
            output_ids.append(next_token)
            if self._end_token >= 0 and next_token == self._end_token:
                break
            arr = (c_int64 * 1)(next_token)
            next_token = infer_with_sampling(arr, 1)

        return output_ids

    def __del__(self):
        try:
            if self._backend_kv is not None:
                LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
                self._backend_kv = None
        except Exception:
            pass
        try:
            if self._backend_model is not None:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._backend_model)
        except Exception:
            pass
