import json
import mmap
import struct
from ctypes import byref, c_int64, c_size_t, c_void_p
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType, llaisysDeviceType_t
from ..tensor import Tensor


class DecoderOnlyModel:
    create_api = ""
    destroy_api = ""
    weights_api = ""
    reset_api = ""
    infer_api = ""
    infer_sample_api = ""
    meta_cls = None
    default_rope_theta = 10000.0

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self._tensor_refs: List[Tensor] = []
        self._config = self._load_config()
        self.end_token = self._normalize_eos(self._config.get("eos_token_id", 2))
        self._model_handle = self._create_backend_model()
        weights_ptr = getattr(LIB_LLAISYS, self.weights_api)(self._model_handle)
        if not weights_ptr:
            raise RuntimeError("后端没有返回 weights 指针。")
        self._weights = weights_ptr.contents
        self._load_weights()

    def _load_config(self):
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            candidates = list(self.model_path.rglob("config.json"))
            if not candidates:
                raise FileNotFoundError("config.json not found under model_path")
            config_path = candidates[0]
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_eos(eos_token):
        if isinstance(eos_token, list):
            return int(eos_token[0])
        return int(eos_token)

    def _build_meta(self):
        config = self._config
        meta = self.meta_cls()
        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        # 当前后端统一按 float32 装载权重并执行，接口层先把不同 safetensors 精度
        # 转成 float32，保证 Qwen2/Llama 两条路径共用同一套推理实现。
        meta.dtype = DataType.F32.value
        meta.nlayer = int(config["num_hidden_layers"])
        meta.hs = hidden_size
        meta.nh = num_heads
        meta.nkvh = int(config.get("num_key_value_heads", num_heads))
        meta.dh = hidden_size // num_heads
        meta.di = int(config["intermediate_size"])
        meta.maxseq = int(config.get("max_position_embeddings", 2048))
        meta.voc = int(config["vocab_size"])
        meta.epsilon = float(config["rms_norm_eps"])
        meta.theta = float(config.get("rope_theta", self.default_rope_theta))
        meta.end_token = self.end_token
        return meta

    def _create_backend_model(self):
        meta = self._build_meta()
        handle = getattr(LIB_LLAISYS, self.create_api)(
            byref(meta),
            llaisysDeviceType_t(self.device.value),
            None,
            0,
        )
        if not handle:
            raise RuntimeError("后端模型创建失败。")
        return handle

    def reset(self):
        getattr(LIB_LLAISYS, self.reset_api)(self._model_handle)

    def _load_weights(self):
        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            files = sorted(self.model_path.rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError("No .safetensors files found under model_path")

        for file_path in files:
            with open(file_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size).decode("utf-8"))
                data_start = 8 + header_size

                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        # safetensors 只负责持久化，真正送入后端前统一转换成 float32 tensor。
                        array = self._to_float32(mm[data_start + info["data_offsets"][0]: data_start + info["data_offsets"][1]], info["dtype"])
                        if array is None:
                            continue
                        self._assign_weight(name, array, info["shape"], DataType.F32)

    @staticmethod
    def _to_float32(raw_bytes: bytes, dtype_str: str):
        if dtype_str == "BF16":
            raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            return (raw_u16.astype(np.uint32) << 16).view(np.float32)
        if dtype_str == "F16":
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        if dtype_str == "F32":
            return np.frombuffer(raw_bytes, dtype=np.float32)
        return None

    def _new_tensor_and_copy(self, shape: List[int], dtype: DataType, data_f32: np.ndarray):
        tensor = Tensor(shape=shape, dtype=dtype, device=self.device)
        tensor.load(data_f32.ctypes.data_as(c_void_p))
        self._tensor_refs.append(tensor)
        return tensor.lib_tensor()

    def _assign_weight(self, name: str, data: np.ndarray, shape: List[int], dtype: DataType):
        handle = self._new_tensor_and_copy(shape, dtype, data)
        weights = self._weights
        # 这里把 Hugging Face 的权重命名映射到后端固定的权重槽位，
        # 这样 Python 上层就可以通过同一套命名规则同时支持 Qwen2 和 Llama。

        if name == "model.embed_tokens.weight":
            weights.in_embed = handle
            return
        if name == "lm_head.weight":
            weights.out_embed = handle
            return
        if name == "model.norm.weight":
            weights.out_norm_w = handle
            return

        if not name.startswith("model.layers."):
            return

        parts = name.split(".")
        try:
            idx = int(parts[2])
        except Exception:
            return
        if idx < 0 or idx >= int(self._config["num_hidden_layers"]):
            return

        module = parts[3]
        if module == "input_layernorm" and parts[-1] == "weight":
            weights.attn_norm_w[idx] = handle
            return
        if module == "post_attention_layernorm" and parts[-1] == "weight":
            weights.mlp_norm_w[idx] = handle
            return
        if module == "self_attn":
            sub = parts[4]
            last = parts[-1]
            if sub == "q_proj" and last == "weight":
                weights.attn_q_w[idx] = handle
                return
            if sub == "k_proj" and last == "weight":
                weights.attn_k_w[idx] = handle
                return
            if sub == "v_proj" and last == "weight":
                weights.attn_v_w[idx] = handle
                return
            if sub == "o_proj" and last == "weight":
                weights.attn_o_w[idx] = handle
                return
            if sub == "q_proj" and last == "bias":
                weights.attn_q_b[idx] = handle
                return
            if sub == "k_proj" and last == "bias":
                weights.attn_k_b[idx] = handle
                return
            if sub == "v_proj" and last == "bias":
                weights.attn_v_b[idx] = handle
                return
            return
        if module == "mlp" and parts[-1] == "weight":
            sub = parts[4]
            if sub == "gate_proj":
                weights.mlp_gate_w[idx] = handle
                return
            if sub == "up_proj":
                weights.mlp_up_w[idx] = handle
                return
            if sub == "down_proj":
                weights.mlp_down_w[idx] = handle

    def _infer_once(self, token_ids: Sequence[int], temperature: float, top_k: int, top_p: float, seed: int):
        arr = (c_int64 * len(token_ids))(*[int(token) for token in token_ids])
        greedy = top_k == 1 and float(top_p) == 1.0 and float(temperature) == 1.0
        if greedy:
            # 贪心模式直接走 infer_api，避免额外采样开销，也方便和 HF 做 token 级对齐。
            return int(getattr(LIB_LLAISYS, self.infer_api)(self._model_handle, arr, c_size_t(len(token_ids))))
        return int(
            getattr(LIB_LLAISYS, self.infer_sample_api)(
                self._model_handle,
                arr,
                c_size_t(len(token_ids)),
                float(temperature),
                int(top_k),
                float(top_p),
                int(seed),
            )
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_p: float = 1.0,
        top_k: int = 1,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> List[int]:
        tokens = [int(token) for token in inputs]
        if not tokens:
            raise ValueError("inputs must be non-empty")

        self.reset()
        # 第一步要把完整 prompt 一次性送入模型，后面才能只喂最后一个 token 做增量解码。
        next_token = self._infer_once(tokens, temperature, top_k, top_p, seed)
        tokens.append(next_token)

        for step in range(max_new_tokens - 1):
            if tokens[-1] == self.end_token:
                break
            next_token = self._infer_once([tokens[-1]], temperature, top_k, top_p, seed + step + 1)
            tokens.append(next_token)

        return tokens

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_p: float = 1.0,
        top_k: int = 1,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        tokens = [int(token) for token in inputs]
        if not tokens:
            raise ValueError("inputs must be non-empty")

        self.reset()

        # 首轮把完整 prompt 一次性送入模型，之后每步只增量送最后一个 token。
        next_token = self._infer_once(tokens, temperature, top_k, top_p, seed)
        tokens.append(next_token)
        yield next_token, list(tokens)

        for step in range(max_new_tokens - 1):
            if tokens[-1] == self.end_token:
                break
            next_token = self._infer_once(
                [tokens[-1]],
                temperature,
                top_k,
                top_p,
                seed + step + 1,
            )
            tokens.append(next_token)
            yield next_token, list(tokens)

    def __del__(self):
        try:
            if getattr(self, "_model_handle", None):
                getattr(LIB_LLAISYS, self.destroy_api)(self._model_handle)
                self._model_handle = None
        except Exception:
            pass
