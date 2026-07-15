import ctypes
import json
import mmap
import os
import struct
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from ..libllaisys import LIB_LLAISYS, llaisysTensor_t, llaisysDataType_t, llaisysDeviceType_t, DataType, DeviceType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model, llaisysQwen2ModelHandle
from ..tensor import Tensor

LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = llaisysQwen2ModelHandle

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2ModelHandle]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2ModelHandle]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelReset.argtypes = [llaisysQwen2ModelHandle]
LIB_LLAISYS.llaisysQwen2ModelReset.restype = None

LIB_LLAISYS.llaisysQwen2ModelTruncate.argtypes = [llaisysQwen2ModelHandle, ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelTruncate.restype = None

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2ModelHandle, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64

LIB_LLAISYS.llaisysQwen2ModelGenerateNext.argtypes = [
    llaisysQwen2ModelHandle,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
]
LIB_LLAISYS.llaisysQwen2ModelGenerateNext.restype = ctypes.c_int64


class Qwen2:
    SUPPORTED_MODEL_TYPES: ClassVar[set[str]] = {"qwen2"}

    @classmethod
    def supports_model_type(cls, model_type: str) -> bool:
        return str(model_type).lower() in cls.SUPPORTED_MODEL_TYPES

    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config
        self.model_type = str(config.get("model_type", "")).lower()
        
        self.device = device
        self.device_id = device_id
        
        self.meta = LlaisysQwen2Meta()
        self._configure_meta(config)
        
        dev_ids = (ctypes.c_int * 1)(device_id)
        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device, dev_ids, 1)
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle)
        self.tensors_ref = []

        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading weights from {file}...")
            weights_data = self._load_safetensors(file)
            for key, (arr, shape, source_dtype) in weights_data.items():
                raw = self._convert_array_for_model_dtype(arr, source_dtype)
                if not raw.flags["C_CONTIGUOUS"]:
                    raw = np.ascontiguousarray(raw)

                t = Tensor(list(shape), self.meta.dtype, device, device_id)
                t.load(ctypes.c_void_p(raw.ctypes.data))
                self._assign_weight(key, t)

        self._finalize_weights()

    def _configure_meta(self, config: Dict[str, object]) -> None:
        dtype_str = str(config.get("torch_dtype", "float32"))
        self.meta.dtype = self._runtime_dtype(dtype_str, self.device)
        self.meta.nlayer = int(config.get("num_hidden_layers", 24))
        self.meta.hs = int(config.get("hidden_size", 2048))
        self.meta.nh = int(config.get("num_attention_heads", 16))
        self.meta.nkvh = int(config.get("num_key_value_heads", self.meta.nh))
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = int(config.get("intermediate_size", 11008))
        self.meta.maxseq = int(config.get("max_position_embeddings", 8192))
        self.meta.voc = int(config.get("vocab_size", 151936))
        self.meta.epsilon = float(config.get("rms_norm_eps", 1e-6))
        self.meta.theta = float(config.get("rope_theta", 1000000.0))
        eos_token = config.get("eos_token_id", 151643)
        if isinstance(eos_token, list):
            eos_token = eos_token[0] if eos_token else 151643
        self.meta.end_token = int(eos_token)

    @staticmethod
    def _llaisys_dtype_from_string(dtype_str: str) -> DataType:
        normalized = str(dtype_str).lower()
        if normalized in {"bfloat16", "bf16"}:
            return DataType.BF16
        if normalized in {"float16", "f16", "half"}:
            return DataType.F16
        return DataType.F32

    @staticmethod
    def _runtime_dtype(dtype_str: str, device: DeviceType) -> DataType:
        # CPU inference is currently optimized only for F32 kernels. Keep that fast path
        # as the default, and allow native 16-bit weights only as an explicit opt-in.
        if device == DeviceType.CPU and os.getenv("LLAISYS_CPU_NATIVE_DTYPE", "").lower() not in {"1", "true", "yes", "on"}:
            return DataType.F32
        return Qwen2._llaisys_dtype_from_string(dtype_str)

    def _load_safetensors(self, path: Path) -> Dict[str, Tuple[np.ndarray, Sequence[int], str]]:
        tensors = {}
        with open(path, "rb") as f:
            length_bytes = f.read(8)
            if not length_bytes:
                return {}
            header_size = struct.unpack("<Q", length_bytes)[0]

            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)

            fileno = f.fileno()
            total_size = os.fstat(fileno).st_size
            mm = mmap.mmap(fileno, total_size, access=mmap.ACCESS_READ)

            data_start = 8 + header_size

            for key, info in header.items():
                if key == "__metadata__":
                    continue

                dtype_str = str(info["dtype"])
                shape = info["shape"]
                start, end = info["data_offsets"]

                abs_start = data_start + start

                if dtype_str in {"BF16", "bfloat16"}:
                    arr = np.array(
                        np.frombuffer(mm, dtype=np.uint16, count=(end - start) // 2, offset=abs_start),
                        copy=True,
                    ).reshape(shape)
                    tensors[key] = (arr, shape, "bf16")
                elif dtype_str in {"F32", "float32"}:
                    arr = np.array(
                        np.frombuffer(mm, dtype=np.float32, count=(end - start) // 4, offset=abs_start),
                        copy=True,
                    ).reshape(shape)
                    tensors[key] = (arr, shape, "f32")
                elif dtype_str in {"F16", "float16"}:
                    arr = np.array(
                        np.frombuffer(mm, dtype=np.float16, count=(end - start) // 2, offset=abs_start),
                        copy=True,
                    ).reshape(shape)
                    tensors[key] = (arr, shape, "f16")

            mm.close()

        return tensors

    def _convert_array_for_model_dtype(self, arr: np.ndarray, source_dtype: str) -> np.ndarray:
        target_dtype = self.meta.dtype
        if target_dtype == DataType.F32:
            return self._to_float32_array(arr, source_dtype)
        if target_dtype == DataType.F16:
            return self._to_float16_array(arr, source_dtype)
        if target_dtype == DataType.BF16:
            return self._to_bfloat16_bytes(arr, source_dtype)
        raise ValueError(f"Unsupported model dtype: {target_dtype}")

    @staticmethod
    def _to_float32_array(arr: np.ndarray, source_dtype: str) -> np.ndarray:
        if source_dtype == "bf16":
            u32 = arr.astype(np.uint32) << 16
            return u32.view(np.float32)
        if source_dtype == "f16":
            return arr.astype(np.float32)
        return np.asarray(arr, dtype=np.float32)

    @staticmethod
    def _to_float16_array(arr: np.ndarray, source_dtype: str) -> np.ndarray:
        if source_dtype == "f16":
            return np.asarray(arr, dtype=np.float16)
        return Qwen2._to_float32_array(arr, source_dtype).astype(np.float16)

    @staticmethod
    def _to_bfloat16_bytes(arr: np.ndarray, source_dtype: str) -> np.ndarray:
        if source_dtype == "bf16":
            return np.asarray(arr, dtype=np.uint16)
        float32_arr = Qwen2._to_float32_array(arr, source_dtype)
        u32 = float32_arr.view(np.uint32)
        return (u32 >> 16).astype(np.uint16)

    def _finalize_weights(self) -> None:
        weights = self.weights_ptr.contents
        if weights.out_embed:
            return
        if self.config.get("tie_word_embeddings") and weights.in_embed:
            weights.out_embed = weights.in_embed
            return
        raise ValueError("Missing output embedding weights: expected `lm_head.weight` or tied word embeddings")

    def _assign_weight(self, name: str, t: Tensor):
        w = self.weights_ptr.contents
        # Keep Python Tensor wrappers alive because the model stores raw C tensor handles.
        self.tensors_ref.append(t)
        if name == "model.embed_tokens.weight":
             w.in_embed = t.lib_tensor()
        elif name == "lm_head.weight":
             w.out_embed = t.lib_tensor()
        elif name == "model.norm.weight":
             w.out_norm_w = t.lib_tensor()
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            suffix = ".".join(parts[3:])
            
            def set_w(target_ptr):
                target_ptr[layer_idx] = t.lib_tensor()

            if suffix == "input_layernorm.weight":
                set_w(w.attn_norm_w)
            elif suffix == "self_attn.q_proj.weight":
                set_w(w.attn_q_w)
            elif suffix == "self_attn.q_proj.bias":
                set_w(w.attn_q_b)
            elif suffix == "self_attn.k_proj.weight":
                set_w(w.attn_k_w)
            elif suffix == "self_attn.k_proj.bias":
                set_w(w.attn_k_b)
            elif suffix == "self_attn.v_proj.weight":
                set_w(w.attn_v_w)
            elif suffix == "self_attn.v_proj.bias":
                set_w(w.attn_v_b)
            elif suffix == "self_attn.o_proj.weight":
                set_w(w.attn_o_w)
            elif suffix == "post_attention_layernorm.weight":
                set_w(w.mlp_norm_w)
            elif suffix == "mlp.gate_proj.weight":
                set_w(w.mlp_gate_w)
            elif suffix == "mlp.up_proj.weight":
                set_w(w.mlp_up_w)
            elif suffix == "mlp.down_proj.weight":
                set_w(w.mlp_down_w)

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def reset(self) -> None:
        LIB_LLAISYS.llaisysQwen2ModelReset(self.handle)

    def truncate(self, position: int) -> None:
        if position < 0:
            raise ValueError("position must be non-negative")
        LIB_LLAISYS.llaisysQwen2ModelTruncate(self.handle, int(position))

    def _generate_next(
        self,
        inputs: Sequence[int],
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> int:
        if not inputs:
            raise ValueError("inputs must not be empty")

        arr = (ctypes.c_int64 * len(inputs))(*inputs)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelGenerateNext(
                self.handle,
                arr,
                len(inputs),
                int(top_k),
                float(top_p),
                float(temperature),
            )
        )

    def generate_next(
        self,
        inputs: Sequence[int],
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        *,
        reset_state: bool = False,
    ) -> int:
        if reset_state:
            self.reset()
        return self._generate_next(inputs, top_k=top_k, top_p=top_p, temperature=temperature)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 20,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        *,
        reset_state: bool = True,
    ) -> List[int]:
        if not inputs:
            raise ValueError("inputs must not be empty")

        if reset_state:
            self.reset()
        generated = []
        tokens = list(inputs)

        next_token = self.generate_next(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generated.append(next_token)
        tokens = [next_token]

        for _ in range(max_new_tokens - 1):
            next_token = self.generate_next(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            generated.append(next_token)
            tokens = [next_token]

            if next_token == self.meta.end_token:
                break

        return list(inputs) + generated

    def stream_generate(
        self,
        inputs: Sequence[int],
        *,
        tokenizer=None,
        max_new_tokens: int = 20,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        reset_state: bool = True,
    ) -> Iterator[Tuple[int, str]]:
        if not inputs:
            raise ValueError("inputs must not be empty")

        if reset_state:
            self.reset()
        prompt_tokens = list(inputs)
        generated: List[int] = []
        last_text = ""

        for step in range(max_new_tokens):
            token_source = prompt_tokens if step == 0 else [generated[-1]]
            next_token = self.generate_next(
                token_source,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            generated.append(next_token)

            text_chunk = ""
            if tokenizer is not None:
                decoded = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                text_chunk = decoded[len(last_text):] if decoded.startswith(last_text) else decoded
                last_text = decoded

            yield next_token, text_chunk

            if next_token == self.meta.end_token:
                break
