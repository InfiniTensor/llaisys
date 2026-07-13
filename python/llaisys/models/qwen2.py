import ctypes
import json
from typing import Iterator, Sequence
from ..libllaisys import LIB_LLAISYS, LlaisysQwen2Meta
from ..libllaisys import DeviceType, DataType
from ..tensor import Tensor

from pathlib import Path
import safetensors

import torch
import numpy as np
from tqdm import tqdm


DEFAULT_MODEL_PATH = "./data"


class Qwen2:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device: DeviceType = DeviceType.CPU,
    ):
        self.device = device
        self._backend = None

        model_path = Path(model_path)
        self.__load_config(model_path / "config.json")
        self.__load_weights(model_path)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        answer = list(inputs)
        for token in self.generate_stream(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ):
            answer.append(token)

        return answer

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Iterator[int]:
        if len(inputs) == 0 or max_new_tokens <= 0:
            return

        array = (ctypes.c_int64 * len(inputs))(*inputs)
        pos_ids = (ctypes.c_int64 * len(inputs))(*range(len(inputs)))

        # Prefill and yield first token.
        output_token = LIB_LLAISYS.llaisysQwen2ModelInferSample(
            self._backend,
            array,
            pos_ids,
            len(inputs),
            True,
            int(top_k),
            float(top_p),
            float(temperature),
        )
        yield output_token

        # Decode loop: one token per step.
        current_token = output_token
        current_pos = len(inputs)
        for _ in range(max_new_tokens - 1):
            if current_token == self.meta.end_token:
                break

            array = (ctypes.c_int64 * 1)(current_token)
            pos_ids = (ctypes.c_int64 * 1)(current_pos)
            output_token = LIB_LLAISYS.llaisysQwen2ModelInferSample(
                self._backend,
                array,
                pos_ids,
                1,
                False,
                int(top_k),
                float(top_p),
                float(temperature),
            )
            yield output_token
            current_token = output_token
            current_pos += 1

    def __load_config(self, config_path: Path):
        with open(config_path, "r") as f:
            config = json.load(f)

        meta = LlaisysQwen2Meta()
        match config.get("torch_dtype", ""):
            case "bfloat16":
                meta.dtype = DataType.BF16
            case "float16":
                meta.dtype = DataType.F16
            case "float32":
                meta.dtype = DataType.F32
            case _:
                raise ValueError(
                    f"Unsupported data type: {config.get('torch_dtype', '')}"
                )
        meta.dtype = DataType.F32  # always use fp32 for now
        meta.nlayer = config.get("num_hidden_layers", 0)
        meta.nh = config.get("num_attention_heads", 0)
        meta.hs = config.get("hidden_size", 0)
        meta.nkvh = config.get("num_key_value_heads", 0)
        meta.dh = config.get("head_dim", int(meta.hs / meta.nh) if meta.nh else 0)
        meta.di = config.get("intermediate_size", 0)
        meta.maxseq = config.get("max_position_embeddings", 0)
        meta.voc = config.get("vocab_size", 0)
        meta.epsilon = config.get("layer_norm_epsilon", 1e-5)
        meta.theta = config.get("rope_theta", 1000000.0)
        meta.end_token = config.get("eos_token_id", 0)

        self.meta = meta

        # Init model
        self._backend = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            self.device,
            None,
            0,
        )

    def __get_name_mapping(self, weights_folder: Path):
        self.name_mapping: dict[str, tuple[str, int, int]] = {
            # (short name, indicator of weight type, optional layer index)
            # see: llaisys/qwen2.cc:setWeights()
            "model.embed_tokens.weight": ("in_embed", 0, -1),
            "lm_head.weight": ("out_embed", 1, -1),
            "model.norm.weight": ("out_norm_w", 2, -1),
        }

        for layer_idx in range(self.meta.nlayer):
            prefix = f"model.layers.{layer_idx}"
            self.name_mapping.update(
                {
                    f"{prefix}.input_layernorm.weight": ("attn_norm_w", 3, layer_idx),
                    f"{prefix}.self_attn.q_proj.weight": ("attn_q_w", 4, layer_idx),
                    f"{prefix}.self_attn.q_proj.bias": ("attn_q_b", 5, layer_idx),
                    f"{prefix}.self_attn.k_proj.weight": ("attn_k_w", 6, layer_idx),
                    f"{prefix}.self_attn.k_proj.bias": ("attn_k_b", 7, layer_idx),
                    f"{prefix}.self_attn.v_proj.weight": ("attn_v_w", 8, layer_idx),
                    f"{prefix}.self_attn.v_proj.bias": ("attn_v_b", 9, layer_idx),
                    f"{prefix}.self_attn.o_proj.weight": ("attn_o_w", 10, layer_idx),
                    f"{prefix}.post_attention_layernorm.weight": (
                        "mlp_norm_w",
                        11,
                        layer_idx,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": ("mlp_gate_w", 12, layer_idx),
                    f"{prefix}.mlp.up_proj.weight": ("mlp_up_w", 13, layer_idx),
                    f"{prefix}.mlp.down_proj.weight": ("mlp_down_w", 14, layer_idx),
                }
            )

    def __load_weights(self, weights_folder: Path):
        self.__get_name_mapping(weights_folder)

        for file in sorted(weights_folder.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    if name_ not in self.name_mapping:
                        raise ValueError(f"Unknown weight name: {name_}")
                    short_name, weight_type, layer_idx = self.name_mapping[name_]
                    weight_data = data_.get_tensor(name_)  # load as torch

                    # Convert to target dtype before numpy conversion
                    if self.meta.dtype == DataType.BF16:
                        weight_data = (
                            weight_data.to(torch.bfloat16).view(torch.uint16).numpy()
                        )
                    elif self.meta.dtype == DataType.F16:
                        weight_data = (
                            weight_data.to(torch.float16).view(torch.uint16).numpy()
                        )
                    else:  # F32
                        weight_data = weight_data.float().numpy()

                    tensor = Tensor(weight_data.shape, self.meta.dtype, self.device)
                    tensor.load(weight_data.ctypes.data)

                    # Set weights in the backend
                    LIB_LLAISYS.llaisysQwen2SetWeights(
                        self._backend,
                        weight_type,
                        layer_idx,
                        tensor.lib_tensor(),
                    )

    def generate_no_decode(self, inputs: Sequence[int], max_new_tokens: int):
        answer = list(inputs)
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            array = (ctypes.c_int64 * len(answer))(*answer)
            pos_ids = (ctypes.c_int64 * len(answer))(*range(len(answer)))

            # Prefill and get first token
            output_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._backend,
                array,
                pos_ids,
                len(answer),
                True,  # prefill
            )
            answer.append(output_token)

        return answer
