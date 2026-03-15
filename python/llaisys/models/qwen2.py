from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import (
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
)
from ..tensor import Tensor

from ctypes import c_int64, c_size_t, POINTER, byref, cast, c_int, c_void_p, c_float
import json
from pathlib import Path
from safetensors.torch import load_file as safetensors_load_file
import torch


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._device = device
        
        # 加载模型配置
        config_path = model_path / "config.json"   # '/'拼接路径
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 提取模型元数据
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.BF16  # 根据模型配置确定
        self.meta.nlayer = config.get("num_hidden_layers", config.get("num_layers", 0))
        self.meta.hs = config.get("hidden_size", 0)
        self.meta.nh = config.get("num_attention_heads", 0)
        self.meta.nkvh = config.get("num_key_value_heads", self.meta.nh)  # GQA
        self.meta.dh = config.get("head_dim", self.meta.hs // self.meta.nh)   # 一般有hs = nh * dh
        if self.meta.dh == 0:
            self.meta.dh = self.meta.hs // self.meta.nh
        # intermediate_size是MLP(前馈层)的中间层维度，一般是hs的几倍；起到先升维再降维的作用，提高非线性表达能力
        self.meta.di = config.get("intermediate_size", 0)
        self.meta.maxseq = config.get("max_position_embeddings", 32768)
        self.meta.voc = config.get("vocab_size", 0)
        self.meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self.meta.theta = config.get("rope_theta", 1000000.0)      # RoPE的基数，控制位置编码的频率分布
        self.meta.end_token = config.get("eos_token_id", 151643)
        
        # 确定设备
        device_id = 0
        device_ids = (c_int * 1)(device_id)
        
        # 创建模型
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta),             # byref用于将Python对象转换为C语言的结构体指针
            device.value,
            device_ids,
            1
        )
        
        if not self.model:
            raise RuntimeError("Failed to create model")
        
        # 获取权重结构
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not self.weights_ptr:
            raise RuntimeError("Failed to get model weights")
        
        self.weights = self.weights_ptr.contents
        # 持有所有权重 Tensor，延长权重的生命周期，避免 Python GC 导致底层 tensorDestroy 释放权重后悬空
        self._weight_tensors = []
        
        # 加载权重
        self._load_weights(model_path)
    
    # 模型safetensors->LLAISYS:Tensor->C:LlaisysQwen2Weights
    def _load_weights(self, model_path):
        """从 safetensors 文件加载权重（流式加载 + BF16 直拷贝 + 进度输出）"""
        safetensors_files = sorted(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        print(f"[llaisys] Loading Qwen2 weights from: {model_path}")
        print(f"[llaisys] Found {len(safetensors_files)} safetensors")

        # qwen2模型权重为bf16
        def to_bf16_cpu_contig(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().cpu()
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            return t.contiguous()

        def load_llaisys_tensor_from_torch(t: torch.Tensor) -> Tensor:
            t_cpu = to_bf16_cpu_contig(t)
            lt = Tensor(shape=list(t_cpu.shape), dtype=DataType.BF16, device=self._device)
            lt.load(c_void_p(t_cpu.data_ptr()))
            self._weight_tensors.append(lt)
            return lt

        def set_field(name: str, t: torch.Tensor):
            lt = load_llaisys_tensor_from_torch(t)
            setattr(self.weights, name, lt.lib_tensor())   # 为对象动态添加属性，等价于self.weights.name = lt.lib_tensor()

        loaded = 0           # 成功加载，没写进权重结构的tensor数量
        skipped = 0          # 遍历到但没用上的tensor数量

        # 遍历所有safetensors文件
        for file_idx, file in enumerate(safetensors_files):
            print(f"[llaisys] [{file_idx + 1}/{len(safetensors_files)}] reading {file.name}")
            weights_dict = safetensors_load_file(str(file))
            print(f"[llaisys]   tensors in shard: {len(weights_dict)}")

            for key, t in weights_dict.items():
                # Global weights
                if key == "model.embed_tokens.weight":     # 输入 embedding：[voc, hs]
                    set_field("in_embed", t)
                    loaded += 1
                    continue
                if key == "lm_head.weight":
                    set_field("out_embed", t)
                    loaded += 1
                    continue
                if key == "model.norm.weight":
                    set_field("out_norm_w", t)
                    loaded += 1
                    continue

                # Per-layer weights
                if not key.startswith("model.layers."):
                    skipped += 1
                    continue

                parts = key.split(".")
                if len(parts) < 4:
                    skipped += 1
                    continue

                try:
                    layer_idx = int(parts[2])
                except ValueError:
                    skipped += 1
                    continue

                if layer_idx < 0 or layer_idx >= int(self.meta.nlayer):
                    skipped += 1
                    continue

                suffix = ".".join(parts[3:])    # 用'.'拼接层号后的元素

                if suffix == "input_layernorm.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_norm_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "self_attn.q_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_q_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue
                if suffix == "self_attn.q_proj.bias":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_q_b[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "self_attn.k_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_k_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue
                if suffix == "self_attn.k_proj.bias":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_k_b[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "self_attn.v_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_v_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue
                if suffix == "self_attn.v_proj.bias":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_v_b[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "self_attn.o_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.attn_o_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "post_attention_layernorm.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.mlp_norm_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                if suffix == "mlp.gate_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.mlp_gate_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue
                if suffix == "mlp.up_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.mlp_up_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue
                if suffix == "mlp.down_proj.weight":
                    lt = load_llaisys_tensor_from_torch(t)
                    self.weights.mlp_down_w[layer_idx] = lt.lib_tensor()
                    loaded += 1
                    continue

                skipped += 1

            # 释放 shard dict 的引用（尽快回收内存）
            del weights_dict

        print(f"[llaisys] Done. loaded={loaded}, skipped={skipped}")
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 重置 KV Cache（开始新的生成序列）
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self.model)

        output_tokens = list(inputs)
        if len(inputs) == 0:
            return output_tokens

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = max(int(max_new_tokens), 1)

        # Prefill 阶段
        next_token = self._infer_next(inputs, top_k, top_p, temperature)
        output_tokens.append(next_token)

        # Decode 阶段
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            next_token = self._infer_next([next_token], top_k, top_p, temperature)
            output_tokens.append(next_token)

        return output_tokens

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self.model)
        if len(inputs) == 0:
            return

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = max(int(max_new_tokens), 1)

        next_token = self._infer_next(inputs, top_k, top_p, temperature)
        yield next_token
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            next_token = self._infer_next([next_token], top_k, top_p, temperature)
            yield next_token

    def _infer_next(
        self,
        tokens: Sequence[int],
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> int:
        token_array = (c_int64 * len(tokens))(*tokens)
        top_k_i = int(top_k)
        top_p_f = float(top_p)
        temp_f = float(temperature)

        if top_k_i == 1 and top_p_f >= 1.0 and abs(temp_f - 1.0) < 1e-8:
            return int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    token_array,
                    len(tokens),
                )
            )

        return int(
            LIB_LLAISYS.llaisysQwen2ModelInferSample(
                self.model,
                token_array,
                len(tokens),
                c_int(top_k_i),
                c_float(top_p_f),
                c_float(temp_f),
            )
        )
    
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None
