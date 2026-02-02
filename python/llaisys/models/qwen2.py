from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, llaisysQwen2Model_t
from ..tensor import Tensor

from pathlib import Path
import json
import numpy as np
from ctypes import c_int64, c_size_t, c_int, c_float, pointer, byref, POINTER, cast
import struct


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # 读取配置文件
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # 提取模型参数
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_attention_heads)
        self.intermediate_size = config["intermediate_size"]
        self.vocab_size = config["vocab_size"]
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.max_position_embeddings = config.get("max_position_embeddings", 131072)

        # 计算每头维度
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 确定数据类型
        torch_dtype = config.get("torch_dtype", "bfloat16")
        if torch_dtype == "bfloat16":
            self.dtype = DataType.BF16
        elif torch_dtype == "float16":
            self.dtype = DataType.F16
        else:
            self.dtype = DataType.F32

        # 创建模型元数据
        meta = LlaisysQwen2Meta()
        meta.dtype = self.dtype
        meta.nlayer = self.num_hidden_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_attention_heads
        meta.nkvh = self.num_key_value_heads
        meta.dh = self.head_dim
        meta.di = self.intermediate_size
        meta.maxseq = self.max_position_embeddings
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = config.get("eos_token_id", 151643)

        # 创建设备ID数组
        device_ids = (c_int * 1)(0)

        # 创建模型
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            device_ids,
            1
        )

        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model")

        # 获取权重结构
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        self.weights = weights_ptr.contents

        # 加载权重
        self._load_weights(model_path)

        # 存储设备信息
        self.device = device

    def _load_weights(self, model_path: Path):
        """加载模型权重"""

        # 收集所有safetensors文件
        safetensor_files = sorted(model_path.glob("*.safetensors"))

        # 逐个文件加载权重
        for file in safetensor_files:
            self._load_weights_from_file(file)

    def _load_weights_from_file(self, file_path: Path):
        """从单个safetensors文件加载权重"""
        # 读取文件头
        with open(file_path, 'rb') as f:
            # 读取长度前缀 (8 bytes, little-endian unsigned long long)
            length_bytes = f.read(8)
            header_len = struct.unpack('<Q', length_bytes)[0]

            # 读取JSON头
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode('utf-8'))

            # 数据从头部之后开始
            data_start = 8 + header_len

            # 加载输入/输出嵌入和最终归一化权重
            self._load_tensor_from_file(f, header, data_start, "model.embed_tokens.weight", self.weights.in_embed)
            self._load_tensor_from_file(f, header, data_start, "lm_head.weight", self.weights.out_embed)
            self._load_tensor_from_file(f, header, data_start, "model.norm.weight", self.weights.out_norm_w)

            # 加载每层权重
            for layer_idx in range(self.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}."

                # Attention norm
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}input_layernorm.weight",
                    self.weights.attn_norm_w[layer_idx]
                )

                # Q projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.q_proj.weight",
                    self.weights.attn_q_w[layer_idx]
                )
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.q_proj.bias",
                    self.weights.attn_q_b[layer_idx]
                )

                # K projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.k_proj.weight",
                    self.weights.attn_k_w[layer_idx]
                )
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.k_proj.bias",
                    self.weights.attn_k_b[layer_idx]
                )

                # V projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.v_proj.weight",
                    self.weights.attn_v_w[layer_idx]
                )
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.v_proj.bias",
                    self.weights.attn_v_b[layer_idx]
                )

                # O projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}self_attn.o_proj.weight",
                    self.weights.attn_o_w[layer_idx]
                )

                # MLP norm
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}post_attention_layernorm.weight",
                    self.weights.mlp_norm_w[layer_idx]
                )

                # Gate projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}mlp.gate_proj.weight",
                    self.weights.mlp_gate_w[layer_idx]
                )

                # Up projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}mlp.up_proj.weight",
                    self.weights.mlp_up_w[layer_idx]
                )

                # Down projection
                self._load_tensor_from_file(
                    f, header, data_start,
                    f"{prefix}mlp.down_proj.weight",
                    self.weights.mlp_down_w[layer_idx]
                )

    def _load_tensor_from_file(self, f, header: dict, data_start: int, name: str, tensor_handle):
        """从文件加载单个张量"""
        if name not in header:
            return

        info = header[name]
        dtype = info['dtype']
        shape = info['shape']
        data_offsets = info['data_offsets']

        # 计算数据位置和大小
        start_offset = data_start + data_offsets[0]
        end_offset = data_start + data_offsets[1]
        size = end_offset - start_offset

        # 读取原始字节
        f.seek(start_offset)
        raw_data = f.read(size)

        # 根据数据类型解析
        if dtype == 'BF16':
            # BFloat16: 每个元素2字节，以uint16存储
            numpy_array = np.frombuffer(raw_data, dtype=np.uint16).reshape(shape)
        elif dtype == 'F16':
            numpy_array = np.frombuffer(raw_data, dtype=np.float16).reshape(shape)
        elif dtype == 'F32':
            numpy_array = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
        elif dtype == 'F64':
            numpy_array = np.frombuffer(raw_data, dtype=np.float64).reshape(shape)
        elif dtype == 'I64':
            numpy_array = np.frombuffer(raw_data, dtype=np.int64).reshape(shape)
        elif dtype == 'I32':
            numpy_array = np.frombuffer(raw_data, dtype=np.int32).reshape(shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # 确保数据是连续的
        if not numpy_array.flags['C_CONTIGUOUS']:
            numpy_array = np.ascontiguousarray(numpy_array)

        # 加载到tensor
        LIB_LLAISYS.tensorLoad(tensor_handle, numpy_array.ctypes.data)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本"""
        if max_new_tokens is None:
            max_new_tokens = 128

        # 准备输入token
        tokens = list(inputs)

        # 生成新token
        for _ in range(max_new_tokens):
            # 将token列表转换为ctypes数组
            token_array = (c_int64 * len(tokens))(*tokens)

            # 调用模型推理
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                token_array,
                len(tokens)
            )

            tokens.append(next_token)

            # 检查是否生成结束token
            if next_token == self.weights.end_token:
                break

        return tokens

    def __del__(self):
        """析构函数，释放模型资源"""
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None
