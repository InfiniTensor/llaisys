from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor

from pathlib import Path
import safetensors
import json
import numpy as np
from ctypes import c_int64, c_int, c_void_p, byref, POINTER


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self._model_ptr = None
        
        # 加载配置文件
        self._load_config()
        
        # 创建模型
        self._create_model()
        
        # 加载权重
        self._load_weights()

    def _load_config(self):
        """从config.json加载模型配置"""
        config_path = self.model_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建模型元数据
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.BF16  # 使用BFloat16
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = config["hidden_size"] // config["num_attention_heads"]
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = config.get("max_position_embeddings", 32768)
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = config.get("eos_token_id", 151645)

    def _create_model(self):
        """创建模型实例"""
        device_id = c_int(0)
        self._model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta), self.device, byref(device_id), 1
        )
        if not self._model_ptr:
            raise RuntimeError("Failed to create Qwen2 model")

    def _load_weights(self):
        """加载模型权重"""
        # 获取权重结构
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model_ptr)
        if not weights_ptr:
            raise RuntimeError("Failed to get model weights")
        
        weights = weights_ptr.contents
        
        # 权重名称映射
        weight_map = {}
        
        # 遍历所有safetensors文件
        for file in sorted(self.model_path.glob("*.safetensors")):
            print(f"Loading weights from {file.name}")
            data = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name in data.keys():
                weight_map[name] = data.get_tensor(name)
        
        print(f"Found {len(weight_map)} weight tensors")
        
        # 加载embedding权重
        if "model.embed_tokens.weight" in weight_map:
            embed_weight = weight_map["model.embed_tokens.weight"].astype(np.float16)
            weights.in_embed = self._create_tensor(embed_weight)
        
        if "lm_head.weight" in weight_map:
            lm_head_weight = weight_map["lm_head.weight"].astype(np.float16)
            weights.out_embed = self._create_tensor(lm_head_weight)
        
        # 加载最终层归一化
        if "model.norm.weight" in weight_map:
            norm_weight = weight_map["model.norm.weight"].astype(np.float16)
            weights.out_norm_w = self._create_tensor(norm_weight)
        
        # 加载每层权重
        print(f"Loading {self.meta.nlayer} transformer layers...")
        for layer_idx in range(self.meta.nlayer):
            self._load_layer_weights(weights, weight_map, layer_idx)
        
        print("All weights loaded successfully!")

    def _create_tensor(self, numpy_array):
        """从numpy数组创建tensor"""
        shape = list(numpy_array.shape)
        if numpy_array.dtype == np.float32:
            dtype = DataType.F32
        elif numpy_array.dtype == np.float16:
            dtype = DataType.F16
        else:
            # 默认转换为float16
            numpy_array = numpy_array.astype(np.float16)
            dtype = DataType.F16
        
        tensor = Tensor.create(shape, dtype, self.device, 0)
        tensor.load(numpy_array)
        return tensor._tensor
    
    def _load_layer_weights(self, weights, weight_map, layer_idx):
        """加载单层权重"""
        layer_prefix = f"model.layers.{layer_idx}"
        
        # Attention weights
        for weight_name, attr_name in [
            ("input_layernorm.weight", "attn_norm_w"),
            ("self_attn.q_proj.weight", "attn_q_w"),
            ("self_attn.q_proj.bias", "attn_q_b"),
            ("self_attn.k_proj.weight", "attn_k_w"),
            ("self_attn.k_proj.bias", "attn_k_b"),
            ("self_attn.v_proj.weight", "attn_v_w"),
            ("self_attn.v_proj.bias", "attn_v_b"),
            ("self_attn.o_proj.weight", "attn_o_w"),
            ("post_attention_layernorm.weight", "mlp_norm_w"),
            ("mlp.gate_proj.weight", "mlp_gate_w"),
            ("mlp.up_proj.weight", "mlp_up_w"),
            ("mlp.down_proj.weight", "mlp_down_w"),
        ]:
            full_name = f"{layer_prefix}.{weight_name}"
            if full_name in weight_map:
                weight_data = weight_map[full_name].astype(np.float16)
                tensor = self._create_tensor(weight_data)
                
                # 设置到权重数组中
                attr = getattr(weights, attr_name)
                if hasattr(attr, '__setitem__'):  # 如果是数组
                    attr[layer_idx] = tensor
                else:  # 如果是单个值
                    setattr(weights, attr_name, tensor)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本序列"""
        if not self._model_ptr:
            raise RuntimeError("Model not initialized")
        
        # 当前简化版本：只实现贪心解码（top_k=1）
        current_tokens = list(inputs)
        
        for _ in range(max_new_tokens):
            # 准备输入
            input_array = (c_int64 * len(current_tokens))(*current_tokens)
            
            # 调用推理
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model_ptr, input_array, len(current_tokens)
            )
            
            # 检查是否结束
            if next_token == self.meta.end_token or next_token < 0:
                break
            
            current_tokens.append(next_token)
            
            # 为了避免过长序列，只保留最近的tokens用于下一次推理
            if len(current_tokens) > 1024:
                current_tokens = current_tokens[-512:]  # 保留后半部分
        
        return current_tokens

    def __del__(self):
        """清理资源"""
        if self._model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_ptr)
            self._model_ptr = None
