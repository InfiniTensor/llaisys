import ctypes
from ctypes import POINTER, c_int, c_long, c_size_t, c_float, c_void_p

from .llaisys_types import (
    llaisysDataType_t,
    llaisysDeviceType_t,
    to_data_type,
    to_torch_type,
)
from .tensor import llaisysTensor_t


class Qwen2WeightsNaming:
    def input_embed(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embed(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def mlp_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    def match(self, state_dict):
        return (
            self.input_embed() in state_dict
            and self.output_norm() in state_dict
            and self.output_embed() in state_dict
            and self.attn_q(0) in state_dict
        )


class Qwen2MetaCStruct(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_long),
    ]

    def __init__(self, config):
        super().__init__()
        self.dtype = to_data_type(config["torch_dtype"])
        self.nlayer = config["num_hidden_layers"]
        self.hs = config["hidden_size"]
        self.nh = config["num_attention_heads"]
        self.nkvh = config["num_key_value_heads"]
        self.dh = self.hs // self.nh
        self.di = config["intermediate_size"]
        self.maxseq = config["max_position_embeddings"]
        self.voc = config["vocab_size"]
        self.epsilon = config["rms_norm_eps"]
        self.theta = config["rope_theta"]
        self.end_token = config["eos_token_id"]

    def __str__(self):
        fields = []
        for name, _ in self._fields_:  # type: ignore
            value = getattr(self, name)
            fields.append(f"{name}: {value}")
        return "\n".join(fields)


class Qwen2WeightsCStruct(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

    def __init__(self, meta, state_dict, naming, ndev):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        di = meta.di

        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0

        torch_dtype = to_torch_type(meta.dtype)

        # in_embed
        input_embed_naming = naming.input_embed()
        self.input_embed_tensor = state_dict[input_embed_naming].to(torch_dtype)
        self.in_embed = self.input_embed_tensor.data_ptr()

        # out_embed
        output_embed_naming = naming.output_embed()
        self.output_embed_tensor = state_dict[output_embed_naming].to(torch_dtype)
        self.out_embed = self.output_embed_tensor.data_ptr()

        # out_norm_w
        output_norm_naming = naming.output_norm()
        self.output_norm_tensor = state_dict[output_norm_naming].to(torch_dtype)
        self.out_norm_w = self.output_norm_tensor.data_ptr()

        # attn_norm
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm_w = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # attn_q
        self.attn_q_tensors = [
            state_dict[naming.attn_q(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_q_ptrs = [self.attn_q_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_q_w = (c_void_p * nlayer)(*self.attn_q_ptrs)

        # attn_q_b
        self.attn_q_b_tensors = [
            state_dict[naming.attn_q_b(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_q_b_ptrs = [
            self.attn_q_b_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_q_b = (c_void_p * nlayer)(*self.attn_q_b_ptrs)

        # attn_k
        self.attn_k_tensors = [
            state_dict[naming.attn_k(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_k_ptrs = [self.attn_k_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_k_w = (c_void_p * nlayer)(*self.attn_k_ptrs)

        # attn_k_b
        self.attn_k_b_tensors = [
            state_dict[naming.attn_k_b(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_k_b_ptrs = [
            self.attn_k_b_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_k_b = (c_void_p * nlayer)(*self.attn_k_b_ptrs)

        # attn_v
        self.attn_v_tensors = [
            state_dict[naming.attn_v(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_v_ptrs = [self.attn_v_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_v_w = (c_void_p * nlayer)(*self.attn_v_ptrs)

        # attn_v_b
        self.attn_v_b_tensors = [
            state_dict[naming.attn_v_b(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_v_b_ptrs = [
            self.attn_v_b_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_v_b = (c_void_p * nlayer)(*self.attn_v_b_ptrs)

        # attn_o
        self.attn_o_tensors = [
            state_dict[naming.attn_o(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_o_w = (c_void_p * nlayer)(*self.attn_o_ptrs)

        # mlp_norm
        self.mlp_norm_tensors = [
            state_dict[naming.mlp_norm(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.mlp_norm_ptrs = [
            self.mlp_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.mlp_norm_w = (c_void_p * nlayer)(*self.mlp_norm_ptrs)

        # mlp_gate
        self.mlp_gate_tensors = [
            state_dict[naming.gate(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.mlp_gate_ptrs = [
            self.mlp_gate_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.mlp_gate_w = (c_void_p * nlayer)(*self.mlp_gate_ptrs)

        # mlp_up
        self.mlp_up_tensors = [
            state_dict[naming.up(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.mlp_up_ptrs = [self.mlp_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.mlp_up_w = (c_void_p * nlayer)(*self.mlp_up_ptrs)

        # mlp_down
        self.mlp_down_tensors = [
            state_dict[naming.down(i)].to(torch_dtype) for i in range(nlayer)
        ]
        self.mlp_down_ptrs = [
            self.mlp_down_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.mlp_down_w = (c_void_p * nlayer)(*self.mlp_down_ptrs)

    def release(self):
        """显式释放所有PyTorch张量引用和C指针数组"""
        # 1. 释放单个张量引用
        self.input_embed_tensor = None
        self.output_embed_tensor = None
        self.output_norm_tensor = None

        # 2. 释放列表中的张量引用
        tensor_lists = [
            "attn_norm_tensors",
            "attn_q_tensors",
            "attn_q_b_tensors",
            "attn_k_tensors",
            "attn_k_b_tensors",
            "attn_v_tensors",
            "attn_v_b_tensors",
            "attn_o_tensors",
            "mlp_norm_tensors",
            "mlp_gate_tensors",
            "mlp_up_tensors",
            "mlp_down_tensors",
        ]
        for attr in tensor_lists:
            if hasattr(self, attr):
                setattr(self, attr, None)  # 解除张量列表引用

        # 3. 释放C指针数组
        ptr_arrays = [
            "attn_norm_w",
            "attn_q_w",
            "attn_q_b",
            "attn_k_w",
            "attn_k_b",
            "attn_v_w",
            "attn_v_b",
            "attn_o_w",
            "mlp_norm_w",
            "mlp_gate_w",
            "mlp_up_w",
            "mlp_down_w",
        ]
        for ptr_attr in ptr_arrays:
            if hasattr(self, ptr_attr):
                setattr(self, ptr_attr, None)  # 清空指针数组


class Qwen2KVCacheCStruct(ctypes.Structure):
    _fields_ = [
        ("kcache", ctypes.POINTER(llaisysTensor_t)),
        ("vcache", ctypes.POINTER(llaisysTensor_t)),
    ]


class Qwen2ModelCStruct(ctypes.Structure):
    _fields_ = [
        ("meta", Qwen2MetaCStruct),
        ("weights", Qwen2WeightsCStruct),
        ("device", llaisysDeviceType_t),
        ("ndevice", c_int),
        ("device_ids", POINTER(c_int)),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(Qwen2MetaCStruct),
        POINTER(Qwen2WeightsCStruct),
        llaisysDeviceType_t,
        c_int,
        POINTER(c_int),
    ]
    lib.llaisysQwen2ModelCreate.restype = POINTER(Qwen2ModelCStruct)

    lib.llaisysQwen2KVCacheCreate.argtypes = [POINTER(Qwen2ModelCStruct), c_size_t]
    lib.llaisysQwen2KVCacheCreate.restype = POINTER(Qwen2KVCacheCStruct)

    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(Qwen2ModelCStruct)]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2KVCacheDestroy.argtypes = [POINTER(Qwen2KVCacheCStruct), c_size_t]
    lib.llaisysQwen2KVCacheDestroy.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        POINTER(Qwen2ModelCStruct),
        POINTER(c_long),
        c_size_t,
        POINTER(Qwen2KVCacheCStruct),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_long
