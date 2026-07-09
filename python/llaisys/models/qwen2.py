from typing import Sequence, Optional, Union
from pathlib import Path
import json
import ctypes

from huggingface_hub import snapshot_download
import safetensors
import torch

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType, llaisysTensor_t
from ..libllaisys.models import load_qwen2, LlaisysQwen2Meta

load_qwen2(LIB_LLAISYS)

class Qwen2:
    """Qwen2 language model implementation."""
    
    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None, 
        device: DeviceType = DeviceType.CPU,
        max_seq_len: Optional[int] = None
    ):
        """Initialize Qwen2 model.
        
        Args:
            model_path: Path to model directory. If None, downloads default model.
            device: Device type for inference.
            max_seq_len: Maximum sequence length. If None, uses model's default.
                        Set this to a larger value (e.g., 8192, 16384) for longer contexts.
            
        Raises:
            ValueError: If unsupported device is specified.
            FileNotFoundError: If required model files are missing.
        """
        if device not in (DeviceType.CPU, DeviceType.NVIDIA):
            raise ValueError(f"Unsupported device: {device}. Only CPU and NVIDIA are supported.")
            
        self.model_path = self._resolve_model_path(model_path)
        self._validate_model_files()
        
        self.device = device
        self.device_id = 0
        
        config = self._load_config()
        self._init_model_params(config)
        
        # Override max sequence length if specified
        if max_seq_len is not None:
            if max_seq_len <= 0:
                raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
            print(f"[Qwen2] Overriding max_position_embeddings: {self.max_position_embeddings} → {max_seq_len}")
            self.max_position_embeddings = max_seq_len
        
        self.data_type = DataType.F32 if device == DeviceType.CPU else DataType.BF16
        
        self._create_model()
        
        self._load_weights()
        

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve model path, downloading if necessary."""
        if model_path is not None and Path(model_path).exists():
            return Path(model_path)
        return Path(snapshot_download(self.DEFAULT_MODEL_ID))

    def _validate_model_files(self) -> None:
        """Validate that required model files exist."""
        required_files = ["config.json", "model.safetensors"]
        for file_name in required_files:
            file_path = self.model_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_path} not found!")

    def _load_config(self) -> dict:
        """Load model configuration from config.json."""
        config_path = self.model_path / "config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def _init_model_params(self, config: dict) -> None:
        """Initialize model parameters from configuration."""
        # Required parameters
        required_params = [
            "hidden_size", "intermediate_size", "max_position_embeddings",
            "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
            "rms_norm_eps", "rope_theta", "vocab_size", "eos_token_id"
        ]
        
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
                
        self.eos_token_id = config["eos_token_id"]
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config["rope_theta"]
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.vocab_size = config["vocab_size"]
        
        # Derived parameters
        self.per_head_dim = self.hidden_size // self.num_attention_heads
        self.per_kvhead_dim = self.per_head_dim  # For Qwen2, dv = d

    def _create_model(self) -> None:
        """Create the model instance."""
        meta = LlaisysQwen2Meta(
            dtype=self.data_type,
            nlayer=self.num_hidden_layers,
            hs=self.hidden_size,
            nh=self.num_attention_heads,
            nkvh=self.num_key_value_heads,
            dh=self.per_head_dim,
            di=self.intermediate_size,
            maxseq=self.max_position_embeddings,
            voc=self.vocab_size,
            epsilon=self.rms_norm_eps,
            theta=self.rope_theta,
            end_token=self.eos_token_id
        )

        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            ctypes.c_int(self.device),
            device_ids,
            ctypes.c_int(1)
        )

        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")

    def _load_weights(self) -> None:
        """Load model weights from safetensors files."""
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights:
            raise RuntimeError("Failed to get Qwen2 weights.")
        def maybe_cast_tensor(tensor):
            """Cast tensor to appropriate dtype if needed."""
            if self.device == DeviceType.CPU:
                return tensor.to(torch.float32).contiguous()
            elif self.device == DeviceType.NVIDIA:
                return tensor.to(torch.bfloat16).contiguous()
            return tensor
        
        
        for file in sorted(self.model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="torch", device="cpu")
            self._load_embedding_layers(data, weights, maybe_cast_tensor)
            self._load_attention_layers(data, weights, maybe_cast_tensor)
            self._load_mlp_layers(data, weights, maybe_cast_tensor)
        
    
    def _load_embedding_layers(self, data, weights, cast_fn):
        """Load embedding and output layers."""
        embedding_mappings = [
            ("model.embed_tokens.weight", "in_embed"),
            ("lm_head.weight", "out_embed"),
            ("model.norm.weight", "out_norm_w")
        ]
        
        for tensor_name, field_name in embedding_mappings:
            tensor = cast_fn(data.get_tensor(tensor_name))
            LIB_LLAISYS.tensorLoad(getattr(weights.contents, field_name), tensor.data_ptr())

    def _load_attention_layers(self, data, weights, cast_fn):
        """Load self-attention layer weights."""
        attention_mappings = [
            ("input_layernorm.weight", "attn_norm_w"),
            ("self_attn.q_proj.weight", "attn_q_w"),
            ("self_attn.q_proj.bias", "attn_q_b"),
            ("self_attn.k_proj.weight", "attn_k_w"),
            ("self_attn.k_proj.bias", "attn_k_b"),
            ("self_attn.v_proj.weight", "attn_v_w"),
            ("self_attn.v_proj.bias", "attn_v_b"),
            ("self_attn.o_proj.weight", "attn_o_w"),
        ]
        
        for base_name, field_name in attention_mappings:
            self._load_layer_array(data, weights, field_name, base_name, cast_fn)

    def _load_mlp_layers(self, data, weights, cast_fn):
        """Load MLP layer weights."""
        mlp_mappings = [
            ("post_attention_layernorm.weight", "mlp_norm_w"),
            ("mlp.gate_proj.weight", "mlp_gate_w"),
            ("mlp.up_proj.weight", "mlp_up_w"),
            ("mlp.down_proj.weight", "mlp_down_w"),
        ]
        
        for base_name, field_name in mlp_mappings:
            self._load_layer_array(data, weights, field_name, base_name, cast_fn)

    def _load_layer_array(self, data, weights, field_name, base_name, cast_fn):
        """Load weights for a layer array."""
        arr_ptr = getattr(weights.contents, field_name)
        arr_type = llaisysTensor_t * self.num_hidden_layers
        arr = ctypes.cast(arr_ptr, ctypes.POINTER(arr_type)).contents

        for i in range(self.num_hidden_layers):
            tensor_name = f"model.layers.{i}.{base_name}"
            tensor = cast_fn(data.get_tensor(tensor_name))
            LIB_LLAISYS.tensorLoad(arr[i], tensor.data_ptr())

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        use_cache: bool = True,
        kcache_array=None,
        vcache_array=None,
        past_len: int = 0,
        seed: Optional[int] = None,
    ) -> Sequence[int]:
        """Generate tokens using the model.
        
        Args:
            inputs: Input token IDs.
                - use_cache=True with external cache: incremental input (current request tail).
                - use_cache=False: full input sequence.
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter  
            temperature: Sampling temperature
            use_cache: Whether to use KV cache for efficiency.
            kcache_array: External KV cache k tensor array managed outside model.
            vcache_array: External KV cache v tensor array managed outside model.
            past_len: Number of valid tokens already written in external KV cache.
            seed: Optional random seed for reproducible sampling. If set,
                each decode step uses `seed + step_index`.
            
        Returns:
            Token IDs for current request + current output.
            - use_cache=True: returns `inputs + new_tokens`.
            - use_cache=False: returns `inputs[past_len:] + new_tokens`.
        """
        if not inputs:
            raise ValueError("Input tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k == 0:
            raise ValueError("top_k cannot be 0")
        if top_p < 0 or top_p > 1:
            raise ValueError("top_p must be in [0, 1]")
        if past_len < 0:
            raise ValueError("past_len must be non-negative")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")
            
        generated = list(inputs)
        use_external_cache = bool(use_cache and kcache_array is not None and vcache_array is not None)
        null_cache = ctypes.POINTER(llaisysTensor_t)()
        active_kcache = kcache_array if use_external_cache else null_cache
        active_vcache = vcache_array if use_external_cache else null_cache

        if not use_external_cache and past_len > len(generated):
            raise ValueError(
                f"past_len={past_len} cannot exceed input length={len(generated)} for the current call"
            )

        output = list(generated) if use_external_cache else list(generated[past_len:])
        cached_token_len = past_len
        sample_step = 0

        # Prefill phase
        prefill_seed = (seed + sample_step) if seed is not None else -1
        next_token = self._infer_tokens(generated, active_kcache, active_vcache,
                        past_len if use_external_cache else 0, temperature, top_k, top_p,
                        prefill_seed)
        output.append(next_token)
        generated.append(next_token)
        cached_token_len +=len(output)-1
        sample_step += 1
        

        # Decode phase  
        for _ in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                break
                
            if use_external_cache:
                step_seed = (seed + sample_step) if seed is not None else -1
                next_token = self._infer_tokens([next_token], active_kcache, active_vcache, cached_token_len,
                                                temperature, top_k, top_p, step_seed)
                cached_token_len += 1
            else:
                step_seed = (seed + sample_step) if seed is not None else -1
                next_token = self._infer_tokens(generated, active_kcache, active_vcache, 0,
                                                temperature, top_k, top_p, step_seed)
                
            generated.append(next_token)
            output.append(next_token)
            sample_step += 1

        return output

    def _infer_tokens(self, tokens: Sequence[int], kcache_array, vcache_array,
                      past_len: int, temperature: float, top_k: int, top_p: float, seed: int = 42) -> int:
        """Perform inference on token sequence."""
        ntokens = len(tokens)
        TokenArrayType = ctypes.c_int64 * ntokens
        input_token_array = TokenArrayType(*tokens)

        return LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_token_array,
            ctypes.c_size_t(ntokens),
            kcache_array,
            vcache_array,
            ctypes.c_size_t(past_len),
            ctypes.c_float(temperature),
            ctypes.c_int(top_k),
            ctypes.c_float(top_p),
            ctypes.c_int64(seed),
        )