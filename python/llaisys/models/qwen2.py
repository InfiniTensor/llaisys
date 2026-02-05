from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import safetensors
import json


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        self.model_path = Path(model_path)
        self.device = device
        self.weights = {}
        self._load_config()
        self._load_weights()
        self._init_model()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []
    def _load_config(self):
        config_path = self.model_path / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_hidden_layers"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.intermediate_size = self.config["intermediate_size"]
        self.max_position_embeddings = self.config["max_position_embeddings"]
        self.rms_norm_eps = self.config["rms_norm_eps"]
        self.rope_theta = self.config["rope_theta"]

    def _load_weights(self):
        for file in sorted(self.model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                tensor = data_.get_tensor(name_)
                print(f"Loaded weight: {name_}, shape: {tensor.shape}, dtype: {tensor.dtype}")
        pass

    def _init_model(self):
        pass
