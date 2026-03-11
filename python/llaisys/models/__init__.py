import json
from pathlib import Path

from .llama import Llama
from .qwen2 import Qwen2


def load_model(model_path, device):
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        candidates = list(Path(model_path).rglob("config.json"))
        if not candidates:
            raise FileNotFoundError("config.json not found under model_path")
        config_path = candidates[0]

    with open(config_path, "r", encoding="utf-8") as f:
        model_type = json.load(f).get("model_type", "")

    if model_type == "qwen2":
        return Qwen2(model_path, device)
    if model_type == "llama":
        return Llama(model_path, device)
    raise ValueError(f"Unsupported model_type: {model_type}")


__all__ = ["Qwen2", "Llama", "load_model"]
