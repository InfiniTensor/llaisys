import json
from pathlib import Path

from .qwen2 import Qwen2

try:
    from .llama import Llama
except ImportError:
    Llama = None


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
        # 当前仓库可能没有同步 llama Python 封装；这里显式报错，避免导入阶段提前失败。
        if Llama is None:
            raise RuntimeError("Llama python wrapper is not available in this checkout")
        return Llama(model_path, device)
    raise ValueError(f"Unsupported model_type: {model_type}")


__all__ = ["Qwen2", "Llama", "load_model"]
