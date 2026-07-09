import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ServerConfig:
    def __init__(self) -> None:
        self.default_local_model_path = "/home/vankari/code/DeepSeek-R1-Distill-Qwen-1.5B/"
        self.configured_model_path = os.getenv("LLAISYS_MODEL_PATH", self.default_local_model_path)
        self.model_path = self._resolve_existing_model_path(self.configured_model_path)
        self.device = os.getenv("LLAISYS_DEVICE", "cpu").strip().lower()
        self.default_max_new_tokens = int(os.getenv("LLAISYS_MAX_NEW_TOKENS", "128"))
        self.default_top_k = int(os.getenv("LLAISYS_TOP_K", "10"))
        self.default_top_p = float(os.getenv("LLAISYS_TOP_P", "0.8"))
        self.default_temperature = float(os.getenv("LLAISYS_TEMPERATURE", "0.8"))
        self.default_use_cache = _env_bool("LLAISYS_USE_CACHE", True)
        self.default_model_name = os.getenv("LLAISYS_SERVE_MODEL_NAME", "llaisys-qwen2")

        if self.device not in {"cpu", "nvidia"}:
            raise ValueError("LLAISYS_DEVICE must be 'cpu' or 'nvidia'")

    @staticmethod
    def _resolve_existing_model_path(path_value: str | None) -> str | None:
        if path_value is None:
            return None
        candidate = Path(path_value).expanduser()
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
        return None


CONFIG = ServerConfig()
