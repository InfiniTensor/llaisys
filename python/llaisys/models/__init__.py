import json
from pathlib import Path
from typing import Type

from ..libllaisys import DeviceType
from .llama import Llama
from .qwen2 import Qwen2
from .tensor_parallel import TensorParallelQwen2

MODEL_CLASSES: tuple[Type[Qwen2], ...] = (Qwen2, Llama)


def detect_model_type(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as handle:
        config = json.load(handle)
    return str(config.get("model_type", "")).lower()


def create_model(
    model_path: str,
    device: DeviceType = DeviceType.CPU,
    device_id: int = 0,
    *,
    tp_size: int = 1,
    tp_device_ids=None,
):
    model_type = detect_model_type(model_path)
    if int(tp_size) > 1:
        if device != DeviceType.NVIDIA:
            raise ValueError("Tensor parallel inference currently supports only NVIDIA devices")
        if TensorParallelQwen2.supports_model_type(model_type):
            return TensorParallelQwen2(
                model_path,
                device,
                device_id,
                tp_size=int(tp_size),
                tp_device_ids=tp_device_ids,
            )
        raise ValueError(f"Tensor parallel inference is not supported for model type: {model_type or '<missing model_type>'}")
    for cls in MODEL_CLASSES:
        if cls.supports_model_type(model_type):
            return cls(model_path, device, device_id)
    raise ValueError(f"Unsupported model type: {model_type or '<missing model_type>'}")


def default_model_name(model_path: str) -> str:
    model_type = detect_model_type(model_path)
    return f"llaisys-{model_type or 'model'}"


__all__ = [
    "Qwen2",
    "Llama",
    "TensorParallelQwen2",
    "create_model",
    "detect_model_type",
    "default_model_name",
]
