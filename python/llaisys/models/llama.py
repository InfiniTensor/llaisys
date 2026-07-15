from .qwen2 import Qwen2


class Llama(Qwen2):
    SUPPORTED_MODEL_TYPES = {"llama", "mistral"}
