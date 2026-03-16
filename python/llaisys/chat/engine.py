from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer

from .. import DeviceType
from ..models import Qwen2


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
    return str(content)


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = _normalize_content(message.get("content", ""))
        conversation.append({"role": role, "content": content})
    return conversation


class ChatEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ):
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.temperature = float(temperature)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if device.lower() == "nvidia":
            model_device = DeviceType.NVIDIA
        else:
            model_device = DeviceType.CPU
        self.model = Qwen2(model_path, model_device)

    def complete(
        self,
        messages: Iterable[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        conversation = _normalize_messages(messages)
        input_content = self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer.encode(input_content)

        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
            top_k=self.top_k if top_k is None else int(top_k),
            top_p=self.top_p if top_p is None else float(top_p),
            temperature=self.temperature if temperature is None else float(temperature),
            seed=seed,
        )

        prompt_tokens = len(input_ids)
        if len(out_ids) >= prompt_tokens:
            generated_ids = out_ids[prompt_tokens:]
        else:
            generated_ids = []
        assistant_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(generated_ids),
            "total_tokens": prompt_tokens + len(generated_ids),
            "output_ids": out_ids,
            "generated_ids": generated_ids,
            "assistant_text": assistant_text,
        }

    def iter_text_deltas(self, generated_ids: Iterable[int]):
        seen = ""
        buffer: List[int] = []
        for token_id in generated_ids:
            buffer.append(int(token_id))
            current = self.tokenizer.decode(buffer, skip_special_tokens=True)
            if current.startswith(seen):
                delta = current[len(seen):]
            else:
                delta = current
            seen = current
            if delta:
                yield delta
