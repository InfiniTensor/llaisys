from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from transformers import AutoTokenizer

from ..libllaisys import DeviceType
from ..models import load_model


def _resolve_device(device_name: str) -> DeviceType:
    if device_name == "cpu":
        return DeviceType.CPU
    if device_name == "nvidia":
        return DeviceType.NVIDIA
    if device_name == "metax":
        return DeviceType.METAX
    raise ValueError(f"Unsupported device name: {device_name}")


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def normalize_messages(messages: Iterable[dict]) -> List[dict]:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": _normalize_content(message.get("content", "")),
            }
        )
    return normalized


def build_chat_prompt(tokenizer, messages: Iterable[dict]) -> str:
    normalized = normalize_messages(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        # 优先复用模型自带 chat template，这样提示词格式与 HF 官方推理保持一致。
        try:
            return tokenizer.apply_chat_template(
                conversation=normalized,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                normalized,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass

    fallback_lines: List[str] = []
    for message in normalized:
        role = message["role"].strip().lower()
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "User"
        fallback_lines.append(f"{prefix}: {message['content']}")
    fallback_lines.append("Assistant:")
    return "\n".join(fallback_lines)


@dataclass
class CompletionResult:
    prompt_text: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_text: str


class ChatService:
    def __init__(self, model_path: str, device_name: str = "cpu"):
        self.model_path = str(Path(model_path).expanduser())
        self.device_name = device_name
        self.device = _resolve_device(device_name)
        self.model = load_model(self.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model_name = Path(self.model_path).name

    def prepare_inputs(self, messages: Iterable[dict]) -> tuple[str, List[int]]:
        prompt_text = build_chat_prompt(self.tokenizer, messages)
        # 这里显式关闭 add_special_tokens，避免 tokenizer 再额外插入一套特殊 token，
        # 从而破坏 chat template 已经组织好的输入格式。
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not input_ids:
            raise ValueError("messages produced an empty prompt")
        return prompt_text, [int(token_id) for token_id in input_ids]

    def _trim_completion(self, token_ids: Sequence[int]) -> List[int]:
        trimmed = [int(token_id) for token_id in token_ids]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and trimmed and trimmed[-1] == int(eos_token_id):
            trimmed = trimmed[:-1]
        return trimmed

    def decode_completion(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def generate_completion(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ) -> tuple[List[int], str]:
        output_ids = self.model.generate(
            list(input_ids),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        # 后端 generate 返回的是“原始输入 + 新生成 token”的完整序列，
        # 这里把前缀 prompt 截掉，只保留回答部分。
        completion_ids = self._trim_completion(output_ids[len(input_ids):])
        return completion_ids, self.decode_completion(completion_ids)

    def generate_chat(
        self,
        messages: Iterable[dict],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ) -> CompletionResult:
        prompt_text, input_ids = self.prepare_inputs(messages)
        completion_ids, completion_text = self.generate_completion(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        return CompletionResult(
            prompt_text=prompt_text,
            prompt_token_ids=input_ids,
            completion_token_ids=completion_ids,
            completion_text=completion_text,
        )

    def stream_completion(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ):
        generated_token_ids: List[int] = []
        emitted_text = ""
        eos_token_id = self.tokenizer.eos_token_id

        # 这里按 token 增量推理，但对外按文本增量输出，避免 BPE 中间态乱码。
        for token_id, _ in self.model.stream_generate(
            list(input_ids),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        ):
            token_id = int(token_id)
            # 同时兼容后端 end_token 和 tokenizer 自己的 eos_token_id。
            if token_id == int(self.model.end_token):
                break
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            generated_token_ids.append(token_id)
            full_text = self.decode_completion(generated_token_ids)
            if full_text.startswith(emitted_text):
                delta_text = full_text[len(emitted_text):]
            else:
                delta_text = full_text
            emitted_text = full_text
            yield token_id, delta_text, list(generated_token_ids), full_text
