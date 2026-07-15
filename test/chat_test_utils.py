import socket
import time
from itertools import count
from threading import Lock
from typing import Iterator, Tuple


class FakeTokenizer:
    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        rendered = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            rendered.append(f"{role}:{content}" if content else f"{role}:")
        prompt = "\n".join(rendered)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return prompt

    def encode(self, text, return_tensors=None):
        tokens = [ord(ch) for ch in text]
        if return_tensors == "pt":
            raise NotImplementedError("FakeTokenizer does not support tensor outputs")
        return tokens or [0]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids if 0 <= token_id < 256)


class _Meta:
    end_token = 33


class FakeModel:
    _instance_ids = count()

    def __init__(self, *, delay_s: float = 0.0, calls_store=None):
        self.meta = _Meta()
        self._reply = [72, 101, 108, 108, 111, 33]
        self.delay_s = float(delay_s)
        self.instance_id = next(self._instance_ids)
        self.calls = calls_store if calls_store is not None else []
        self._calls_lock = Lock()
        self._base_prompt_len = 0
        self._cur_pos = 0

    def _record(self, payload):
        with self._calls_lock:
            self.calls.append(payload)

    def generate_next(self, inputs, top_k=1, top_p=1.0, temperature=1.0, reset_state=False):
        if reset_state:
            self.reset()
        if not inputs:
            raise ValueError("inputs must not be empty")

        self._record(
            {
                "mode": "generate_next",
                "inputs": list(inputs),
                "reset_state": reset_state,
                "instance_id": self.instance_id,
            }
        )
        if self.delay_s > 0:
            time.sleep(self.delay_s)

        if reset_state or self._base_prompt_len == 0 or len(inputs) > 1 or self._cur_pos < self._base_prompt_len:
            self._base_prompt_len = self._cur_pos + len(inputs)

        self._cur_pos += len(inputs)
        index = max(0, self._cur_pos - self._base_prompt_len)
        if index >= len(self._reply):
            index = len(self._reply) - 1
        return self._reply[index]

    def generate(self, inputs, max_new_tokens=20, top_k=1, top_p=1.0, temperature=1.0, reset_state=True):
        self._record(
            {
                "mode": "generate",
                "inputs": list(inputs),
                "reset_state": reset_state,
                "instance_id": self.instance_id,
            }
        )
        generated = []
        token_source = list(inputs)
        for step in range(max_new_tokens):
            token = self.generate_next(
                token_source,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=reset_state if step == 0 else False,
            )
            generated.append(token)
            token_source = [token]
            if token == self.meta.end_token:
                break
        return list(inputs) + generated

    def stream_generate(
        self,
        inputs,
        *,
        tokenizer=None,
        max_new_tokens=20,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        reset_state=True,
    ) -> Iterator[Tuple[int, str]]:
        self._record(
            {
                "mode": "stream_generate",
                "inputs": list(inputs),
                "reset_state": reset_state,
                "instance_id": self.instance_id,
            }
        )
        produced = []
        token_source = list(inputs)
        for step in range(max_new_tokens):
            token = self.generate_next(
                token_source,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=reset_state if step == 0 else False,
            )
            produced.append(token)
            text = tokenizer.decode(produced, skip_special_tokens=True, clean_up_tokenization_spaces=False) if tokenizer else ""
            prev = tokenizer.decode(produced[:-1], skip_special_tokens=True, clean_up_tokenization_spaces=False) if tokenizer else ""
            yield token, text[len(prev):] if text.startswith(prev) else text
            token_source = [token]
            if token == self.meta.end_token:
                break

    def truncate(self, position: int) -> None:
        self._cur_pos = int(position)
        if self._cur_pos == 0:
            self._base_prompt_len = 0
        self._record({"mode": "truncate", "position": int(position), "instance_id": self.instance_id})

    def reset(self) -> None:
        self._cur_pos = 0
        self._base_prompt_len = 0
        self._record({"mode": "reset", "instance_id": self.instance_id})


def serve_fake_app(port: int) -> None:
    import uvicorn
    from llaisys.chat_server import create_app

    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def wait_for_server(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"Server on port {port} did not start in time")
