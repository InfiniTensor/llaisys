import argparse
import json
import sys
from typing import Any
from urllib import error, request


class ChatTUI:
    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stream: bool,
        system_prompt: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.messages: list[dict[str, str]] = []
        self.system_prompt = ""

        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, text: str) -> None:
        self.system_prompt = text
        self.messages = [m for m in self.messages if m.get("role") != "system"]
        self.messages.insert(0, {"role": "system", "content": text})

    def clear_history(self) -> None:
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def _post_json(self, path: str, payload: dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
            method="POST",
        )
        return request.urlopen(req, timeout=600)

    def _chat_non_stream(self) -> str:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": False,
        }
        with self._post_json("/v1/chat/completions", payload) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _chat_stream(self) -> str:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": True,
        }

        chunks: list[str] = []
        with self._post_json("/v1/chat/completions", payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue

                event = line[6:]
                if event == "[DONE]":
                    break

                try:
                    data = json.loads(event)
                except json.JSONDecodeError:
                    continue

                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue

                chunks.append(content)
                print(content, end="", flush=True)

        print("")
        return "".join(chunks)

    def _chat(self) -> str:
        if self.stream:
            return self._chat_stream()
        answer = self._chat_non_stream()
        print(answer)
        return answer

    def send_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        print("assistant> ", end="", flush=True)

        try:
            answer = self._chat()
        except error.HTTPError as ex:
            detail = ex.read().decode("utf-8", errors="ignore")
            print(f"\n[HTTP {ex.code}] {detail}")
            if self.messages and self.messages[-1]["role"] == "user" and self.messages[-1]["content"] == text:
                self.messages.pop()
            return
        except error.URLError as ex:
            print(f"\n[Network error] {ex}")
            if self.messages and self.messages[-1]["role"] == "user" and self.messages[-1]["content"] == text:
                self.messages.pop()
            return

        self.messages.append({"role": "assistant", "content": answer})

    def retry_last(self) -> None:
        if not self.messages:
            print("No history to retry.")
            return

        if self.messages[-1]["role"] == "assistant":
            self.messages.pop()

        last_user_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx == -1:
            print("No user message found to retry.")
            return

        print("assistant> ", end="", flush=True)
        try:
            answer = self._chat()
        except Exception as ex:  # keep retry resilient for terminal usage
            print(f"\n[Retry failed] {ex}")
            return

        self.messages.append({"role": "assistant", "content": answer})

    def print_help(self) -> None:
        print("Commands:")
        print("  /help                 Show this help")
        print("  /exit                 Exit TUI")
        print("  /clear                Clear local conversation history")
        print("  /retry                Regenerate assistant answer for last user turn")
        print("  /system <prompt>      Set or replace system prompt")

    def repl(self) -> None:
        print("LLAISYS Chat TUI")
        print(f"Server: {self.base_url}")
        print("Type /help for commands.")

        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not line:
                continue

            if line == "/exit":
                print("Bye.")
                break
            if line == "/help":
                self.print_help()
                continue
            if line == "/clear":
                self.clear_history()
                print("History cleared.")
                continue
            if line == "/retry":
                self.retry_last()
                continue
            if line.startswith("/system "):
                prompt = line[len("/system ") :].strip()
                if not prompt:
                    print("Usage: /system <prompt>")
                    continue
                self.set_system_prompt(prompt)
                print("System prompt updated.")
                continue

            self.send_user_message(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLAISYS chat TUI client")
    parser.add_argument("--url", default="http://127.0.0.1:9108", help="Base URL of chat server")
    parser.add_argument("--model", default="qwen2", help="Model name in request payload")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--no-stream", action="store_true", help="Disable SSE streaming")
    args = parser.parse_args(argv)

    tui = ChatTUI(
        base_url=args.url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=not args.no_stream,
        system_prompt=args.system or None,
    )
    tui.repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
