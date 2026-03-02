import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def stream_sse(url: str, payload: Dict[str, Any]):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data_part = line[6:]
            if data_part == "[DONE]":
                break
            yield json.loads(data_part)


def request_assistant_reply(
    url: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    stream: bool,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "stream": stream,
    }

    if not stream:
        obj = post_json(url, payload)
        return obj["choices"][0]["message"]["content"]

    pieces: List[str] = []
    for chunk in stream_sse(url, payload):
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        text = delta.get("content", "")
        if text:
            pieces.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(pieces)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive CLI for LLAISYS chat server")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions", type=str)
    parser.add_argument("--model", default="llaisys-qwen2", type=str)
    parser.add_argument("--system", default="", type=str)
    parser.add_argument("--max-tokens", default=256, type=int)
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    history: List[Dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    print("Interactive chat started.")
    print("Commands: /reset clears history, /exit quits.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("Bye.")
            return
        if user_text == "/reset":
            history = []
            if args.system:
                history.append({"role": "system", "content": args.system})
            print("History cleared.")
            continue

        history.append({"role": "user", "content": user_text})
        try:
            if not args.stream:
                print("Assistant: ", end="")
            reply = request_assistant_reply(
                url=args.url,
                model_name=args.model,
                messages=history,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                stream=args.stream,
            )
            if not args.stream:
                print(reply)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {body}")
            history.pop()
            continue
        except urllib.error.URLError as exc:
            print(f"Connection error: {exc}")
            history.pop()
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed: {exc}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
