import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _post_json(url: str, payload: Dict[str, Any], stream: bool):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=600)


def _send_chat(
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: Optional[int],
    stream: bool,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "stream": bool(stream),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    if not stream:
        with _post_json(endpoint, payload, stream=False) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    assistant_text = ""
    with _post_json(endpoint, payload, stream=True) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            content = line[5:].strip()
            if content == "[DONE]":
                break

            chunk = json.loads(content)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                assistant_text += piece
    sys.stdout.write("\n")
    sys.stdout.flush()
    return assistant_text


def main():
    parser = argparse.ArgumentParser(description="LLAISYS interactive chat CLI")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2")
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--system", default=None, type=str)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    endpoint = args.server.rstrip("/") + "/v1/chat/completions"
    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Commands: /exit to quit, /reset to clear history.")
    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            if args.stream:
                sys.stdout.write("Assistant: ")
                sys.stdout.flush()
            assistant_text = _send_chat(
                endpoint=endpoint,
                model=args.model,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                stream=args.stream,
            )
            if not args.stream:
                print(f"Assistant: {assistant_text}")
            messages.append({"role": "assistant", "content": assistant_text})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # pragma: no cover - network/runtime errors
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
