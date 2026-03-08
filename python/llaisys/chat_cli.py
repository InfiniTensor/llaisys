import argparse
import json

try:
    import requests
except Exception as exc:  # pragma: no cover
    raise RuntimeError("requests is required. Install with: pip install requests") from exc


def stream_chat(base_url: str, payload: dict) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    with requests.post(url, json=payload, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        full_text = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[len("data: "):]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_text += content
        print()
        return full_text


def once_chat(base_url: str, payload: dict) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=3600)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="CLI chat UI for LLAISYS chat server")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llaisys-qwen2")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    args = parser.parse_args()

    messages = [{"role": "system", "content": args.system}]

    print("LLAISYS Chat CLI. Type /exit to quit.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stream": args.stream,
        }

        print("Assistant: ", end="", flush=True)
        if args.stream:
            assistant_text = stream_chat(args.server, payload)
        else:
            assistant_text = once_chat(args.server, payload)

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
