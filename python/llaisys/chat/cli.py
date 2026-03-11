import argparse
import ipaddress
import json
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse


def _post_json(url: str, payload: dict, accept: str = "application/json"):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": accept,
        },
        method="POST",
    )
    if _should_bypass_proxy(url):
        # 当前云环境默认注入了 HTTP_PROXY，访问本地服务时需要显式绕过代理，
        # 否则 127.0.0.1 请求会被转发到无效代理地址并返回 502。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=300)
    return urllib.request.urlopen(request, timeout=300)


def _should_bypass_proxy(url: str) -> bool:
    # 聊天服务通常本地起在回环地址上，这类请求应该直连，
    # 否则云环境里预置的 HTTP_PROXY 可能把它错误转发出去。
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _request_chat(base_url: str, payload: dict) -> dict:
    with _post_json(f"{base_url}/v1/chat/completions", payload) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_chat(base_url: str, payload: dict):
    # OpenAI 风格流式接口按 SSE 返回，这里逐行提取 data: 片段，
    # 再把每个 chunk 里的 delta.content 拼成最终回复。
    with _post_json(
        f"{base_url}/v1/chat/completions",
        payload,
        accept="text/event-stream",
    ) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if delta:
                yield delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llaisys-chat")
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-tokens", default=128, type=int)
    parser.add_argument("--system", default="")
    args = parser.parse_args()

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user_text = input("你> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            history = []
            if args.system:
                history.append({"role": "system", "content": args.system})
            print("已清空对话历史。")
            continue

        history.append({"role": "user", "content": user_text})
        # CLI 自己维护完整历史，服务端每次都按完整 messages 重新构造 prompt，
        # 这样实现简单，也符合课程要求的单用户会话场景。
        payload = {
            "model": args.model,
            "messages": history,
            "stream": args.stream,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        }

        try:
            if args.stream:
                assistant_chunks = []
                print("助手> ", end="", flush=True)
                for delta in _stream_chat(args.base_url, payload):
                    assistant_chunks.append(delta)
                    print(delta, end="", flush=True)
                print()
                assistant_text = "".join(assistant_chunks)
            else:
                response = _request_chat(args.base_url, payload)
                assistant_text = (
                    response["choices"][0]["message"].get("content", "")
                )
                print(f"助手> {assistant_text}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"请求失败: HTTP {exc.code}\n{body}", file=sys.stderr)
            history.pop()
            continue
        except Exception as exc:
            print(f"请求失败: {exc}", file=sys.stderr)
            history.pop()
            continue

        history.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
