"""
命令行聊天客户端：多轮与 LLAISYS Chatbot Server 对话。
用法：先启动 server，再在本机运行
  python -m llaisys_py.server.chat_cli
  python -m llaisys_py.server.chat_cli --base-url http://127.0.0.1:8000 --max-tokens 128
"""
import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("请安装 requests: pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chatbot 命令行客户端（多轮对话）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="服务器地址")
    parser.add_argument("--max-tokens", type=int, default=128, help="每轮最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    health_url = args.base_url.rstrip("/") + "/health"

    print("LLAISYS 命令行聊天（多轮对话）")
    print(f"服务器: {args.base_url}")
    try:
        r = requests.get(health_url, timeout=5)
        if r.status_code != 200:
            print("警告: /health 返回非 200，请确认服务已启动且已加载模型")
        else:
            d = r.json()
            print("模型已加载" if d.get("model_loaded") else "模型未加载，请求可能返回 503")
    except requests.RequestException as e:
        print(f"无法连接服务器: {e}")
        print("请先启动: python -m llaisys_py.server --model <path> --port 8000")
        sys.exit(1)

    print("输入内容后回车发送；输入 quit / exit / q 退出。\n")
    messages = []

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break

        messages.append({"role": "user", "content": user_input})
        payload = {
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "stream": False,
        }

        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            if hasattr(e, "response") and e.response is not None and e.response.text:
                print(e.response.text[:500])
            messages.pop()
            continue

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if not content:
            print("(无回复内容)")
        else:
            print("助手:", content)
        messages.append({"role": "assistant", "content": content})
        print()


if __name__ == "__main__":
    main()
