#!/usr/bin/env python3
"""
项目#4 多用户排队测试：并发向 /v1/chat/completions 发请求，验证排队与 503。

用法（先启动服务，如 --port 8002）：
  PYTHONPATH=. .venv/bin/python test/test_multi_user_chat.py --base-url http://127.0.0.1:8002
  PYTHONPATH=. .venv/bin/python test/test_multi_user_chat.py --base-url http://127.0.0.1:8002 --test-queue-full
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("需要 requests: pip install requests", file=sys.stderr)
    sys.exit(1)


def send_one(base_url: str, user_id: int, stream: bool = False, max_tokens: int = 20) -> dict:
    """发一条 chat 请求，返回 {user_id, status_code, elapsed, content_preview, error}。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    start = time.perf_counter()
    try:
        r = requests.post(
            url,
            json={
                "messages": [{"role": "user", "content": f"用户{user_id}说：请用一句话介绍你自己。"}],
                "stream": stream,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        if r.status_code != 200:
            return {"user_id": user_id, "status_code": r.status_code, "elapsed": elapsed, "error": r.text[:200]}
        if stream:
            content = ""
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                try:
                    import json as _json
                    part = _json.loads(line[5:].decode("utf-8"))
                    content += (part.get("choices") or [{}])[0].get("delta", {}).get("content", "") or ""
                except Exception:
                    pass
        else:
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return {
            "user_id": user_id,
            "status_code": r.status_code,
            "elapsed": round(elapsed, 2),
            "content_preview": (content[:80] + "…") if len(content) > 80 else content,
        }
    except Exception as e:
        return {"user_id": user_id, "status_code": -1, "elapsed": time.perf_counter() - start, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="多用户 chat 并发测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002", help="服务 base URL")
    parser.add_argument("--concurrent", type=int, default=3, help="并发请求数")
    parser.add_argument("--max-tokens", type=int, default=30, help="每条请求最多生成 token 数")
    parser.add_argument("--test-queue-full", action="store_true", help="测试队列满时是否返回 503（会发很多并发请求）")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    try:
        r = requests.get(base + "/health", timeout=5)
        if r.status_code != 200:
            print("服务未就绪或未加载模型，请先启动: python -m llaisys_py.server --model <path> --port 8002", file=sys.stderr)
            sys.exit(1)
    except requests.RequestException as e:
        print(f"无法连接 {base}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.test_queue_full:
        # 队列满测试：并发数大于服务端 LLAISYS_REQUEST_QUEUE_MAX（默认 64）时较难触发，这里用较大并发看是否有 503
        n = 70
        print(f"发送 {n} 个并发请求，检查是否出现 503（队列满）…")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(send_one, base, i, False, 5) for i in range(n)]
            results = [f.result() for f in as_completed(futures)]
        total = time.perf_counter() - start
        statuses = {}
        for res in results:
            sc = res.get("status_code", -1)
            statuses[sc] = statuses.get(sc, 0) + 1
        print(f"总耗时: {total:.1f}s, 状态码分布: {statuses}")
        if 503 in statuses:
            print("✓ 已出现 503，说明队列满时正确返回。")
        else:
            print("(未出现 503，可能队列容量较大或请求较快完成；可设置 LLAISYS_REQUEST_QUEUE_MAX=2 再试)")
        return

    print(f"并发 {args.concurrent} 个请求，每个 max_tokens={args.max_tokens}…")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = [ex.submit(send_one, base, i, False, args.max_tokens) for i in range(args.concurrent)]
        results = [f.result() for f in as_completed(futures)]
    total = time.perf_counter() - start

    print(f"总耗时: {total:.1f}s")
    for res in sorted(results, key=lambda x: x["user_id"]):
        u = res["user_id"]
        sc = res.get("status_code")
        el = res.get("elapsed", 0)
        err = res.get("error")
        preview = res.get("content_preview", "")
        if err:
            print(f"  用户{u}: status={sc} elapsed={el}s error={err[:60]}")
        else:
            print(f"  用户{u}: status={sc} elapsed={el}s -> {preview}")
    if all(r.get("status_code") == 200 for r in results):
        print("✓ 所有请求均成功，多用户排队正常。")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
