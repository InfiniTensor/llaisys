from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, Any], str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            code = int(resp.status)
            data = json.loads(text) if text else {}
            return code, data, ""
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        data = {}
        try:
            data = json.loads(text) if text else {}
        except Exception:
            pass
        return int(exc.code), data, text or str(exc)
    except Exception as exc:
        return -1, {}, str(exc)


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
        return json.loads(text) if text else {}


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = p * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    w = rank - low
    return float(sorted_values[low] * (1 - w) + sorted_values[high] * w)


def run_benchmark(args: argparse.Namespace) -> int:
    endpoint = args.endpoint.rstrip("/")
    chat_url = f"{endpoint}/chat"
    scheduler_url = f"{endpoint}/debug/scheduler"
    health_url = f"{endpoint}/health"

    try:
        health = _get_json(health_url, timeout=args.timeout)
    except Exception as exc:
        print(f"[ERROR] health check failed: {exc}")
        return 2

    print(f"[INFO] health: {health}")
    before_debug: Dict[str, Any] = {}
    try:
        before_debug = _get_json(scheduler_url, timeout=args.timeout)
    except Exception:
        before_debug = {}

    if args.warmup > 0:
        print(f"[INFO] warmup requests: {args.warmup}")
        for i in range(args.warmup):
            payload: Dict[str, Any] = {
                "prompt": f"{args.prompt} [warmup-{i}]",
                "stream": False,
                "max_new_tokens": args.max_new_tokens,
            }
            _post_json(chat_url, payload, timeout=args.timeout)

    total = int(args.total_requests)
    concurrency = int(args.concurrency)
    print(f"[INFO] start benchmark: total={total}, concurrency={concurrency}, endpoint={chat_url}")

    t0 = time.perf_counter()
    latencies_ms: List[float] = []
    errors: List[str] = []
    status_count: Dict[int, int] = {}

    def _one_request(i: int) -> Tuple[float, int, str]:
        payload: Dict[str, Any] = {
            "prompt": f"{args.prompt} [req-{i}]",
            "stream": False,
            "max_new_tokens": args.max_new_tokens,
        }
        if args.session_mode == "shared":
            payload["session_id"] = args.shared_session_id
        elif args.session_mode == "unique":
            payload["session_id"] = f"{args.session_prefix}-{uuid.uuid4()}"
        if args.sampling:
            payload["sampling"] = args.sampling
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.top_p is not None:
            payload["top_p"] = args.top_p

        s = time.perf_counter()
        code, data, err = _post_json(chat_url, payload, timeout=args.timeout)
        elapsed_ms = (time.perf_counter() - s) * 1000.0
        if code == 200 and not data.get("error"):
            return elapsed_ms, code, ""
        detail = err or str(data.get("error") or f"HTTP {code}")
        return elapsed_ms, code, detail

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one_request, i) for i in range(total)]
        for fut in as_completed(futures):
            elapsed_ms, code, detail = fut.result()
            status_count[code] = status_count.get(code, 0) + 1
            if code == 200 and not detail:
                latencies_ms.append(elapsed_ms)
            else:
                errors.append(f"[{code}] {detail}")

    total_elapsed_s = max(1e-9, time.perf_counter() - t0)
    success = len(latencies_ms)
    failed = len(errors)
    throughput = total / total_elapsed_s

    latencies_sorted = sorted(latencies_ms)
    p50 = _percentile(latencies_sorted, 0.50)
    p95 = _percentile(latencies_sorted, 0.95)
    p99 = _percentile(latencies_sorted, 0.99)
    avg = statistics.mean(latencies_ms) if latencies_ms else 0.0

    after_debug: Dict[str, Any] = {}
    try:
        after_debug = _get_json(scheduler_url, timeout=args.timeout)
    except Exception:
        after_debug = {}

    print("\n=== Benchmark Summary ===")
    print(f"success: {success}/{total} ({(success / total) * 100:.1f}%)")
    print(f"failed: {failed}")
    print(f"elapsed_s: {total_elapsed_s:.3f}")
    print(f"throughput_rps: {throughput:.2f}")
    print(f"latency_ms: avg={avg:.1f}, p50={p50:.1f}, p95={p95:.1f}, p99={p99:.1f}")
    print(f"status_count: {status_count}")

    if before_debug:
        print("\n=== /debug/scheduler (before) ===")
        print(json.dumps(before_debug, ensure_ascii=False, indent=2))
    if after_debug:
        print("\n=== /debug/scheduler (after) ===")
        print(json.dumps(after_debug, ensure_ascii=False, indent=2))

    if errors:
        print("\n=== Sample Errors (up to 10) ===")
        for line in errors[:10]:
            print(line)
    return 0 if success > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLAISYS scheduler benchmark (non-stream chat requests)")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000", type=str)
    p.add_argument("--total-requests", default=20, type=int)
    p.add_argument("--concurrency", default=5, type=int)
    p.add_argument("--prompt", default="请用一句话介绍北京", type=str)
    p.add_argument("--max-new-tokens", default=32, type=int)
    p.add_argument("--timeout", default=60.0, type=float, help="per-request timeout in seconds")
    p.add_argument("--warmup", default=1, type=int)
    p.add_argument("--session-mode", choices=["none", "shared", "unique"], default="none")
    p.add_argument("--shared-session-id", default="bench-shared-session", type=str)
    p.add_argument("--session-prefix", default="bench-session", type=str)
    p.add_argument("--sampling", default="", type=str)
    p.add_argument("--temperature", default=None, type=float)
    p.add_argument("--top-k", default=None, type=int)
    p.add_argument("--top-p", default=None, type=float)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_benchmark(args))


if __name__ == "__main__":
    main()
