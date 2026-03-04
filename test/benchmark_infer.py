import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Dict, List


PROMPT_PRESETS: Dict[str, str] = {
    "short": "Who are you?",
    "medium": (
        "Explain the role of KV cache in transformer decoding, and give a short "
        "step-by-step example with one prompt token and two generated tokens."
    ),
    "long": (
        "I am building a tiny LLM inference system from scratch. Please provide a "
        "concise engineering checklist that covers model loading, tensor layout, "
        "runtime abstraction, memory reuse, operator profiling, and end-to-end "
        "benchmarking. Keep the answer practical and implementation-oriented."
    ),
}


JSON_SENTINEL = "__BENCH_JSON__"


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_case(latencies: List[float], new_tokens: List[int]) -> Dict[str, float]:
    mean_s = statistics.mean(latencies)
    return {
        "mean_ms": mean_s * 1000.0,
        "p50_ms": percentile(latencies, 0.50) * 1000.0,
        "p95_ms": percentile(latencies, 0.95) * 1000.0,
        "min_ms": min(latencies) * 1000.0,
        "max_ms": max(latencies) * 1000.0,
        "mean_new_tokens": statistics.mean(new_tokens),
        "tokens_per_sec": (statistics.mean(new_tokens) / mean_s) if mean_s > 0 else 0.0,
    }


def hash_tokens(tokens: List[int]) -> str:
    payload = ",".join(str(x) for x in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_torch_case(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    device: str,
):
    import torch

    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)

    if device == "nvidia":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    if device == "nvidia":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    out_tokens = outputs[0].tolist()
    new_tokens = len(out_tokens) - int(inputs.shape[1])
    return elapsed, new_tokens, out_tokens


def run_llaisys_case(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    device: str,
):
    import llaisys
    from test_utils import llaisys_device

    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)

    api = llaisys.RuntimeAPI(llaisys_device(device))
    api.device_synchronize()
    start = time.perf_counter()
    out_tokens = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    api.device_synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = len(out_tokens) - len(inputs)
    return elapsed, new_tokens, out_tokens


def worker_main(args):
    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    cases = json.loads(args.cases_json)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.backend == "torch":
        import torch
        from transformers import AutoModelForCausalLM
        from test_utils import torch_device

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=torch_device(args.device),
            trust_remote_code=True,
        )

        runner = run_torch_case
    elif args.backend == "llaisys":
        import llaisys
        from test_utils import llaisys_device

        model = llaisys.models.Qwen2(model_path, llaisys_device(args.device))
        runner = run_llaisys_case
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    all_results = []
    for case in cases:
        prompt_name = case["prompt_name"]
        prompt = case["prompt"]
        max_new_tokens = int(case["max_new_tokens"])

        for _ in range(args.warmup):
            runner(
                tokenizer,
                model,
                prompt,
                max_new_tokens,
                args.top_k,
                args.top_p,
                args.temperature,
                args.device,
            )

        latencies: List[float] = []
        generated: List[int] = []
        first_tokens: List[int] = []
        for i in range(args.repeat):
            elapsed, new_tokens, out_tokens = runner(
                tokenizer,
                model,
                prompt,
                max_new_tokens,
                args.top_k,
                args.top_p,
                args.temperature,
                args.device,
            )
            latencies.append(elapsed)
            generated.append(new_tokens)
            if i == 0:
                first_tokens = out_tokens

        summary = summarize_case(latencies, generated)
        all_results.append(
            {
                "backend": args.backend,
                "prompt_name": prompt_name,
                "max_new_tokens": max_new_tokens,
                **summary,
                "output_hash": hash_tokens(first_tokens),
                "output_len": len(first_tokens),
            }
        )

    print(JSON_SENTINEL + json.dumps({"backend": args.backend, "results": all_results}))


def run_worker_subprocess(
    backend: str,
    model: str,
    device: str,
    cases: List[Dict[str, object]],
    warmup: int,
    repeat: int,
    top_k: int,
    top_p: float,
    temperature: float,
):
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--backend",
        backend,
        "--model",
        model,
        "--device",
        device,
        "--cases-json",
        json.dumps(cases),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--temperature",
        str(temperature),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"{backend} worker failed:\n{proc.stdout}")

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(JSON_SENTINEL):
            payload = json.loads(line[len(JSON_SENTINEL):])
            break
    if payload is None:
        raise RuntimeError(f"Failed to parse worker output for {backend}:\n{proc.stdout}")
    return payload


def print_report(rows: List[Dict[str, object]], deterministic: bool, backends: List[str]):
    key_order = sorted({(r["prompt_name"], r["max_new_tokens"]) for r in rows}, key=lambda x: (x[0], x[1]))
    row_map = {(r["backend"], r["prompt_name"], r["max_new_tokens"]): r for r in rows}

    print("\n=== Comprehensive Inference Benchmark ===")
    print("| Case | Backend | mean(ms) | p50(ms) | p95(ms) | new_tokens | tok/s | output_match |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for prompt_name, max_new_tokens in key_order:
        ref_hash = None
        if deterministic and len(backends) >= 2:
            ref = row_map.get((backends[0], prompt_name, max_new_tokens))
            ref_hash = ref["output_hash"] if ref else None

        for backend in backends:
            row = row_map.get((backend, prompt_name, max_new_tokens))
            if row is None:
                continue
            match = "-"
            if ref_hash is not None:
                match = "Y" if row["output_hash"] == ref_hash else "N"
            case_name = f"{prompt_name}/{max_new_tokens}"
            print(
                f"| {case_name} | {backend} | "
                f"{row['mean_ms']:.2f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
                f"{row['mean_new_tokens']:.1f} | {row['tokens_per_sec']:.2f} | {match} |"
            )


def orchestrator_main(args):
    prompt_names = parse_csv_strings(args.prompts)
    max_new_tokens_list = parse_csv_ints(args.max_new_tokens)
    backends = parse_csv_strings(args.backends)

    for name in prompt_names:
        if name not in PROMPT_PRESETS:
            raise ValueError(f"Unknown prompt preset: {name}. Valid keys: {list(PROMPT_PRESETS.keys())}")

    cases = []
    for prompt_name in prompt_names:
        for max_new_tokens in max_new_tokens_list:
            cases.append(
                {
                    "prompt_name": prompt_name,
                    "prompt": PROMPT_PRESETS[prompt_name],
                    "max_new_tokens": max_new_tokens,
                }
            )

    all_rows: List[Dict[str, object]] = []
    for backend in backends:
        payload = run_worker_subprocess(
            backend=backend,
            model=args.model,
            device=args.device,
            cases=cases,
            warmup=args.warmup,
            repeat=args.repeat,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        all_rows.extend(payload["results"])

    deterministic = (
        args.top_k == 1
        and abs(args.top_p - 1.0) < 1e-8
        and abs(args.temperature - 1.0) < 1e-8
    )
    print_report(all_rows, deterministic=deterministic, backends=backends)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "device": args.device,
                    "backends": backends,
                    "prompts": prompt_names,
                    "max_new_tokens": max_new_tokens_list,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "results": all_rows,
                },
                f,
                indent=2,
            )
        print(f"\nSaved JSON report to: {args.json_out}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Path to local model directory.")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"], type=str)
    parser.add_argument("--backends", default="torch,llaisys", type=str)
    parser.add_argument("--prompts", default="short,medium,long", type=str)
    parser.add_argument("--max-new-tokens", default="32,64,128", type=str)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--json-out", default="", type=str)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", default="", choices=["", "torch", "llaisys"])
    parser.add_argument("--cases-json", default="", type=str)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        worker_main(args)
    else:
        orchestrator_main(args)
