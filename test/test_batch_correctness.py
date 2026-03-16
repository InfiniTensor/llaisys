#!/usr/bin/env python3
"""
项目#4 批处理正确性测试（4.5.2）：对比「同一组请求分别单序列推理」与「同一组请求进 batch 推理」的输出是否一致。

在相同 seed 与确定性采样（top_k=1, temperature=0）下，顺序调用 model.generate 与通过 Engine 批量调度
应得到相同的 token 序列。

用法（需模型路径，可不启动 HTTP 服务）：
  PYTHONPATH=.:python python test/test_batch_correctness.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B
  或设置环境变量 MODEL_PATH
"""
import argparse
import os
import sys

# 确保能 import 项目包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

try:
    from transformers import AutoTokenizer
    from llaisys_py.models.qwen2 import Qwen2
    from llaisys_py.libllaisys import DeviceType
    from llaisys_py.server.engine import Engine, RequestState
except ImportError as e:
    print("依赖缺失:", e, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="批处理正确性：顺序 vs Engine 输出一致")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"), help="模型目录")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="每条最多生成 token 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=3, help="Engine 槽位数与请求数")
    args = parser.parse_args()
    if not args.model or not os.path.isdir(args.model):
        print("请提供有效模型目录: --model <path> 或 MODEL_PATH", file=sys.stderr)
        sys.exit(1)

    device = DeviceType.CPU
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2(args.model, device=device, max_batch_size=max(2, args.batch_size))

    prompts = [
        "你好，请用一句话介绍你自己。",
        "1+1等于几？",
        "写一个词：天空",
    ]
    prompts = prompts[: args.batch_size]
    # 确定性采样，便于对比
    temperature = 0.0
    top_k = 1
    top_p = 1.0

    # ---------- 顺序：每条单独 generate ----------
    sequential_outputs = []
    for p in prompts:
        inp = tokenizer.encode(p)
        full = model.generate(
            inp,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=args.seed,
        )
        new_tokens = full[len(inp) :]
        sequential_outputs.append(new_tokens)

    # ---------- Engine：同一组请求进 batch ----------
    import queue
    engine = Engine(
        model,
        max_batch_size=max(2, args.batch_size),
        pending_maxsize=16,
        get_kv=None,
        put_kv=None,
    )
    out_queues = []
    for p in prompts:
        inp = tokenizer.encode(p)
        q = queue.Queue()
        st = RequestState(
            request_id="test",
            prompt_tokens=inp,
            max_tokens=args.max_new_tokens,
            out_queue=q,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=args.seed,
        )
        engine.submit_request(st)
        out_queues.append((st, q))

    batch_outputs = []
    for st, q in out_queues:
        tokens = []
        while True:
            x = q.get(timeout=60)
            if x is None:
                break
            tokens.append(x)
        batch_outputs.append(tokens)

    # ---------- 对比 ----------
    ok = True
    for i, (seq_tok, batch_tok) in enumerate(zip(sequential_outputs, batch_outputs)):
        if seq_tok != batch_tok:
            print(f"请求 {i} 不一致: 顺序 len={len(seq_tok)} batch len={len(batch_tok)}", file=sys.stderr)
            print(f"  顺序: {seq_tok}", file=sys.stderr)
            print(f"  batch: {batch_tok}", file=sys.stderr)
            ok = False
    if ok:
        print("批处理正确性测试通过：顺序与 Engine 输出一致。")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
