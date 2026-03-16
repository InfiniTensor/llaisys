"""
Phase 2 单测：KV Export/Import 与 suffix prefill。
用法：python test/test_kv_cache.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B
"""
import argparse
import os
import sys

# 确保能 import 项目里的 llaisys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llaisys_py
from llaisys_py.models.qwen2 import Qwen2
from llaisys_py.libllaisys import DeviceType
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="模型目录")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dev = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU
    model = Qwen2(args.model, device=dev)

    prompt = "你好，1+1等于几？"
    conv = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(text)
    if len(tokens) < 4:
        tokens = tokenizer.encode("你好。")  # 保证足够长以便 split

    prefix_len = max(1, len(tokens) // 2)
    print(f"[test_kv_cache] prompt len={len(tokens)} prefix_len={prefix_len}")

    # 1) 全量 prefill + 生成 2 个 token
    out1 = model.generate(tokens, max_new_tokens=2, temperature=0.0, top_k=1)
    print(f"[test_kv_cache] full prefill out len={len(out1)} cache_len={model.cache_len}")

    # 2) 导出 KV（只取前缀部分字节用于 import）
    blob = model.export_kv_cache()
    n_bytes = model.kv_cache_bytes(prefix_len)
    assert len(blob) >= n_bytes, f"export size {len(blob)} < prefix bytes {n_bytes}"
    print(f"[test_kv_cache] export size={len(blob)} prefix_bytes={n_bytes}")

    # 3) 导入前缀
    model.import_kv_cache(blob[:n_bytes], prefix_len)
    assert model.cache_len == prefix_len

    # 4) suffix prefill + 生成 2 个 token
    out2 = model.generate(tokens, max_new_tokens=2, temperature=0.0, top_k=1, prefix_len=prefix_len)
    print(f"[test_kv_cache] suffix prefill out len={len(out2)}")

    assert len(out2) >= prefix_len + 1, "suffix prefill should produce at least one new token"
    print("[test_kv_cache] OK: export/import and suffix prefill passed.")


if __name__ == "__main__":
    main()
