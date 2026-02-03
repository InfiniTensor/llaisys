#!/usr/bin/env python3
"""简单测试KV-Cache功能"""
import sys
sys.path.insert(0, '/home/lapuluma/llaisys/python')

import llaisys
from pathlib import Path

print("Testing KV-Cache implementation...")

# 加载模型
model_path = Path("/home/lapuluma/llaisys/DeepSeek-R1-Distill-Qwen-1.5B")
print(f"Loading model from {model_path}")

model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
print("Model loaded successfully!")

# 测试tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备输入
prompt = "Hello"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content)
print(f"Input tokens: {inputs}")

# 只生成2个token来测试
print("Generating 2 tokens with KV-Cache...")
tokens = model.generate(inputs, max_new_tokens=2)
print(f"Output tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens, skip_special_tokens=True)}")

print("\nKV-Cache test completed!")
