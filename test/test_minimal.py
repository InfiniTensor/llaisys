import ctypes
import os
import sys
from tqdm import tqdm
import llaisys.models
from llaisys import LIB_LLAISYS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
from test_utils import torch_device


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


tokenizer, hf_model, hf_model_path = load_hf_model(
    model_path="./data",
    device_name="cpu",
)
sentence = "Who are you?"
MAX_TOKENS = 128

input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": sentence}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content, return_tensors="pt").to("cpu")



hf_output = hf_model.generate(
    inputs,
    max_new_tokens=MAX_TOKENS,
    top_k=1,
    top_p=1,
    temperature=1,
)


model = llaisys.models.Qwen2(model_path="./data")

listinput = inputs[0].tolist()
output = model.generate(listinput, max_new_tokens=MAX_TOKENS)
# listinput.append(output)
print(output)
print(tokenizer.decode(output, skip_special_tokens=True))
print(hf_output)
print(tokenizer.decode(hf_output, skip_special_tokens=True))


"""
Answer:
91786
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]
"""
