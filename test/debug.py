import argparse
from test_utils import *
import llaisys
import sys
import ctypes
from pathlib import Path

# 不需要 snapshot_download 了，因为你已经下载好了
# from huggingface_hub import snapshot_download 

def test_binding_only():
    print("--- Start Binding Test ---")
    
    # 1. 直接指定你刚才下载好的、确定的绝对路径
    # 注意：确保这个文件夹里真的有 .safetensors 文件
    real_model_path = "/home/cpp/ai-models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"1. Using local model at: {real_model_path}")

    # 2. 检查一下路径对不对 (防御性编程)
    if not Path(real_model_path).exists():
        print(f"!!! Error: Path does not exist: {real_model_path}")
        return

    # 3. 尝试加载 C++ 模型
    try:
        print("2. Calling C++ Qwen2 Init (Create + LoadWeights)...")
        
        # 直接传路径字符串！
        model = llaisys.models.Qwen2(real_model_path, llaisys_device("cpu"))
        
        print("3. Success! C++ Object Created & Weights Loaded.")
        print(f"   Model Object: {model}")
        
    except Exception as e:
        print(f"!!! Error Occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binding_only()