"""
启动 Chatbot Server。
用法:
  python -m llaisys_py.server   # 使用下方 DEFAULT_MODEL_PATH 或环境变量 MODEL_PATH
  python -m llaisys_py.server --port 8000 --device nvidia
"""
# 最先修正 CUDA_VISIBLE_DEVICES：若为空串，CUDA 会认为 0 张卡且之后无法更改。
# 必须在 import 任何会间接加载 torch/CUDA 的模块之前执行（如 create_app -> transformers -> torch）。
import os
import sys
# 发生 segfault 时打印 Python 栈，便于定位是否在 C 扩展内崩溃
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

# 未传 --model 且未设置 MODEL_PATH 时使用的默认模型目录（请按本机实际路径修改）
DEFAULT_MODEL_PATH = "/home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B"


def _log(msg: str) -> None:
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chatbot Server (OpenAI chat-completion API)")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument("--model", default=None, help="model path (overrides MODEL_PATH env)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], help="device")
    args = parser.parse_args()

    model_path = args.model or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    if not model_path or not os.path.isdir(model_path):
        _log("Warning: MODEL_PATH not set or not a directory. Set MODEL_PATH or use --model. Requests will return 503 until model is loaded.")
    else:
        _log(f"Model path: {model_path}")

    _log("Importing uvicorn...")
    try:
        import uvicorn
    except ImportError:
        _log("Install uvicorn and fastapi: pip install uvicorn fastapi")
        sys.exit(1)

    _log("Importing create_app...")
    try:
        from .app import create_app
    except Exception as e:
        _log(f"Import failed (e.g. torch/llaisys): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _log("Loading tokenizer and model (may take 1-2 minutes)...")
    try:
        app = create_app(model_path=model_path, device=args.device)
    except Exception as e:
        _log(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _log("Model ready. Starting server...")
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        _log(f"Server exited: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("\nStopped by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        _log(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
