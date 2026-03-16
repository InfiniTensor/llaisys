#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Hugging Face 下载 LLAISYS 使用的模型（默认 DeepSeek-R1-Distill-Qwen-1.5B）。

用法:
  python scripts/download_model.py
  python scripts/download_model.py --dir /自定义/保存目录
  python scripts/download_model.py --repo 其他组织/其他模型名

依赖: 在虚拟环境中安装 huggingface_hub，例如:
  python3 -m venv .venv && .venv/bin/pip install huggingface_hub
  然后运行: .venv/bin/python scripts/download_model.py
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载模型到本地目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_model.py
  python scripts/download_model.py --dir ./my_models/DeepSeek-R1-Distill-Qwen-1.5B
  python scripts/download_model.py --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dir ./models
        """,
    )
    parser.add_argument(
        "--repo",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Hugging Face 仓库 ID，默认: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="保存目录。不指定时使用: <项目根>/models/<仓库名>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传（已存在的文件会跳过）",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装 huggingface_hub:", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        print("或在项目 venv 中: .venv/bin/pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    repo_id = args.repo
    if args.dir:
        local_dir = os.path.abspath(os.path.expanduser(args.dir))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(script_dir)
        # 用仓库名最后一段作为子目录名，如 DeepSeek-R1-Distill-Qwen-1.5B
        repo_name = repo_id.split("/")[-1]
        local_dir = os.path.join(root, "models", repo_name)

    parent = os.path.dirname(local_dir)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"仓库: {repo_id}")
    print(f"目标: {local_dir}")
    if args.resume:
        print("模式: 断点续传（已存在文件将跳过）")
    print()

    try:
        path = snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=args.resume,
        )
        print(f"下载完成: {path}")
        print(f"\n启动 Project3 服务示例:")
        print(f"  python -m llaisys_py.server --model {path}")
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
