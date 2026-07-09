#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/4] 安装 Python 依赖"
python -m pip install --upgrade pip
python -m pip install -e "${ROOT_DIR}/python"

echo "[2/4] 配置并编译 LLAISYS"
pushd "${ROOT_DIR}" >/dev/null
xmake f --nv-gpu=y -cv
xmake -r
popd >/dev/null

if [[ $# -ge 1 ]]; then
  MODEL_ID="$1"
  MODEL_DIR="${2:-${ROOT_DIR}/models/$(basename "${MODEL_ID}")}"
  echo "[3/4] 下载模型 ${MODEL_ID} -> ${MODEL_DIR}"
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)"
else
  echo "[3/4] 跳过模型下载（如需下载，可执行：scripts/setup_pai_nvidia.sh <model_id> [model_dir]）"
fi

echo "[4/4] 建议的验证命令"
cat <<'EOF'
python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/test_ops.py --device cpu
python test/test_runtime.py --device nvidia
python test/test_ops.py --device nvidia

# Qwen2 一致性测试
python test/test_infer.py --device cpu --test --model /path/to/qwen2
python test/test_infer.py --device nvidia --test --model /path/to/qwen2

# TinyLlama 一致性测试
python test/test_infer.py --device cpu --test --model /path/to/tinyllama --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
python test/test_infer.py --device nvidia --test --model /path/to/tinyllama --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 启动聊天服务
llaisys-chat-server --model /path/to/model --device nvidia --host 0.0.0.0 --port 8000
llaisys-chat-cli --base-url http://127.0.0.1:8000 --stream
EOF
