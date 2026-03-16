#!/bin/bash
# 不依赖「pip install -e ./python/」也能启动服务：用 PYTHONPATH 找到 llaisys 包。
# 前提：已执行 xmake && xmake install（动态库在 python/llaisys_py/libllaisys/ 下）。
#
# 用法: ./scripts/run_server.sh [模型目录]
# 示例: ./scripts/run_server.sh /home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
PORT="${2:-8000}"

if [ -z "$MODEL" ]; then
  echo "用法: $0 <模型目录> [端口]"
  echo "示例: $0 $ROOT/DeepSeek-R1-Distill-Qwen-1___5B 8000"
  exit 1
fi

if [ ! -d "$MODEL" ]; then
  echo "错误: 模型目录不存在: $MODEL"
  exit 1
fi

# 用项目里的 python 包，不要求 pip install
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

# 优先用 venv 的 Python（里面可能有 torch、transformers 等）
if [ -x "${ROOT}/.venv/bin/python" ]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python3"
fi

echo "PYTHONPATH=$PYTHONPATH"
echo "Python: $PY"
echo "模型: $MODEL"
echo "端口: $PORT"
exec "$PY" -m llaisys_py.server --model "$MODEL" --port "$PORT"
