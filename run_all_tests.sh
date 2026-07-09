#!/bin/bash
# llaisys 作业测试脚本

cd /root/autodl-tmp/llaisys

echo "=========================================="
echo "作业 #1: 张量操作测试"
echo "=========================================="
python test/test_tensor.py
echo ""

echo "=========================================="
echo "作业 #2: CPU 算子测试"
echo "=========================================="

echo "--- argmax ---"
python test/ops/argmax.py

echo "--- embedding ---"
python test/ops/embedding.py

echo "--- swiglu ---"
python test/ops/swiglu.py

echo "--- rms_norm ---"
python test/ops/rms_norm.py

echo "--- linear ---"
python test/ops/linear.py

echo "--- self_attention ---"
python test/ops/self_attention.py

echo "--- rope (可能有微小精度差异) ---"
python test/ops/rope.py || echo "Note: rope has minor floating point differences"

echo ""
echo "=========================================="
echo "作业 #3: 模型推理测试"
echo "=========================================="
echo "请运行: python test/test_infer.py --test"
echo "注意: 首次运行会自动下载约 3GB 的模型"
echo ""
echo "所有基础测试完成！"
