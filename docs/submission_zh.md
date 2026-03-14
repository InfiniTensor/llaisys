# LLAISYS 作业提交总览

## 1. 当前提交范围

本次提交按一份完整课程交付来组织，覆盖：

- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化
- Project #2：第二平台 MetaX/MACA
- Project #3：聊天服务

其中：

- Assignment #1/#2/#3 与 Project #1/#3 主要在本地 CPU 开发环境完成实现与验证
- Project #2 在真实沐曦机器上完成 MetaX/MACA 实机验证

## 2. 验证环境

### 2.1 本地 CPU 开发环境

- Python：`3.12.3`
- xmake：`v3.0.7+20260308`
- 本地模型目录：`models/DeepSeek-R1-Distill-Qwen-1.5B`

### 2.2 沐曦 MetaX 验证环境

- GPU：`MetaX C500`
- `mx-smi`：`2.2.9`
- `MACA`：`3.2.1.10`
- 驱动：`3.0.11`
- 编译器：`mxcc 1.0.0`
- Python：`3.10.10`
- PyTorch：`2.6.0+metax3.2.1.3`
- xmake：`v2.8.7+20240401`

## 3. 已完成验证

### 3.1 本地 CPU 基线

```bash
xmake f --nv-gpu=n --metax-gpu=n -cv
xmake -r

python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt hi --max_steps 1
```

### 3.2 聊天服务最小验证

```bash
PYTHONPATH=python python -m llaisys.chat.server --model models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --host 127.0.0.1 --port 8011
curl --noproxy '*' -s http://127.0.0.1:8011/health
curl --noproxy '*' -s -X POST http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":8}'
```

### 3.3 MetaX 主链路

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install

python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

## 4. 关键说明

- 为保持 PR 干净，本次提交只包含实现代码与正式提交文档；本地学习材料与外部 PDF 均未纳入仓库
- Assignment #1/#2/#3 与 Project #1/#3 以本地 CPU 路径验证为主
- Project #2 的 MetaX 结论来自真实沐曦机器
- MetaX 在 C/C++ SDK 层不是 CUDA drop-in 兼容平台，因此后端采用独立适配
- 当前推理验证聚焦 `Qwen2`
- 当前机器没有 NVIDIA 硬件，因此没有新增 `--device nvidia` 的实机回归数据

## 5. 提交材料入口

- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
- PR 文案：[`pr_zh.md`](pr_zh.md)

以上 3 份文档配合当前代码改动与实际 GitHub PR，可覆盖课程提交需要的核心内容：

- `report_zh.md`：完整实现说明与验证结论
- `reproduce_zh.md`：分环境复现流程
- `pr_zh.md`：可直接提交的 GitHub PR 标题与正文
