# GitHub PR 文案

## 标题

`feat: complete LLAISYS assignments 1 2 3 and projects 1 2 3 6`

## 正文

本 PR 完成 LLAISYS 的以下课程内容，并补齐中文提交文档：

- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化
- Project #2：第二平台 MetaX/MACA
- Project #3：聊天服务
- Project #6：支持新模型

### 主要改动

- 完成 Tensor 基础能力，包括 `load`、`isContiguous`、`view`、`permute`、`slice`
- 完成 CPU 侧关键算子：`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`
- 完成 Qwen2 推理链路、权重装载与 token 级对照验证
- 基于 OpenMP 完成 CPU 热点算子优化
- 新增独立 `METAX` 设备类型与 `--metax-gpu=y` 构建开关
- 完成 MetaX/MACA runtime 与关键算子路径接入，`linear` 对接 `mcblasGemmEx`
- 实现聊天服务与流式返回接口
- 新增 `Llama/TinyLlama` 路径的 C++/Python 包装与基于 `config.json` 的模型类型自动分发
- 补齐提交总览、实现报告与复现流程
- 本 PR 只包含实现代码与正式提交文档，本地学习材料与外部 PDF 未纳入提交

### 已验证命令

本地 CPU 路径：

```bash
xmake f --nv-gpu=n --metax-gpu=n -cv
xmake -r

python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt hi --max_steps 1
```

聊天服务最小验证：

```bash
PYTHONPATH=python python -m llaisys.chat.server --model models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --host 127.0.0.1 --port 8011
curl --noproxy '*' -s http://127.0.0.1:8011/health
curl --noproxy '*' -s -X POST http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":8}'
```

新模型验证入口：

```bash
python test/test_infer.py --device cpu --test --model /path/to/local/llama_or_tinyllama_model --prompt hi --max_steps 1
```

MetaX 路径：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install

python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

### 说明

- Assignment #1/#2/#3 与 Project #1/#3/#6 主要在本地 CPU 环境完成验证
- Project #2 在真实沐曦 `MetaX C500` 机器上完成实机验证
- MetaX 在 C/C++ SDK 层不是 CUDA drop-in 兼容平台，因此后端采用独立适配
- 当前推理验证以 `Qwen2` 为主；Project #6 提供 `Llama/TinyLlama` 新模型接入与本地模型目录验证入口
- 当前机器没有 NVIDIA 硬件，因此本次没有新增 `--device nvidia` 的实机回归数据
- 根目录外部 PDF 保持未跟踪状态，不提交进仓库

### 提交文档

- 提交总览：[`submission_zh.md`](submission_zh.md)
- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
