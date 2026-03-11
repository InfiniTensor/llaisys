# PR 正文（中文，可直接提交）

## 标题
`feat: 完成 LLAISYS 项目 1/2/3/6，补充 TinyLlama 与中文交付文档`

## 变更摘要
- 完成 CPU 算子并行优化，重点优化 `linear`
- 完成 NVIDIA runtime 与 8 个推理关键算子的 CUDA 路径
- 完成随机采样、OpenAI 风格聊天服务、SSE 流式 CLI
- 新增 `Llama` / `TinyLlama` 模型支持与模型工厂自动识别
- 补充中文报告、复现说明、Metax 设计稿与复试问答文档

## 完成范围

### 项目 1
- 在 CPU 构建中启用 OpenMP
- 优化 `linear`、`embedding`、`rms_norm`、`rope`、`self_attention`、`swiglu`

### 项目 2
- 完成 NVIDIA runtime API
- 完成 `add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu` 的 NVIDIA 路径
- 第二平台选择 `Metax`，本 PR 仅提交设计稿，不提交可执行实现

### 项目 3
- 新增随机采样
- 新增 `GET /health` 与 `POST /v1/chat/completions`
- 支持非流式与 SSE 流式返回
- 新增单用户串行 CLI

### 项目 6
- 新增 `Llama` C API、ctypes 绑定、Python 封装
- 支持 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 模型工厂按 `config.json` 自动识别 `qwen2` / `llama`

## 实测验证

### 基础测试
- [x] `python test/test_runtime.py --device cpu`
- [x] `python test/test_tensor.py`
- [x] `python test/test_ops.py --device cpu`
- [x] `python test/test_runtime.py --device nvidia`
- [x] `python test/test_ops.py --device nvidia`

### Qwen2
- [x] `python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B`
- [x] `python test/test_infer.py --device nvidia --test --model models/DeepSeek-R1-Distill-Qwen-1.5B`
- [x] `python test/test_infer.py --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt "请用中文介绍一下你自己。"`
- [x] Qwen2 聊天服务 `/health`、非流式、流式 CLI 冒烟通过

### TinyLlama
- [x] `python test/test_infer.py --device nvidia --test --model models/TinyLlama-1.1B-Chat-v1.0`
- [x] HF `float32 GPU` 与 LLAISYS `NVIDIA` 前 32 个贪心 token 完全一致
- [x] LLAISYS `CPU` 与 `NVIDIA` 前 32 个贪心 token 完全一致
- [x] HF `CPU float32` 与 LLAISYS `CPU` 前 8 个贪心 token 完全一致
- [x] `python test/test_infer.py --device nvidia --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "Please introduce yourself in one sentence." --max_steps 32`
- [x] TinyLlama 聊天服务 `POST /v1/chat/completions` 冒烟通过

## 额外说明
- TinyLlama 在 GPU 严格测试中，HF 必须在 `--test` 模式下强制使用 `float32`，否则会因为 `bf16/f32` 精度口径不一致造成 token 提前分叉。
- 当前聊天 CLI 已对 `localhost/127.0.0.1/::1` 自动绕过代理，避免云环境默认 `HTTP_PROXY` 导致本地请求 `502`。
- TinyLlama 的 CPU 全量 128-step 严格校验在当前环境较慢，因此本 PR 同时保留了完整 NVIDIA 严格校验结果与 CPU 侧局部严格校验结果。
- 第二平台 `Metax` 以设计稿形式提交，见 [metax_design_zh.md](/home/saber/llaisys/docs/metax_design_zh.md)。

## 交付文档
- 实现报告：[report_zh.md](/home/saber/llaisys/docs/report_zh.md)
- 复现流程：[reproduce_zh.md](/home/saber/llaisys/docs/reproduce_zh.md)
- 复试问答：[interview_qa_zh.md](/home/saber/llaisys/docs/interview_qa_zh.md)
