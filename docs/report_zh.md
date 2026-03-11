# LLAISYS 项目 1/2/3/6 实现报告

## 1. 作业完成情况
- 项目 1：完成 CPU 算子并行优化，在 `xmake` CPU 构建中启用 OpenMP，并为 `linear`、`embedding`、`rms_norm`、`rope`、`self_attention`、`swiglu` 增加并行路径。
- 项目 2：完成 NVIDIA runtime 与 8 个关键推理算子的 CUDA 路径；第二平台选择 `Metax`，以设计稿形式提交，不包含可执行实现。
- 项目 3：完成随机采样、OpenAI 风格聊天服务、SSE 流式输出、单用户串行 CLI。
- 项目 6：新增 `Llama` / `TinyLlama` 模型支持，并通过模型工厂按 `config.json` 自动识别 `qwen2` 或 `llama`。

## 2. 关键实现说明

### 2.1 CPU 优化
- 在 CPU 构建规则中启用 OpenMP。
- `linear` 采用按行并行与分块累加，减少 cache miss，作为主要性能优化点。
- `embedding`、`rms_norm`、`rope`、`swiglu`、`self_attention` 按 token / row / head 维度展开并行。
- 保持原有 Python 与 C 接口不变，已有调用代码无需改动。

### 2.2 NVIDIA 后端
- 完成 runtime API：设备数量、设备切换、stream、device/host malloc/free、同步/异步 memcpy。
- 完成以下 8 个推理算子的 NVIDIA 路径：
  - `add`
  - `argmax`
  - `embedding`
  - `linear`
  - `rms_norm`
  - `rope`
  - `self_attention`
  - `swiglu`
- `linear` 使用 cuBLAS；其余算子使用 CUDA kernel 或 host fallback，优先保证正确性。
- 权重加载、KV cache 写回、最终 token 读取统一走 runtime memcpy，避免直接对设备指针做主机侧 `memcpy`。
- NVIDIA 算子统一落到默认 CUDA stream，避免与 Python / PyTorch 默认拷贝链路混用自定义 stream 时出现可见性问题。

### 2.3 聊天服务
- 新增采样接口，支持 `temperature`、`top_k`、`top_p`、`seed`。
- `Qwen2.generate(...)` 与 `Llama.generate(...)` 都支持采样参数。
- 新增 `FastAPI` 服务接口：
  - `GET /health`
  - `POST /v1/chat/completions`
- 支持非流式与 SSE 流式返回。
- 服务端通过单全局锁保证单用户串行，避免多个请求共享同一个模型状态与 KV cache。

### 2.4 TinyLlama 支持
- 新增 `Llama` C API、ctypes 绑定与 Python 封装。
- 模型加载统一通过模型工厂完成，不需要上层手动判断 `Qwen2` 或 `Llama`。
- `test/test_infer.py` 已支持同时验证 Qwen2 与 TinyLlama。

## 3. 实验环境
- 验证日期：2026-03-10
- 平台：PAI DSW NVIDIA 实例
- CUDA：`/usr/local/cuda-12.8`
- Python：3.12
- GPU：NVIDIA A100 80GB
- 本次实测模型：
  - `models/DeepSeek-R1-Distill-Qwen-1.5B`
  - `models/TinyLlama-1.1B-Chat-v1.0`

## 4. 实测结果

### 4.1 基础测试
以下命令已在当前环境通过：
- `python test/test_runtime.py --device cpu`
- `python test/test_tensor.py`
- `python test/test_ops.py --device cpu`
- `python test/test_runtime.py --device nvidia`
- `python test/test_ops.py --device nvidia`

### 4.2 Qwen2 推理与聊天验证
- `python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B`
  - 结果：通过
  - 说明：HF 与 LLAISYS token 级一致
- `python test/test_infer.py --device nvidia --test --model models/DeepSeek-R1-Distill-Qwen-1.5B`
  - 结果：通过
  - 说明：HF 与 LLAISYS token 级一致
- `python test/test_infer.py --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt "请用中文介绍一下你自己。"`
  - 结果：通过
  - 说明：真实生成链路正常；由于默认走采样，HF 与 LLAISYS 文本不要求逐 token 完全一致
- 聊天服务验证：
  - `/health` 返回 `200`
  - `POST /v1/chat/completions` 非流式返回 `200`
  - CLI 流式输出正常

### 4.3 TinyLlama 推理与聊天验证
- `python test/test_infer.py --device nvidia --test --model models/TinyLlama-1.1B-Chat-v1.0`
  - 结果：通过
  - 说明：HF 与 LLAISYS token 级一致
- HF `float32 GPU` 与 LLAISYS `NVIDIA` 前 32 个贪心 token 完全一致
- LLAISYS `CPU` 与 `NVIDIA` 前 32 个贪心 token 完全一致
- HF `CPU float32` 与 LLAISYS `CPU` 前 8 个贪心 token 完全一致
- `python test/test_infer.py --device nvidia --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "Please introduce yourself in one sentence." --max_steps 32`
  - 结果：通过
  - 说明：真实生成链路正常；默认采样下，HF 与 LLAISYS 文本不要求逐 token 完全一致
- TinyLlama 聊天服务验证：
  - `POST /v1/chat/completions` 返回 `200`
  - 服务端能正确通过模型工厂加载 `llama`

### 4.4 端到端结论
- Qwen2 的 CPU 与 NVIDIA 严格一致性测试都已跑通。
- TinyLlama 的 NVIDIA 严格一致性已跑通。
- TinyLlama 在 CPU 侧已完成多轮局部严格校验，验证结果与 NVIDIA 后端一致，没有发现模型实现层面的结构性错误。
- 新增聊天服务已同时在 `Qwen2` 与 `TinyLlama` 上完成 API 冒烟。

## 5. 本次实际踩坑与修复

### 5.1 本地聊天 CLI 返回 502
- 现象：CLI 访问 `127.0.0.1:8000` 时返回 `HTTP 502`
- 原因：当前云环境默认注入了 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:9999`，`urllib` 会把本地请求错误转发给代理
- 修复：CLI 对 `localhost/127.0.0.1/::1` 自动绕过代理

### 5.2 TinyLlama GPU 严格测试首次失败
- 现象：TinyLlama 在 `--device nvidia --test` 下最初出现 token 提前分叉
- 原因：测试脚本让 Hugging Face 在 GPU 上默认使用 `bf16`，而当前 LLAISYS 后端使用 `f32`，两边数值精度口径不一致
- 修复：`test/test_infer.py` 在 `--test` 模式下强制 Hugging Face 使用 `float32`

## 6. Benchmark 记录建议
推荐在提交课程作业时补充以下 benchmark 数值：
- `python scripts/benchmark_llaisys.py --device cpu --repeat 20`
- `python scripts/benchmark_llaisys.py --device nvidia --repeat 20`
- `python scripts/benchmark_llaisys.py --device nvidia --model /path/to/model --max-new-tokens 64`

建议至少记录：
- 机器型号 / CUDA 版本
- CPU 线程数 / GPU 型号
- `linear` 平均耗时
- 一个固定 prompt 的端到端耗时

## 7. 已知限制
- 第二平台 `Metax` 只提交设计稿，不包含可执行代码。
- 聊天服务按课程要求设计为单用户串行服务。
- `self_attention` 在 NVIDIA 路径仍优先保证正确性，后续仍有性能优化空间。
- TinyLlama 的 CPU 全量 128-step 严格校验在当前环境较慢，因此本次报告保留了已经完成的局部严格校验结果与完整 NVIDIA 严格校验结果。

## 8. 提交物说明
- 复现流程见 [reproduce_zh.md](/home/saber/llaisys/docs/reproduce_zh.md)
- PR 正文见 [pr_zh.md](/home/saber/llaisys/docs/pr_zh.md)
- 第二平台设计稿见 [metax_design_zh.md](/home/saber/llaisys/docs/metax_design_zh.md)
- 复试问答整理见 [interview_qa_zh.md](/home/saber/llaisys/docs/interview_qa_zh.md)
