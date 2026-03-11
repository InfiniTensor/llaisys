# LLAISYS 复现流程（中文，可直接提交）

## 1. 环境
- 平台：PAI DSW NVIDIA 实例
- CUDA：`/usr/local/cuda-12.8`
- Python：3.10+ 均可，本次实测环境为 Python 3.12

## 2. 一键初始化
在仓库根目录执行：

```bash
bash scripts/setup_pai_nvidia.sh
```

如果需要顺手下载模型：

```bash
bash scripts/setup_pai_nvidia.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B models/DeepSeek-R1-Distill-Qwen-1.5B
bash scripts/setup_pai_nvidia.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/TinyLlama-1.1B-Chat-v1.0
```

## 3. 构建
```bash
xmake -r
PATH=/usr/local/cuda-12.8/bin:$PATH xmake f --nv-gpu=y -cv
xmake -r
```

## 4. 基础测试

### 4.1 CPU
```bash
python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/test_ops.py --device cpu
```

### 4.2 NVIDIA
```bash
python test/test_runtime.py --device nvidia
python test/test_ops.py --device nvidia
```

## 5. 推理一致性

### 5.1 Qwen2
```bash
python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B
python test/test_infer.py --device nvidia --test --model models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 5.2 TinyLlama
```bash
python test/test_infer.py --device nvidia --test --model models/TinyLlama-1.1B-Chat-v1.0
```

补充说明：
- `--test` 模式下脚本会强制让 Hugging Face 侧使用 `float32`。
- 这样做是为了让 GPU 严格一致性校验与当前 `LLAISYS` 的 `float32` 后端保持同一数值精度口径，避免 `bf16` 带来的 token 提前分叉。
- 如果你要在 CPU 上继续做 TinyLlama 全量严格校验，也可以执行：

```bash
python test/test_infer.py --device cpu --test --model models/TinyLlama-1.1B-Chat-v1.0 --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 6. 真实生成验证
```bash
python test/test_infer.py --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt "请用中文介绍一下你自己。"
python test/test_infer.py --device nvidia --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "Please introduce yourself in one sentence." --max_steps 32
```

说明：真实生成默认走采样，因此 HF 与 LLAISYS 文本不要求逐 token 完全一致，只要推理链路正常即可。

## 7. 聊天演示

### 7.1 启动服务
```bash
llaisys-chat-server --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --host 127.0.0.1 --port 8000
```

或者：

```bash
llaisys-chat-server --model models/TinyLlama-1.1B-Chat-v1.0 --device nvidia --host 127.0.0.1 --port 8000
```

### 7.2 启动 CLI
```bash
llaisys-chat-cli --base-url http://127.0.0.1:8000 --stream
```

### 7.3 CLI 常用命令
- `/clear`：清空当前会话
- `/exit` 或 `/quit`：退出

说明：CLI 已自动对 `localhost/127.0.0.1/::1` 绕过代理，不需要手工处理当前云环境中的 `HTTP_PROXY`。

## 8. Benchmark
```bash
python scripts/benchmark_llaisys.py --device cpu --repeat 20
python scripts/benchmark_llaisys.py --device nvidia --repeat 20
python scripts/benchmark_llaisys.py --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --max-new-tokens 64
```

## 9. 提交建议
- 报告直接使用 [report_zh.md](/home/saber/llaisys/docs/report_zh.md)
- PR 正文直接使用 [pr_zh.md](/home/saber/llaisys/docs/pr_zh.md)
- 第二平台设计说明使用 [metax_design_zh.md](/home/saber/llaisys/docs/metax_design_zh.md)
- 复试准备材料使用 [interview_qa_zh.md](/home/saber/llaisys/docs/interview_qa_zh.md)
