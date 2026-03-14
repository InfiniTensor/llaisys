# LLAISYS 课程作业与项目复现流程

## 1. 适用范围

本文档对应当前仓库的完整课程交付，覆盖：

- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化
- Project #2：第二平台 MetaX/MACA
- Project #3：聊天服务

为便于复现，本文档按两类环境组织：

- 本地 CPU 开发环境：用于 Assignment #1/#2/#3 与 Project #1/#3
- 沐曦 MetaX 机器：用于 Project #2

## 2. 本地 CPU 路径复现

### 2.1 环境准备

建议准备：

- Python 3.10+ 
- xmake
- 本地 Qwen2 模型目录，例如 `models/DeepSeek-R1-Distill-Qwen-1.5B`

如果当前机器没有 `xmake`，可先安装：

```bash
apt-get update
apt-get install -y xmake
```

如果系统源里没有 `xmake` 包，需要改用 xmake 官方安装方式；只要最终 `xmake --version` 可用即可。

Python 侧至少需要：

```bash
python -m pip install transformers huggingface_hub accelerate fastapi uvicorn
```

### 2.2 CPU-only 构建

先显式关闭其他设备后端，避免沿用旧构建配置：

```bash
xmake f --nv-gpu=n --metax-gpu=n -cv
xmake -r
```

### 2.3 Assignment #1 / #2 基线验证

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
```

### 2.4 Assignment #3 / Project #1 推理验证

推荐直接使用本地模型目录：

```bash
python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt hi --max_steps 1
```

预期结果：

- Hugging Face 与 LLAISYS 的 token 序列严格一致
- 输出末尾打印 `Test passed!`

### 2.5 Project #3 聊天服务验证

启动服务：

```bash
PYTHONPATH=python python -m llaisys.chat.server --model models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --host 127.0.0.1 --port 8011
```

另开一个终端，先测健康检查：

```bash
curl --noproxy '*' -s http://127.0.0.1:8011/health
```

再测一次非流式聊天：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":8}'
```

预期结果：

- `/health` 返回 `status: ok`
- `POST /v1/chat/completions` 返回合法 JSON 响应

## 3. 沐曦 MetaX 路径复现

### 3.1 环境检查

在仓库根目录执行：

```bash
mx-smi
mxcc --version
python --version
python -c "import torch; print(torch.__version__); ok = torch.cuda.is_available(); print(ok); print(torch.cuda.get_device_name(0) if ok else 'no visible metax device')"
echo "$LD_LIBRARY_PATH"
ls /opt/maca/include | head
ls /opt/maca/lib | head
ls /opt/mxdriver/lib | head
```

参考环境：

- `mx-smi 2.2.9`
- `MetaX C500`
- `MACA 3.2.1.10`
- 驱动 `3.0.11`
- `mxcc version 1.0.0`
- Python `3.10.10`
- PyTorch `2.6.0+metax3.2.1.3`

如果 `LD_LIBRARY_PATH` 里没有 MetaX 运行库，建议补上：

```bash
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
```

### 3.2 MetaX 构建

如果以 root 身份运行，所有 `xmake` 命令都需要带 `XMAKE_ROOT=y`：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install
```

### 3.3 MetaX runtime / ops

```bash
python test/test_runtime.py --device metax
python test/test_ops.py --device metax
```

### 3.4 MetaX infer

推荐优先使用本地 Qwen2 模型目录：

```bash
python test/test_infer.py --device metax --test --model /path/to/local/qwen2_model --prompt hi --max_steps 1
```

如果当前机器网络可用，也可以使用公开的小模型：

```bash
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

预期结果：

- Hugging Face 与 LLAISYS 的 token 序列严格一致
- 输出末尾打印 `Test passed!`

## 4. 常见问题

### 4.1 `xmake` 提示 root 用户危险并退出

补上：

```bash
XMAKE_ROOT=y
```

### 4.2 CPU 推理时意外触发其他设备后端

请先显式执行：

```bash
xmake f --nv-gpu=n --metax-gpu=n -cv
```

再重新构建 CPU-only 版本。

### 4.3 本地服务请求被代理转发

当前云环境常常预置 `HTTP_PROXY`。访问本地聊天服务时，建议使用：

```bash
curl --noproxy '*'
```

### 4.4 MetaX 设备在受限环境里不可见

MetaX 测试必须跑在真实沐曦机器上。  
如果设备节点、驱动或运行库不可见，`mcGetDeviceCount` 可能会失败，这不是仓库逻辑错误。
