# LLAISYS Chatbot Server

OpenAI chat-completion 风格的 HTTP 服务，单用户、支持流式 (SSE)。

## 依赖

```bash
pip install fastapi uvicorn
```

（若已安装 `transformers` 用于分词器则无需额外依赖。）

## 启动

指定模型目录（与 `test_infer.py` 使用的 Qwen2 模型一致）：

```bash
# 方式一：环境变量
set MODEL_PATH=C:\path\to\DeepSeek-R1-Distill-Qwen-1.5B
python -m llaisys_py.server

# 方式二：命令行参数
python -m llaisys_py.server --model "C:\path\to\DeepSeek-R1-Distill-Qwen-1.5B" --port 8000
```

可选参数：`--host`, `--port`, `--device`（cpu / nvidia）。

## 接口

- `GET /health`：健康检查，返回是否已加载模型。
- `POST /v1/chat/completions`：与 OpenAI 兼容的对话补全。

请求体示例：

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 128,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 50,
  "stream": false,
  "seed": null
}
```

- `stream: true` 时返回 SSE 流式响应。

## 交互式聊天 UI（多轮对话）

### 方式一：Web 页面

服务启动后，在浏览器打开：

- **http://127.0.0.1:8000/chat**

即可使用网页聊天：输入框输入内容，点击发送或回车，对话历史会保留在页面上，支持连续多轮对话。

### 方式二：命令行（CLI）

先启动服务，再在**另一个终端**运行：

```bash
pip install requests
python -m llaisys_py.server.chat_cli
```

默认连到 `http://127.0.0.1:8000`。输入一句话回车发送，收到回复后继续输入下一句；输入 `quit` 或 `q` 退出。

可选参数：`--base-url`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`。

## 从项目根目录运行

若未 `pip install -e python`，需把 `python` 加入 PYTHONPATH：

```bash
set PYTHONPATH=python
python -m llaisys_py.server --model "C:\path\to\model"
```

## 答非所问时

若模型经常跑题、只回复客套话或固定“我是 DeepSeek-R1…”自我介绍，可尝试：

1. **默认不加系统提示**：服务端已改为不注入任何 system 内容（避免触发模型自报家门）。若需要自定义系统人设，可设环境变量 `LLAISYS_SYSTEM_PROMPT="你的说明"` 再启动。
2. **降低 temperature**：请求里传 `"temperature": 0.3` 或 `0.5`，回答会更聚焦。
3. **新开会话**：在 `/sessions` 里新建对话再问，避免被之前的长回复干扰。
4. **看实际发给模型的 prompt**：启动前设 `LLAISYS_DEBUG=1`，控制台会打印每轮 prompt 长度和末尾 300 字符，便于排查。
5. **模型能力**：DeepSeek-R1-Distill-Qwen-1.5B 为 1.5B 小模型，知识型问题可能表现有限，可换更大模型或仅作演示。
