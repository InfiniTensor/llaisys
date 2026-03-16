# 项目三报告：基于 FastAPI 的大模型推理 API 服务与 Web 界面集成

## 一、 概述

本项目是 `LLAiSYS` 大语言模型推理框架的顶层工程应用。在项目一和项目二实现了底层 Tensor 内存管理与高效 CUDA 推理算子的基础上，本项目旨在打破底层 C++/CUDA 代码与终端用户之间的交互壁垒，将其构建为一个现代化的 AI Web 服务。

本项目基于 Python 的高性能异步 Web 框架 `FastAPI`，设计并实现了一套完全兼容 OpenAI 官方标准定义（`/v1/chat/completions`）的 RESTful API 接口。同时，利用 Server-Sent Events (SSE) 技术实现了模型推理过程的实时流式（Streaming）输出，并成功将我们的本地推理引擎与业界主流的开源前端 UI `ChatGPT-Next-Web`（NextChat）无缝集成，最终交付了一个端到端的完整大语言模型对话系统。

## 二、 运行环境

- **硬件平台**：NVIDIA GPU / CPU
- **操作系统**：Linux / Windows 跨平台支持
- **核心语言**：Python 3.10+, TypeScript (前端)
- **后端依赖库**：`fastapi`, `uvicorn`, `pydantic`, `sse-starlette`, `transformers`, `huggingface_hub`
- **前端系统**：`ChatGPT-Next-Web` (基于 Next.js 与 React 构建)
- **测试模型**：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (Qwen2 架构)

## 三、 核心架构与具体实现

### 3.1 OpenAI 兼容协议设计与数据模型

为了使我们的推理框架能够直接被市面上成熟的第三方 AI 客户端（如 NextChat、LobeChat 等）调用，服务端必须严格遵守 OpenAI 的接口规范。本项目使用 `pydantic` 构建了严谨的数据校验模型。

**核心数据结构实现：**

```Python
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 50
```

### 3.2 Prompt 模板化与 Tokenizer 接入

大语言模型的对话能力高度依赖于特定的特殊占位符（如 `<|im_start|>`、`<|im_end|>` 等）。本项目通过引入 Hugging Face 的 `transformers.AutoTokenizer`，并调用其 `apply_chat_template` 机制，将前端传入的 JSON 格式历史对话上下文，自动拼装成符合 Qwen2 模型底层训练格式的单行 Prompt 字符串，随后进行 Encode 转换为 `input_ids` 序列，送入底层的 C++ `llaisys.models.Qwen2` 引擎进行推理。

### 3.3 Server-Sent Events (SSE) 流式输出机制

大模型自回归生成的特性决定了如果采用同步阻塞返回，用户将面临巨大的等待延迟。本项目基于 FastAPI 的 `StreamingResponse` 实现了实时流式传输。

**流式生成核心逻辑：**

```Python
async def generate_stream():
    # 执行模型自回归生成新 Token
    for token_id in new_tokens:
        if token_id == tokenizer.eos_token_id:
            break
        # 解码单步 Token 为文本
        word = tokenizer.decode([token_id], skip_special_tokens=True)
        # 组装符合 OpenAI 规范的 SSE 数据块
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": word}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.02) # 模拟平滑输出的打字机效果
        
    yield "data: [DONE]\n\n"
```

通过上述生成器，底层的每一次 C++ `llaisysQwen2ModelInfer` 调用结果都能被瞬间推送到前端，极大地提升了系统的人机交互体验。

### 3.4 跨域资源共享 (CORS) 与网络安全配置

由于前端 Web 页面（如跑在 `localhost:3000` 的 NextChat）与 FastAPI 后端服务（运行于 `0.0.0.0:8199`）通常处于不同的端口或域名下，浏览器会触发跨域安全拦截（发送 `OPTIONS` 预检请求）。本项目通过配置 FastAPI 的 `CORSMiddleware` 中间件，全面放行了前端发起的跨域请求，彻底解决了 `405 Method Not Allowed` 的网络拦截问题。

## 四、 构建与测试

### 4.1 启动推理后端服务

本项目使用 `argparse` 暴露了启动配置参数，支持自动从 Hugging Face Hub 下载并缓存模型权重，使用 `uvicorn` 承载 ASGI HTTP 服务。

```Bash
python python/server.py --device nvidia --port 8199
```

启动成功后，终端将输出 `[READY] Server starting on http://0.0.0.0:8199`。

**终端图像：**

![image-20260316235034869](assets/image-20260316235034869.png)

### 4.2 接入 ChatGPT-Next-Web 前端进行验证

本项目选择业界主流的开源大模型前端 `ChatGPT-Next-Web`（又称 NextChat）作为可视化交互界面。通过该 UI 验证了推理后端对 OpenAI 标准协议的兼容性及流式传输的稳定性。具体操作流程如下：

#### 1. 前端环境部署

- **访问方式**：可直接访问 NextChat 的 Web 托管版本，例如：https://app.nextchat.club/#/chat，或通过 Docker 分布在本地 `3000` 端口。
- **通信准备**：确保前端所在的浏览器环境能够访问到 `server.py` 运行的后端 IP 地址及端口（如 `http://127.0.0.1:8199`）。

#### 2. 自定义接口配置 (Settings)

进入 NextChat 左下角的“设置”面板，进行以下关键参数的绑定：

- **模型服务商 (Model Provider)**：选择 `OpenAI`。
- **自定义接口地址 (Endpoint URL)**：填写我们的 FastAPI 服务端地址 `http://127.0.0.1:8199`。
  - *注：NextChat 会自动在末尾拼接 `/v1/chat/completions` 路径。*
- **API Key**：由于本地测试未开启鉴权，此处可随意填写（如 `sk-llaisys`），以绕过前端的非空校验。
- **自定义模型 (Custom Models)**：在自定义模型列表中输入 `qwen2` 并添加。

#### 3. 核心交互验证

- **模型切换**：在聊天窗口顶部下拉菜单中选中刚才添加的 `qwen2` 模型。
- **流式生成测试**：在输入框发送长文本问题（如“请写一段 200 字左右关于人工智能的介绍”）。
- **响应观察**：
  - **SSE 验证**：观察文字是否以“打字机”效果逐个跳出。这证明了后端的 `StreamingResponse` 正在实时推送 Token，而非等待生成结束后一次性返回。
  - **标题自动总结**：NextChat 会在对话开始后自动发送一个 `stream: False` 的后台请求。验证左侧历史记录栏是否成功根据模型回复生成了简短标题，这证明了后端对非流式 JSON 响应格式的正确处理。
  - **CORS 预检**：通过浏览器开发者工具（F12）观察，确认浏览器发出的 `OPTIONS` 预检请求已被 FastAPI 成功拦截并允许跨域，从而保证了 `POST` 请求的顺利下发。

#### 4. 交互原理示意

`用户输入` $\rightarrow$ `NextChat UI (JSON 封装)` $\rightarrow$ `HTTP POST 请求` $\rightarrow$ `FastAPI 后端 (路由解析)` $\rightarrow$ `LLAiSYS C++ 引擎` $\rightarrow$ `GPU 并行计算` $\rightarrow$ `SSE 流式写回` $\rightarrow$ `前端 Markdown 渲染`。

## 五、 结论

本项目成功为 `LLAiSYS` 框架构建了应用层的服务端基础设施。通过实现标准化的 OpenAI API 协议、跨域中间件以及 SSE 流式传输机制，不仅使底层的 C++ 算子引擎具备了作为云端微服务独立运行的能力，还实现了与业界主流 Web UI 的零成本集成。至此，本系统已具备了从底层内存分配到前端可视化交互的完整大模型基础设施能力。