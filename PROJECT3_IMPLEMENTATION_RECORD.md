# 项目三实现记录（AI Chatbot）

最后更新：2026-03-02  
项目范围：Project #3（Random Sampling + Chat Server + Interactive UI）

---

## 1. 需求完成情况

| 项目三要求 | 实现状态 | 说明 |
|---|---|---|
| 随机采样（Temperature / Top-K / Top-P） | 已完成 | C++ 模型层 + C API + Python 绑定全链路支持 |
| 聊天服务端（OpenAI 风格） | 已完成 | FastAPI 实现 `/v1/chat/completions`，支持流式 SSE |
| 交互式 UI | 已完成 | 提供 CLI（命令行）和 Web UI（浏览器）两种入口 |

---

## 2. 实现总览

### 2.1 关键文件

- C/C++ 推理与接口：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - `include/llaisys/models/qwen2.h`
  - `src/llaisys/models.cc`
- Python 绑定与模型封装：
  - `python/llaisys/libllaisys/models.py`
  - `python/llaisys/models/qwen2.py`
- 服务与交互：
  - `test/chat_server.py`
  - `test/chat_cli.py`
  - `test/chat_web.html`

### 2.2 调用链

1. 前端（CLI/Web）调用 `POST /v1/chat/completions`。  
2. 服务端将 `messages` 转成 chat template token，调用 `llaisys.models.Qwen2.generate(...)` 或 `generate_stream(...)`。  
3. Python 封装层通过 ctypes 调用：
   - greedy：`llaisysQwen2ModelInfer`
   - sampling：`llaisysQwen2ModelInferSample`
4. C++ 模型执行 forward，并在 sampling 路径使用 `top_k/top_p/temperature` 选 token。

---

## 3. 随机采样实现

### 3.1 C++ 模型层

`Model::infer` 扩展为：

```cpp
int64_t infer(int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature);
```

核心逻辑：

1. **Greedy 快路径**  
当 `top_k==1 && top_p>=1.0 && temperature==1.0` 时，走原 `argmax` 算子路径，减少开销。

2. **Sampling 路径**  
读取最后一步 logits 到 host（支持 `F32/F16/BF16`），执行：
   - 参数归一：
     - `top_k<=0` 或超过 vocab：裁剪到 vocab
     - `top_p<=0` 或 `>1`：回退为 `1.0`
     - `temperature<=0`：回退 argmax
   - `top_k` 截断（按 logits 排序）
   - `temperature` 缩放 softmax
   - `top_p` nucleus 截断（按累计概率）
   - `std::discrete_distribution` 抽样返回 token id

3. **实现位置**  
`src/models/qwen2/model.cpp` 中新增：
   - `logits_to_host_f32(...)`
   - `sample_from_logits(...)`
   - `argmax_host(...)`

### 3.2 C API 与 Python 绑定

新增 C API：

```c
int64_t llaisysQwen2ModelInferSample(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature);
```

落地文件：
- 声明：`include/llaisys/models/qwen2.h`
- 实现：`src/llaisys/models.cc`
- ctypes 注册：`python/llaisys/libllaisys/models.py`

### 3.3 Python 模型封装

`python/llaisys/models/qwen2.py` 里新增 `_infer_next(...)` 路由：

- greedy 参数：调用 `llaisysQwen2ModelInfer`
- 非 greedy 参数：调用 `llaisysQwen2ModelInferSample`

并新增：
- `generate_stream(...)`：按 token 迭代输出

---

## 4. Chat Server 实现（OpenAI 风格）

文件：`test/chat_server.py`

### 4.1 路由

- `GET /`：返回 Web UI 页面（`test/chat_web.html`）
- `GET /health`：健康检查
- `POST /v1/chat/completions`：聊天接口（兼容 OpenAI 样式）

### 4.2 请求字段（支持）

- `model`
- `messages`（role: `system/user/assistant`）
- `max_tokens`（兼容 `max_new_tokens`）
- `top_k`
- `top_p`
- `temperature`
- `stream`

### 4.3 响应行为

1. `stream=false`  
返回 `chat.completion`，包含：
- `choices[0].message.content`
- `usage.prompt_tokens/completion_tokens/total_tokens`

2. `stream=true`  
返回 SSE（`text/event-stream`），顺序为：
- 首包：assistant role
- 增量包：`delta.content`
- 结束包：`finish_reason=stop`
- usage 包（可选）
- `[DONE]`

### 4.4 单用户串行约束

`ChatEngine` 内使用 `threading.Lock` 包住生成，满足项目三“可阻塞单用户”的要求，避免并发请求互相污染状态。

### 4.5 兼容与稳健性处理

- 优先导入仓库本地 `python/llaisys`，避免误用环境中的旧版本包。
- 若运行环境中 `Qwen2` 暂无 `generate_stream`，服务端自动回退为“单块流式”输出，接口仍可用。

---

## 5. 交互端实现

### 5.1 CLI（`test/chat_cli.py`）

能力：
- 持续对话（维护 `history`）
- 系统提示词（`--system`）
- 参数透传（`--max-tokens/--top-k/--top-p/--temperature`）
- 支持 `--stream`
- 命令：
  - `/reset` 清空会话
  - `/exit` 或 `/quit` 退出

### 5.2 Web UI（`test/chat_web.html`）

能力：
- 可视化聊天窗口 + 设置面板
- 参数调节：`model/system/max_tokens/top_k/top_p/temperature`
- 流式开关
- `Stop` 中断当前请求（AbortController）
- `Reset Conversation` 清空会话
- 响应式布局（桌面/移动端）

---

## 6. 验证记录

### 6.1 脚本语法与启动

```bash
python -m py_compile test/chat_server.py test/chat_cli.py
python test/chat_server.py --help
python test/chat_cli.py --help
```

### 6.2 API Smoke Test（本地）

验证项：
1. `GET /` 返回 Web UI HTML。  
2. `POST /v1/chat/completions` 非流式返回 `object=chat.completion`。  
3. `POST /v1/chat/completions` 流式返回多段 chunk + `[DONE]`。  

验证现象（样例）：
- 非流式可返回完整 answer 与 usage。
- 流式在同一请求中可观察到连续 token 增量输出（非空 chunk）。

### 6.3 推理一致性

```bash
python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test
```

结果：`Test passed`（确定性配置下 token 对齐）。

---

## 7. 已知限制

1. 当前 sampling 在 C++ 侧为“logits 拉回 host 后抽样”，每 token 有 D2H 开销；高吞吐场景可继续做设备侧采样。  
2. 服务端按项目三要求采用“单用户串行”模型，不支持多用户并发调度。  
3. 未实现多会话管理、历史编辑重生成、KV cache 前缀复用池（属于项目三可选项/项目四方向）。  

---

## 8. 运行说明（快速开始）

### 8.1 安装依赖

```bash
pip install fastapi uvicorn
```

### 8.2 启动服务

```bash
python test/chat_server.py \
  --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --port 8000
```

### 8.3 使用方式

- Web UI：打开 `http://127.0.0.1:8000/`
- CLI：

```bash
python test/chat_cli.py --url http://127.0.0.1:8000/v1/chat/completions --stream
```

---

## 9. 阶段结论

项目三核心目标已落地：
1. 采样能力从 argmax 扩展到 `top_k/top_p/temperature`。  
2. 提供 OpenAI 风格聊天服务接口，并支持流式输出。  
3. 提供 CLI 与 Web UI 两种可连续对话入口。  

当前系统可作为项目三提交版本，并为项目四（多用户 + 连续批处理）提供稳定起点。

