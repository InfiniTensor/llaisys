# Project 3: AI Chatbot 实现报告

## 项目概述

Project 3 的目标是在 LLAISYS 推理框架上搭建一个可以真正对话的 AI 聊天机器人。整体分三块：随机采样算子、后端服务器、前端聊天界面。

---

## 一、随机采样（Sample 算子）

前两个作业里模型生成 token 用的是 argmax——每次直接取概率最高的那个词。这样生成的文字很确定，但也很死板，同一个问题永远给同一个答案。Project 3 要求实现真正的随机采样，让模型的回复更自然。

### 实现思路

采样算子在 `src/ops/sample/cpu/sample_cpu.cpp` 里，核心逻辑分六步：

**第一步：温度缩放（Temperature）**

模型最后一层输出的是 logits（原始分数），不是概率。在转成概率之前，先把所有 logits 除以温度参数 `temperature`：

```cpp
scores[i] = val / temperature;
```

温度低于 1 时，高分和低分的差距被放大，模型更倾向于选高概率词（更保守）；温度高于 1 时差距缩小，选择更随机（更有创意）。温度等于 1 时不做任何改变。

**第二步：排序**

把所有 token 按分数从高到低排序。这里排的是下标数组而不是分数本身，方便后面映射回原始 token ID：

```cpp
std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return scores[a] > scores[b];
});
```

**第三步：Top-K 过滤**

只保留分数最高的 K 个候选词，其余直接丢掉。`top_k=1` 就退化成 argmax，`top_k=0` 表示不过滤。

**第四步：Softmax**

对保留下来的候选词做 softmax，把分数转成概率。为了数值稳定，先减去最大值再做 exp，避免数值溢出：

```cpp
float max_score = scores[indices[0]];
probs[i] = std::exp(scores[indices[i]] - max_score);
```

**第五步：Top-P（核采样）**

按概率从高到低累加，一旦累积概率超过阈值 `top_p` 就截断，剩下的 token 重新归一化。比如 `top_p=0.9`，意思是只从"概率之和占 90%"的那些词里采样，尾部的低概率词被排除。

**第六步：多项式采样**

用 `std::discrete_distribution` 按概率权重随机抽一个 token：

```cpp
std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
*sampled_token = static_cast<int64_t>(indices[dist(rng)]);
```

随机数生成器用 `thread_local` 修饰，每个线程独立一份，避免多线程竞争。种子混合了硬件熵和当前时间，解决 Windows 上 `random_device` 可能返回固定值的问题。

---

## 二、后端服务器

服务器用 FastAPI 实现，代码在 `test/server/` 目录下，分 `main.py`（启动入口）和 `routes.py`（路由逻辑）两个文件。

### 整体架构

```
浏览器  <──SSE──>  FastAPI (routes.py)  <──ctypes──>  C++ 推理后端
                        │
                   HuggingFace Tokenizer（负责编解码）
```

C++ 模型通过 ctypes 封装暴露给 Python，tokenizer 直接用 HuggingFace 的 `AutoTokenizer`。

### 启动流程

`main.py` 里的 `main()` 函数负责启动：先加载 C++ 模型和 tokenizer，再启动 uvicorn HTTP 服务器。模型在服务器启动前就加载好，不会让第一个请求等待。

```python
model = llaisys.models.Qwen2(args.model, device=device)
tokenizer = AutoTokenizer.from_pretrained(args.model)
set_model(model, tokenizer)
uvicorn.run(app, host=args.host, port=args.port)
```

### 会话管理

服务器用一个全局字典 `SESSIONS` 存所有会话，每个会话有唯一的 8 位 UUID、标题和完整的对话历史：

```python
SESSIONS: dict = {}  # session_id -> {id, title, history}
```

提供四个 REST 接口：
- `GET /v1/sessions` — 列出所有会话
- `POST /v1/sessions` — 新建会话
- `GET /v1/sessions/{id}` — 获取某个会话的完整历史
- `DELETE /v1/sessions/{id}` — 删除会话

### 聊天接口

核心接口是 `POST /v1/chat/completions`，兼容 OpenAI 的 API 格式，支持流式和非流式两种模式。

每次请求时，客户端把完整的对话历史发过来，服务器用 tokenizer 的 `apply_chat_template` 把历史格式化成模型训练时用的 prompt 格式，再编码成 token ID 序列送给 C++ 模型。

```python
text = TOKENIZER.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
tokens = TOKENIZER.encode(text)
```

### 流式响应（SSE）

流式模式是这个服务器最有意思的部分。浏览器不用等模型生成完才看到回复，而是每生成一个 token 就立刻收到一段文字，体验上就像模型在"打字"。

技术上用的是 SSE（Server-Sent Events）协议，每个事件格式如下：

```
data: {"choices": [{"delta": {"content": "你好"}}]}\n\n
```

但有个问题：C++ 推理是同步阻塞的，而 FastAPI 跑在 asyncio 事件循环上。如果直接在 async 函数里调用阻塞代码，整个服务器会卡死。解决方法是用 `run_in_executor` 把每次推理调用丢到线程池里执行：

```python
next_token = await loop.run_in_executor(None, _infer_one)
```

这样事件循环可以在等待推理结果的同时，把已经生成的 chunk 刷给客户端。

另一个细节是**增量解码**。BPE 分词器的特性是，有些 token 单独解码会产生乱码，必须和前后 token 一起解码才能得到正确文字（比如中文字符经常跨多个 token）。所以每步都解码全部已生成的 token，然后取新增的部分作为 delta：

```python
raw = TOKENIZER.decode(generated, skip_special_tokens=False)
clean = _clean_output(raw)
delta = clean[prev_clean_len:]
```

### DeepSeek 思维链过滤

DeepSeek-R1 模型会在回答前先输出一段 `<think>...</think>` 包裹的推理过程，这部分不应该展示给用户。`_clean_output` 函数用正则把它过滤掉：

```python
text = re.sub(r"<think>[\s\S]*?</think>", "", text)
# 处理 <think> 在 prompt 模板里、只有 </think> 出现在输出里的情况
text = re.sub(r"^[\s\S]*?</think>", "", text)
# 过滤 <|end_of_sentence|> 等特殊 token
text = re.sub(r"<[|｜][^|｜]*[|｜]>", "", text)
```

---

## 三、前端聊天界面

前端是纯原生 HTML + CSS + JavaScript，不依赖任何框架，代码在 `test/server/static/` 目录下。

### 页面结构

页面分左右两栏：左侧是会话列表（sidebar），右侧是聊天窗口。

```
┌─────────────┬──────────────────────────────┐
│  Chats  [+] │  LLAISYS Chatbot        [⚙]  │
│─────────────│──────────────────────────────│
│ > 会话 1    │                              │
│   会话 2    │        消息区域              │
│   会话 3    │                              │
│             │──────────────────────────────│
│             │  [输入框]          [Send]    │
└─────────────┴──────────────────────────────┘
```

右上角的齿轮按钮可以展开参数面板，调整 Temperature、Top-K、Top-P 和最大生成长度。

### 客户端状态管理

所有会话数据存在一个 `sessions` 对象里，`activeId` 记录当前显示的会话：

```javascript
let sessions = {};  // { session_id: { id, title, history } }
let activeId = null;
```

切换会话时直接从本地 `sessions` 里读历史重新渲染，不需要再请求服务器，切换是即时的。

### 流式接收

发送消息后，前端用 `fetch` + `ReadableStream` 读取 SSE 流：

```javascript
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // 按行解析 SSE 事件，提取 delta 内容追加到气泡里
}
```

每收到一个 delta 就更新气泡的文字内容，实现打字机效果。

### 页面刷新恢复

页面加载时会从服务器拉取已有的会话列表和历史，刷新页面后对话不会丢失：

```javascript
async function init() {
    const existing = await fetchSessions();
    for (const s of existing) {
        const history = await fetchSessionHistory(s.id);
        sessions[s.id] = { id: s.id, title: s.title, history };
    }
    // 恢复第一个会话，或自动新建一个
}
```

---

## 四、模型推理核心（KV Cache）

模型推理在 `src/models/qwen2.cpp` 里，每次生成一个 token 调用 `forward_token`，流程是标准的 Transformer decoder：

```
token_id → embedding → [×28层: RMSNorm → QKV Linear → RoPE → Attention → Linear → Add
                                       → RMSNorm → Gate/Up Linear → SwiGLU → Down Linear → Add]
         → RMSNorm → LM Head Linear → logits → sample → next token
```

KV Cache 是让推理速度可用的关键。没有 KV Cache 的话，每生成一个新 token，都要把整个历史序列重新过一遍 attention，计算量随序列长度线性增长。有了 KV Cache，每层的 K 和 V 矩阵在计算过后就存起来，下一步只需要计算新 token 的 Q，然后和缓存里的 K、V 做 attention：

```cpp
// 把当前 token 的 k, v 写入缓存
write_kv_cache(layer, pos, k_rope, v_view);

// 取出从位置 0 到当前位置的全部 k, v
auto k_total = _k_cache[layer]->slice(0, 0, pos + 1);
auto v_total = _v_cache[layer]->slice(0, 0, pos + 1);

// 只有当前 token 的 q，但 k/v 包含全部历史
llaisys::ops::self_attention(attn_val, q_rope, k_total, v_total, scale);
```

每层的 KV Cache 预分配为 `(maxseq, nkvh, dh)` 的张量，`maxseq` 是最大序列长度，避免运行时动态分配内存。

---

## 五、启动方式

```bash
# 安装依赖
pip install fastapi uvicorn

# 启动服务器
python test/server/main.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --port 8000

# 浏览器访问
http://localhost:8000
```
