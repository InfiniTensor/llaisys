# 文件位置：python/server.py
import os
import json
import time
import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# 导入你编译好的 llaisys 库
import llaisys

app = FastAPI(title="LLAiSYS Chat Server")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. 定义 OpenAI 兼容的请求数据结构 ---
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

# 全局变量存放模型和分词器
tokenizer = None
model = None

# --- 2. 核心路由：处理聊天请求 ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global tokenizer, model
    
    # 提取历史消息，并使用 Qwen2 自带的 Chat Template 拼接 Prompt
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )
    
    # 编码为 Token ID 列表
    input_ids = tokenizer.encode(prompt)
    
    # --- 3. 阻塞执行模型推理 ---
    outputs = model.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    )
    
    # 切片拿到新生成的 token
    new_tokens = outputs[len(input_ids):] if len(outputs) > len(input_ids) else outputs
    # 提前解码出完整文本，供非流式使用
    full_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # --- 4. 核心流式生成逻辑 (SSE) ---
    async def generate_stream():
        # 模拟流式输出打字机效果
        for token_id in new_tokens:
            if token_id == tokenizer.eos_token_id:
                break
            word = tokenizer.decode([token_id], skip_special_tokens=True)
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": word}}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.02) # 控制打字机速度
            
        yield "data: [DONE]\n\n"

    # --- 5. 根据前端请求，返回流式或非流式数据格式 ---
    if request.stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        # 正规的 OpenAI 非流式响应结构 (NextChat 的后台总结标题会走这里)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }]
        }


# --- 6. 服务启动与初始化 ---
def main():
    global tokenizer, model

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--device-id", default=0, type=int) 
    parser.add_argument("--port", default=8199, type=int)
    args = parser.parse_args()

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"🚀 [INIT] Finding or downloading model: {model_id}...")
    
    # 自动获取本地缓存路径
    model_path = snapshot_download(model_id)
    print(f"📦 [INIT] Model cache path: {model_path}")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载你的自研模型
    device_type = llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.CPU
    print(f"⚙️ [INIT] Loading LLAiSYS model to {device_type.name}:{args.device_id}...")
    
    # 【注意】这里如果你的 C++ 库目前只接受2个参数(model_path, device_type)，请把 args.device_id 删掉。
    # 如果你已经按我之前说的改了代码支持 device_id，那这里就保留不变
    try:
        model = llaisys.models.Qwen2(model_path, device_type, args.device_id)
    except TypeError:
        model = llaisys.models.Qwen2(model_path, device_type)
    
    print(f"✅ [READY] Server starting on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()