import uvicorn
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ==========================================
# 1. 导入你的模型和分词器
# ==========================================
from llaisys.models.qwen2 import Qwen2
from transformers import AutoTokenizer

# ⚠️ 注意：请将这里的路径替换为你电脑上真实的 Qwen2 模型文件夹路径！
MODEL_PATH = "/home/koishiyo/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" 

print(f"正在加载分词器: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("正在将大模型权重载入 C++ 引擎内存，请稍候...")
model = Qwen2(MODEL_PATH)
print("模型加载完毕！LLAISYS 引擎启动成功！")

# 初始化 FastAPI
app = FastAPI(title="LLAISYS Chatbot")

class ChatRequest(BaseModel):
    message: str

# ==========================================
# 2. 前端 HTML + CSS + JS (流式打字机效果)
# ==========================================
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>LLAISYS 纯 C++ 推理引擎</title>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0; display: flex; justify-content: center; height: 100vh; align-items: center; }
        .chat-container { width: 100%; max-width: 800px; height: 90vh; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; }
        .header { background: #1a73e8; color: white; padding: 20px; text-align: center; font-size: 1.2em; font-weight: bold; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 8px; line-height: 1.6; word-wrap: break-word; }
        .user-message { background: #e3f2fd; color: #0d47a1; align-self: flex-end; border-bottom-right-radius: 0; }
        .bot-message { background: #f1f3f4; color: #202124; align-self: flex-start; border-bottom-left-radius: 0; }
        .input-area { display: flex; padding: 20px; background: #fff; border-top: 1px solid #e0e0e0; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ccc; border-radius: 24px; outline: none; font-size: 16px; }
        button { margin-left: 10px; padding: 12px 24px; background: #1a73e8; color: white; border: none; border-radius: 24px; cursor: pointer; font-size: 16px; font-weight: bold; transition: 0.2s;}
        button:hover { background: #1557b0; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">LLAISYS - 纯 C++ 驱动的 AI 助手</div>
        <div class="chat-box" id="chat-box">
            <div class="message bot-message">你好！我是由你从零手写的 LLAISYS 推理引擎驱动的。我们来聊天吧！</div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="输入你想说的话..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');

        function appendMessage(text, className, id = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${className}`;
            msgDiv.innerText = text;
            if (id) msgDiv.id = id;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return msgDiv;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            // 显示用户消息
            appendMessage(text, 'user-message');
            userInput.value = '';

            // 创建一个空的气泡，准备接收流式字符
            const loadingId = 'loading-' + Date.now();
            const botMsgDiv = appendMessage('', 'bot-message', loadingId); 

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });

                // 流式读取 SSE 数据
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let fullText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\\n');
                    
                    for (let line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            fullText += data.delta; 
                            botMsgDiv.innerText = fullText; // 逐字更新气泡
                            chatBox.scrollTop = chatBox.scrollHeight; // 自动滚到底部
                        }
                    }
                }
            } catch (error) {
                botMsgDiv.innerText = '网络连接失败，请检查 C++ 引擎状态。';
            }
        }

        function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    return HTMLResponse(content=html_content)

# ==========================================
# 3. 后端流式推理接口 (真正呼叫 C++ 底层)
# ==========================================
@app.post("/v1/chat/completions")
async def chat_api(req: ChatRequest):
    
    def stream_generator():
        try:
            # 1. 使用官方标准的对话模板 (Chat Template) 
            messages = [
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": req.message}
            ]
            # apply_chat_template 会自动且正确地把特殊符号转换成真正的控制 Token ID
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt_text)
            
            # 2. 呼叫 C++ 底层的 stream_generate 算子
            # 这里传的参数 (temperature, top_p, top_k) 会直接穿透到你手写的 sampler_cpu.cpp 里！
            for token_id in model.stream_generate(
                inputs=input_ids, 
                max_new_tokens=512, 
                temperature=0.8, 
                top_p=0.9, 
                top_k=50
            ):
                # 3. ID 还原成汉字
                word = tokenizer.decode([token_id], skip_special_tokens=True)
                
                # 4. 组装成 Server-Sent Events (SSE) 格式发送给前端
                if word:
                    yield f"data: {json.dumps({'delta': word}, ensure_ascii=False)}\n\n"
                    
        except Exception as e:
            print(f"推理时发生错误: {e}")
            yield f"data: {json.dumps({'delta': '[Engine Error]'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    # 启动命令
    uvicorn.run(app, host="127.0.0.1", port=8000)