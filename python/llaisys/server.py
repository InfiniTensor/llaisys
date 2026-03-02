import os
import time
import uuid
import json
import asyncio
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer

import llaisys
from llaisys import DeviceType

# ==============================================================================
# Global State & Configuration
# ==============================================================================

class GlobalState:
    model: Optional[llaisys.models.Qwen2] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_path: str = "/home/wsl/model/DeepSeek-R1-Distill-Qwen-1.5B"  # Default path
    session_store: Dict[str, Any] = {}

state = GlobalState()

# Lock for single-user blocking processing (Project #3 Requirement)
# We use an asyncio Lock to ensure only one request is processed at a time.
generation_lock = asyncio.Lock()

# ==============================================================================
# Data Models (OpenAI Compatible)
# ==============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r1-distill-qwen-1.5b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    seed: Optional[int] = -1
    session_id: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

# ==============================================================================
# Lifespan & App Initialization
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    print(f"Loading tokenizer from {state.model_path}...")
    state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    
    print(f"Loading LLAISYS model from {state.model_path}...")
    state.model = llaisys.models.Qwen2(state.model_path, DeviceType.CPU)
    
    print("Model loaded successfully!")
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

app = FastAPI(title="LLAISYS Chatbot Server", lifespan=lifespan)

# ==============================================================================
# Helper Functions
# ==============================================================================

def _format_sse(data: BaseModel) -> str:
    return data.model_dump_json()

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not state.model or not state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Acquire lock for single-user processing
    async with generation_lock:
        # 0. Resolve Session
        session = None
        if request.session_id:
            if request.session_id in state.session_store:
                session = state.session_store[request.session_id]
            else:
                print(f"Creating new session for id: {request.session_id}")
                session = state.model.create_session()
                state.session_store[request.session_id] = session

        # 1. Prepare Prompt
        try:
            # Convert Pydantic models to dicts for apply_chat_template
            messages_list = [msg.model_dump() for msg in request.messages]
            prompt_text = state.tokenizer.apply_chat_template(
                messages_list, tokenize=False, add_generation_prompt=True
            )
            input_ids = state.tokenizer.encode(prompt_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Tokenization failed: {str(e)}")

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        # 2. Stream Response
        if request.stream:
            async def event_generator():
                stream_gen = state.model.generate(
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                    top_p=request.top_p if request.top_p is not None else 0.0,
                    temperature=request.temperature,
                    seed=request.seed if request.seed is not None else -1,
                    stream=True,
                    session=session
                )
                
                # Send initial role
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None
                    )]
                )
                yield {"data": _format_sse(chunk)}

                # Stream tokens
                generated_tokens = []
                for token_id in stream_gen:
                    generated_tokens.append(token_id)
                    # Decode incrementally (simplification: decode single token)
                    # In production, use a proper incremental decoder for UTF-8 safety
                    word = state.tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=request.model,
                        choices=[ChatCompletionStreamResponseChoice(
                            index=0,
                            delta={"content": word},
                            finish_reason=None
                        )]
                    )
                    yield {"data": _format_sse(chunk)}
                    
                    # Yield to event loop to allow other async tasks (like heartbeats)
                    # though we are holding the generation_lock, so no other generation happens.
                    await asyncio.sleep(0)

                # Send finish
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )]
                )
                yield {"data": _format_sse(chunk)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(event_generator())

        # 3. Non-Stream Response
        else:
            output_ids = state.model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                top_k=request.top_k,
                top_p=request.top_p if request.top_p is not None else 0.0,
                temperature=request.temperature,
                seed=request.seed if request.seed is not None else -1,
                stream=False,
                session=session
            )
            
            # The generate method returns the full sequence (input + output)
            # We need to extract only the new tokens
            new_tokens = output_ids[len(input_ids):]
            content = state.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Construct usage dict
            usage = {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(new_tokens),
                "total_tokens": len(output_ids)
            }

            return ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop"
                )],
                usage=usage
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
