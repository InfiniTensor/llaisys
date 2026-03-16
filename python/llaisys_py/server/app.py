"""
LLAISYS 聊天机器人服务端（FastAPI）。

- 提供 OpenAI 风格的 Chat Completions API（POST /v1/chat/completions）。
- 项目#4：多用户请求入队，单 worker 线程顺序处理；支持流式响应（SSE）与非流式。
- 会话管理：GET/POST/PATCH/DELETE /v1/sessions、GET /v1/sessions/{id}、POST /v1/sessions/{id}/regenerate。
- 模型与分词器在 create_app() 中按 MODEL_PATH 或参数加载，未加载时请求返回 503。
"""
import os
import json
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# ---------- 全局状态（在 create_app 中按需加载） ----------
_tokenizer = None   # HuggingFace AutoTokenizer，用于编码/解码与对话模板
_model = None       # LLAISYS Qwen2 模型实例
_device_type = None # 当前设备类型（如 "cpu"），供后续扩展用
_engine = None      # 连续批处理 Engine（当 LLAISYS_USE_ENGINE_LOOP=1 且 max_batch_size>1 时创建）

# ---------- 多轮对话 Web 页面（内联 HTML+JS，由 GET /chat 直接返回） ----------
# 包含：样式（气泡、滚动、状态栏）、对话历史 DOM、流式请求与 SSE 解析逻辑
_CHAT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLAISYS 聊天</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg: #f0f2f5;
      --surface: #fff;
      --user-bubble: #1890ff;
      --assistant-bubble: #e8e8e8;
      --text: #1f1f1f;
      --text-secondary: #666;
      --border: #e5e5e5;
      --shadow: 0 1px 2px rgba(0,0,0,.06);
      --radius: 12px;
      --radius-sm: 8px;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px 16px 24px;
    }
    .container { width: 100%; max-width: 640px; display: flex; flex-direction: column; height: calc(100vh - 88px); min-height: 420px; }
    .header {
      text-align: center;
      margin-bottom: 16px;
      padding: 14px 20px;
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .header h1 { margin: 0; font-size: 1.15rem; font-weight: 600; color: var(--text); }
    .header .hint { margin: 6px 0 0; font-size: 0.8rem; color: var(--text-secondary); }
    #history {
      flex: 1;
      overflow-y: auto;
      padding: 16px 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    #history .msg {
      display: flex;
      max-width: 85%;
      animation: fadeIn 0.25s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    #history .msg.user { align-self: flex-end; }
    #history .msg.assistant { align-self: flex-start; }
    #history .bubble {
      padding: 10px 14px;
      border-radius: var(--radius);
      font-size: 0.95rem;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: var(--shadow);
    }
    #history .msg.user .bubble {
      background: var(--user-bubble);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    #history .msg.assistant .bubble {
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }
    #history .msg .label { font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 4px; }
    #history .msg.user .label { text-align: right; }
    .input-area {
      flex-shrink: 0;
      padding: 12px 0 0;
      background: var(--bg);
    }
    #inputRow {
      display: flex;
      gap: 10px;
      align-items: center;
      background: var(--surface);
      padding: 10px 12px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }
    #userInput {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: border-color .2s;
    }
    #userInput:focus { border-color: var(--user-bubble); }
    #userInput:disabled { background: #f5f5f5; cursor: not-allowed; }
    #sendBtn {
      padding: 10px 20px;
      background: var(--user-bubble);
      color: #fff;
      border: none;
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: opacity .2s, transform .05s;
    }
    #sendBtn:hover:not(:disabled) { opacity: .92; }
    #sendBtn:active:not(:disabled) { transform: scale(0.98); }
    #sendBtn:disabled { opacity: .6; cursor: not-allowed; }
    .status {
      font-size: 0.8rem;
      margin-top: 10px;
      padding: 6px 10px;
      border-radius: var(--radius-sm);
      min-height: 1.2em;
    }
    .status.waiting { background: #fff7e6; color: #ad6800; }
    .status.error { background: #fff2f0; color: #cf1322; }
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <h1>LLAISYS 多轮对话</h1>
      <p class="hint">CPU 推理较慢，首次回复约需 1～3 分钟，请耐心等待。</p>
    </header>
    <div id="history"></div>
    <div class="input-area">
      <div id="inputRow">
        <input type="text" id="userInput" placeholder="输入消息，回车发送" autocomplete="off">
        <button type="button" id="sendBtn">发送</button>
      </div>
      <div class="status" id="status"></div>
    </div>
  </div>
  <script>
    // --- DOM 与状态 ---
    const historyEl = document.getElementById('history');  // 对话历史容器
    const inputEl = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const statusEl = document.getElementById('status');    // 底部状态/错误提示
    const messages = [];   // 多轮对话内容，每次请求会整段发给后端
    const MAX_TOKENS = 512; // 单次回复最多生成 token 数，可按需改大

    /** 设置底部状态文案与样式：空则清空；isWaiting 为 true 时显示“等待”样式；含“错误”则红色 */
    function setStatus(msg, isWaiting) {
      statusEl.textContent = msg || '';
      statusEl.className = 'status' + (isWaiting ? ' waiting' : '') + (msg && msg.indexOf('错误') === 0 ? ' error' : '');
    }

    /** 在对话历史末尾追加一条已完整内容的消息（用户或助手），并滚动到底部；返回气泡元素 */
    function addToHistory(role, content) {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + role;
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = role === 'user' ? '你' : '助手';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = content;
      wrap.appendChild(label);
      wrap.appendChild(bubble);
      historyEl.appendChild(wrap);
      historyEl.scrollTop = historyEl.scrollHeight;
      return bubble;
    }

    /** 先追加一条空的“助手”消息行，用于流式输出时逐段写入；返回该气泡元素供后续更新 */
    function addAssistantBubbleStreaming() {
      const wrap = document.createElement('div');
      wrap.className = 'msg assistant';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = '助手';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = '';
      wrap.appendChild(label);
      wrap.appendChild(bubble);
      historyEl.appendChild(wrap);
      historyEl.scrollTop = historyEl.scrollHeight;
      return bubble;
    }

    /**
     * 发送当前输入内容：追加用户消息到 history 与 messages，新建空助手气泡，流式请求并逐段更新气泡。
     * 请求使用 stream: true，通过 ReadableStream 读取 SSE（data: {...}\n\n），解析 delta.content 累加显示。
     * 结束后将完整回复写入 messages 以便下一轮多轮上下文正确。
     */
    async function send() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = '';
      sendBtn.disabled = true;
      inputEl.disabled = true;
      setStatus('正在生成回复（流式输出），请稍候…', true);
      messages.push({ role: 'user', content: text });
      addToHistory('user', text);

      const bubble = addAssistantBubbleStreaming();
      let fullContent = '';

      try {
        const r = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: messages,
            max_tokens: MAX_TOKENS,
            temperature: 0.3,
            stream: true
          })
        });
        if (!r.ok) {
          const err = await r.text();
          throw new Error(r.status + ' ' + err.slice(0, 300));
        }
        // 按 SSE 格式解析：每条事件为 "data: {...}"，用 getReader 逐块读取后按分割
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          buf += dec.decode(value, { stream: true });
          const lines = buf.split(/\\n\\n/);
          buf = lines.pop() || '';  // 未完成的一行留在 buf，下次循环再处理
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();  // trim 去掉可能的（Windows 换行）
            if (payload === '[DONE]') continue;
            try {
              const data = JSON.parse(payload);
              if (data && data.__error) {
                fullContent = data.__error || '请求失败';
                bubble.textContent = fullContent;
                break;
              }
              const delta = data.choices?.[0]?.delta?.content;
              if (delta) {
                fullContent += delta;
                bubble.textContent = fullContent;
                historyEl.scrollTop = historyEl.scrollHeight;
              }
            } catch (_) {}
          }
          if (done) break;
        }
        if (!fullContent) fullContent = '(无回复)';
        messages.push({ role: 'assistant', content: fullContent });
        setStatus('');
      } catch (e) {
        bubble.textContent = fullContent || '(请求失败)';
        if (fullContent) messages.push({ role: 'assistant', content: fullContent });
        setStatus('错误: ' + e.message, false);
        if (!fullContent) messages.pop();
      }
      sendBtn.disabled = false;
      inputEl.disabled = false;
    }

    sendBtn.onclick = send;
    inputEl.onkeydown = (e) => { if (e.key === 'Enter') send(); };
  </script>
</body>
</html>
"""

# ---------- Agent 风格页面（新路由 /agent，同 API，多块展示：思考 + 回答） ----------
_AGENT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLAISYS Agent</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg: #1a1b26;
      --surface: #24283b;
      --panel: #32364a;
      --user-bubble: #7aa2f7;
      --think-bg: #3b3f5c;
      --answer-bg: #32364a;
      --accent: #bb9af7;
      --text: #c0caf5;
      --text-dim: #a9b1d6;
      --border: #414868;
      --radius: 10px;
      --radius-sm: 6px;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px;
    }
    .container { width: 100%; max-width: 720px; display: flex; flex-direction: column; height: calc(100vh - 72px); min-height: 400px; }
    .header {
      text-align: center;
      margin-bottom: 12px;
      padding: 12px 16px;
      background: var(--surface);
      border-radius: var(--radius);
      border: 1px solid var(--border);
    }
    .header h1 { margin: 0; font-size: 1.1rem; font-weight: 600; color: var(--accent); }
    .header .hint { margin: 4px 0 0; font-size: 0.78rem; color: var(--text-dim); }
    #history {
      flex: 1;
      overflow-y: auto;
      padding: 12px 0;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    #history .msg { animation: fadeIn 0.2s ease; }
    #history .msg.user { display: flex; justify-content: flex-end; }
    #history .msg.user .bubble {
      max-width: 85%;
      padding: 10px 14px;
      background: var(--user-bubble);
      color: #1a1b26;
      border-radius: var(--radius);
      border-bottom-right-radius: 4px;
      font-size: 0.95rem;
    }
    #history .msg.assistant { display: flex; flex-direction: column; gap: 8px; }
    .msg.assistant .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .msg.assistant .card-head {
      padding: 6px 12px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--accent);
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }
    .msg.assistant .card-body {
      padding: 10px 14px;
      font-size: 0.9rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.assistant .think .card-body { background: var(--think-bg); color: var(--text-dim); }
    .msg.assistant .answer .card-body { background: var(--answer-bg); }
    .msg.assistant .think summary {
      cursor: pointer;
      list-style: none;
      padding: 6px 12px;
      font-size: 0.75rem;
      color: var(--text-dim);
    }
    .msg.assistant .think summary::-webkit-details-marker { display: none; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .input-area { flex-shrink: 0; padding-top: 12px; }
    #inputRow {
      display: flex;
      gap: 10px;
      align-items: center;
      background: var(--surface);
      padding: 10px 12px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
    }
    #userInput {
      flex: 1;
      padding: 10px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      color: var(--text);
      font-family: inherit;
      outline: none;
    }
    #userInput::placeholder { color: var(--text-dim); opacity: 0.8; }
    #userInput:focus { border-color: var(--accent); }
    #userInput:disabled { opacity: 0.6; cursor: not-allowed; }
    #sendBtn {
      padding: 10px 18px;
      background: var(--accent);
      color: var(--bg);
      border: none;
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
    }
    #sendBtn:hover:not(:disabled) { opacity: 0.9; }
    #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; }
    .status {
      font-size: 0.78rem;
      margin-top: 8px;
      padding: 6px 10px;
      border-radius: var(--radius-sm);
      min-height: 1.2em;
      color: var(--text-dim);
    }
    .status.waiting { background: var(--panel); color: var(--accent); }
    .status.error { background: #542426; color: #f7768e; }
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <h1>LLAISYS Agent</h1>
      <p class="hint">流式回复；若模型输出 think 标签内容，将自动拆为「思考」与「回答」两块展示。</p>
    </header>
    <div id="history"></div>
    <div class="input-area">
      <div id="inputRow">
        <input type="text" id="userInput" placeholder="输入任务或问题，回车发送" autocomplete="off">
        <button type="button" id="sendBtn">发送</button>
      </div>
      <div class="status" id="status"></div>
    </div>
  </div>
  <script>
    const historyEl = document.getElementById('history');
    const inputEl = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const statusEl = document.getElementById('status');
    const messages = [];
    const MAX_TOKENS = 256;

    function setStatus(msg, isWaiting) {
      statusEl.textContent = msg || '';
      statusEl.className = 'status' + (isWaiting ? ' waiting' : '') + (msg && msg.indexOf('错误') === 0 ? ' error' : '');
    }

    function addUserMessage(content) {
      const wrap = document.createElement('div');
      wrap.className = 'msg user';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = content;
      wrap.appendChild(bubble);
      historyEl.appendChild(wrap);
      historyEl.scrollTop = historyEl.scrollHeight;
    }

    function createAssistantReplyContainer() {
      const wrap = document.createElement('div');
      wrap.className = 'msg assistant';
      historyEl.appendChild(wrap);
      historyEl.scrollTop = historyEl.scrollHeight;
      return wrap;
    }

    function parseThinkAnswer(full) {
      var openTag = '\\u003cthink\\u003e';
      var closeTag = '\\u003c/think\\u003e';
      var openIdx = full.indexOf(openTag);
      var closeIdx = full.indexOf(closeTag);
      if (openIdx !== -1 && closeIdx > openIdx) {
        var think = full.slice(openIdx + openTag.length, closeIdx).trim();
        var answer = (full.slice(0, openIdx) + full.slice(closeIdx + closeTag.length)).trim();
        return { think: think, answer: answer || '(无)' };
      }
      return { think: '', answer: full };
    }

    function addThinkCard(container, thinkText) {
      if (!thinkText) return;
      const details = document.createElement('details');
      details.className = 'card think';
      details.innerHTML = '<summary>思考过程（点击展开）</summary><div class="card-body">' + escapeHtml(thinkText) + '</div>';
      container.insertBefore(details, container.firstChild);
    }

    function addAnswerCard(container, content, isStreaming) {
      const card = document.createElement('div');
      card.className = 'card answer';
      card.innerHTML = '<div class="card-head">回答</div><div class="card-body">' + escapeHtml(content) + '</div>';
      container.appendChild(card);
      const bodyEl = card.querySelector('.card-body');
      historyEl.scrollTop = historyEl.scrollHeight;
      return bodyEl;
    }

    function escapeHtml(s) {
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }

    async function send() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = '';
      sendBtn.disabled = true;
      inputEl.disabled = true;
      setStatus('正在生成…', true);
      messages.push({ role: 'user', content: text });
      addUserMessage(text);

      const container = createAssistantReplyContainer();
      let fullContent = '';
      let answerEl = null;

      try {
        const r = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: messages,
            max_tokens: MAX_TOKENS,
            temperature: 0.6,
            stream: true
          })
        });
        if (!r.ok) {
          const err = await r.text();
          throw new Error(r.status + ' ' + err.slice(0, 300));
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          buf += dec.decode(value, { stream: true });
          const lines = buf.split(/\\n\\n/);
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') continue;
            try {
              const data = JSON.parse(payload);
              if (data && data.__error) {
                fullContent = data.__error || '请求失败';
                break;
              }
              const delta = data.choices?.[0]?.delta?.content;
              if (delta) {
                fullContent += delta;
                const { think, answer } = parseThinkAnswer(fullContent);
                if (think && !container.querySelector('.think')) {
                  addThinkCard(container, think);
                }
                if (!answerEl) answerEl = addAnswerCard(container, answer || '(生成中…)', true);
                else answerEl.textContent = answer || '(生成中…)';
                historyEl.scrollTop = historyEl.scrollHeight;
              }
            } catch (_) {}
          }
          if (done) break;
        }
        const { think, answer } = parseThinkAnswer(fullContent);
        if (think && !container.querySelector('.think')) addThinkCard(container, think);
        if (!answerEl) answerEl = addAnswerCard(container, answer || fullContent || '(无回复)', false);
        else answerEl.textContent = answer || fullContent || '(无回复)';
        messages.push({ role: 'assistant', content: fullContent });
        setStatus('');
      } catch (e) {
        if (!answerEl) answerEl = addAnswerCard(container, fullContent || '(请求失败)', false);
        else answerEl.textContent = fullContent || '(请求失败)';
        if (fullContent) messages.push({ role: 'assistant', content: fullContent });
        setStatus('错误: ' + e.message, false);
        if (!fullContent) messages.pop();
      }
      sendBtn.disabled = false;
      inputEl.disabled = false;
    }

    sendBtn.onclick = send;
    inputEl.onkeydown = (e) => { if (e.key === 'Enter') send(); };
  </script>
</body>
</html>
"""

# ---------- 多会话管理页面：侧栏列表、新建/删除/切换、编辑消息并从此处重新生成 ----------
_SESSIONS_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLAISYS 会话</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg: #1a1b26;
      --sidebar: #16161e;
      --surface: #24283b;
      --border: #414868;
      --accent: #7aa2f7;
      --text: #c0caf5;
      --text-dim: #a9b1d6;
      --danger: #f7768e;
    }
    body { margin: 0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; }
    .sidebar {
      width: 260px; min-width: 200px; background: var(--sidebar); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; padding: 12px;
    }
    .sidebar h2 { margin: 0 0 12px; font-size: 1rem; color: var(--text-dim); }
    .sidebar .btn-new {
      padding: 8px 12px; background: var(--accent); color: var(--bg); border: none; border-radius: 6px;
      cursor: pointer; font-weight: 500; margin-bottom: 12px;
    }
    .sidebar .btn-new:hover { opacity: 0.9; }
    .session-list { flex: 1; overflow-y: auto; }
    .session-item {
      padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 4px;
      display: flex; justify-content: space-between; align-items: center; gap: 8px;
    }
    .session-item:hover { background: var(--surface); }
    .session-item.active { background: var(--surface); border: 1px solid var(--accent); }
    .session-item .title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9rem; }
    .session-item .btn-del {
      padding: 4px 8px; background: transparent; color: var(--text-dim); border: none; border-radius: 4px;
      cursor: pointer; font-size: 0.8rem; flex-shrink: 0;
    }
    .session-item .btn-del:hover { background: var(--danger); color: #fff; }
    .main {
      flex: 1; display: flex; flex-direction: column; min-width: 0; max-width: 100%;
    }
    .main-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--text-dim); }
    .chat-area { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { display: flex; flex-direction: column; gap: 4px; animation: fadeIn 0.2s ease; }
    .msg.user { align-items: flex-end; }
    .msg.assistant { align-items: flex-start; }
    .msg .bubble {
      max-width: 85%; padding: 10px 14px; border-radius: 10px; white-space: pre-wrap; word-break: break-word;
    }
    .msg.user .bubble { background: var(--accent); color: var(--bg); border-bottom-right-radius: 4px; }
    .msg.assistant .bubble { background: var(--surface); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .msg .label { font-size: 0.75rem; color: var(--text-dim); }
    .msg.user .actions { display: flex; gap: 6px; margin-top: 4px; }
    .msg .btn-edit, .msg .btn-regen {
      padding: 4px 10px; font-size: 0.78rem; border-radius: 6px; cursor: pointer; border: none; background: var(--surface); color: var(--text-dim);
    }
    .msg .btn-edit:hover, .msg .btn-regen:hover { background: var(--accent); color: var(--bg); }
    .input-row {
      padding: 12px 16px; background: var(--sidebar); border-top: 1px solid var(--border);
      display: flex; gap: 10px; align-items: center;
    }
    #userInput {
      flex: 1; padding: 10px 14px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font-size: 0.95rem; outline: none;
    }
    #userInput:focus { border-color: var(--accent); }
    #sendBtn {
      padding: 10px 20px; background: var(--accent); color: var(--bg); border: none; border-radius: 8px;
      font-weight: 500; cursor: pointer;
    }
    #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { font-size: 0.8rem; color: var(--text-dim); padding: 4px 0; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body>
  <aside class="sidebar">
    <h2>会话</h2>
    <button type="button" class="btn-new" id="btnNew">+ 新建对话</button>
    <div class="session-list" id="sessionList"></div>
  </aside>
  <main class="main">
    <div class="main-empty" id="mainEmpty">选择或新建一个对话</div>
    <div class="chat-area" id="chatArea" style="display:none;">
      <div id="history"></div>
    </div>
    <div class="input-row" id="inputRow" style="display:none;">
      <input type="text" id="userInput" placeholder="输入消息…" autocomplete="off">
      <button type="button" id="sendBtn">发送</button>
    </div>
    <div class="status" id="status"></div>
  </main>
  <script>
    const sessionListEl = document.getElementById('sessionList');
    const mainEmpty = document.getElementById('mainEmpty');
    const chatArea = document.getElementById('chatArea');
    const historyEl = document.getElementById('history');
    const inputRow = document.getElementById('inputRow');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const statusEl = document.getElementById('status');
    let currentSessionId = null;
    let currentAbortController = null;
    const MAX_TOKENS = 256;

    function setStatus(msg) { statusEl.textContent = msg || ''; }

    function abortCurrentStream() {
      if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
      }
    }

    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error(r.status + ' ' + (await r.text()));
      if (r.headers.get('content-type')?.includes('json')) return r.json();
      return r.text();
    }

    async function loadSessions() {
      const data = await api('/v1/sessions');
      sessionListEl.innerHTML = '';
      (data.sessions || []).forEach(s => {
        const div = document.createElement('div');
        div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
        div.innerHTML = '<span class="title">' + escapeHtml(s.title || '未命名') + '</span><button type="button" class="btn-del" data-id="' + escapeHtml(s.id) + '">删除</button>';
        div.querySelector('.title').onclick = () => switchSession(s.id);
        div.querySelector('.btn-del').onclick = (e) => { e.stopPropagation(); deleteSession(s.id); };
        sessionListEl.appendChild(div);
      });
    }

    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    let streamPollTimer = null;
    let streamPollLastContent = '';
    let streamPollUnchangedCount = 0;

    async function switchSession(sessionId) {
      if (streamPollTimer) {
        clearInterval(streamPollTimer);
        streamPollTimer = null;
      }
      currentSessionId = sessionId;
      await loadSessions();
      const s = await api('/v1/sessions/' + sessionId);
      if (currentSessionId !== sessionId) return;
      mainEmpty.style.display = 'none';
      chatArea.style.display = 'flex';
      inputRow.style.display = 'flex';
      renderMessages(s.messages || []);
      if ((s.messages || []).length > 0) {
        streamPollLastContent = JSON.stringify(s.messages);
        streamPollUnchangedCount = 0;
        streamPollTimer = setInterval(pollSessionStream, 500);
      }
    }

    async function pollSessionStream() {
      if (!currentSessionId) return;
      try {
        const s = await api('/v1/sessions/' + currentSessionId);
        const msgs = s.messages || [];
        const snapshot = JSON.stringify(msgs);
        if (snapshot !== streamPollLastContent) {
          streamPollLastContent = snapshot;
          streamPollUnchangedCount = 0;
          renderMessages(msgs);
          chatArea.scrollTop = chatArea.scrollHeight;
        } else {
          streamPollUnchangedCount++;
          if (streamPollUnchangedCount >= 4) {
            if (streamPollTimer) { clearInterval(streamPollTimer); streamPollTimer = null; }
          }
        }
      } catch (_) {}
    }

    function renderMessages(messages) {
      historyEl.innerHTML = '';
      const userIndices = [];
      messages.forEach((m, i) => {
        if (m.role === 'user') userIndices.push(historyEl.children.length);
        const wrap = document.createElement('div');
        wrap.className = 'msg ' + m.role;
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = m.role === 'user' ? '你' : '助手';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = m.content || '';
        wrap.appendChild(label);
        wrap.appendChild(bubble);
        if (m.role === 'user') {
          const actions = document.createElement('div');
          actions.className = 'actions';
          const idx = userIndices.length - 1;
          const btnEdit = document.createElement('button');
          btnEdit.className = 'btn-edit';
          btnEdit.textContent = '编辑';
          btnEdit.onclick = () => editAndRegenerate(idx, m.content);
          const btnRegen = document.createElement('button');
          btnRegen.className = 'btn-regen';
          btnRegen.textContent = '从此处重新生成';
          btnRegen.onclick = () => regenerateFrom(idx);
          actions.append(btnEdit, btnRegen);
          wrap.appendChild(actions);
        }
        historyEl.appendChild(wrap);
      });
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    async function editAndRegenerate(userMsgIndex, oldContent) {
      const newContent = prompt('编辑本条消息：', oldContent || '');
      if (newContent === null) return;
      await regenerateFrom(userMsgIndex, newContent);
    }

    async function regenerateFrom(userMsgIndex, newContent) {
      if (!currentSessionId) return;
      if (streamPollTimer) { clearInterval(streamPollTimer); streamPollTimer = null; }
      if (currentAbortController) currentAbortController.abort();
      currentAbortController = new AbortController();
      const streamSessionId = currentSessionId;
      setStatus('正在重新生成…');
      sendBtn.disabled = true;
      try {
        const r = await fetch('/v1/sessions/' + currentSessionId + '/regenerate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            from_message_index: userMsgIndex,
            new_content: newContent || undefined,
            max_tokens: MAX_TOKENS,
            stream: true
          }),
          signal: currentAbortController.signal
        });
        if (!r.ok) throw new Error(await r.text());
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let fullContent = '';
        const wrap = document.createElement('div');
        wrap.className = 'msg assistant';
        wrap.innerHTML = '<div class="label">助手</div><div class="bubble"></div>';
        const bubble = wrap.querySelector('.bubble');
        historyEl.appendChild(wrap);
        while (true) {
          const { value, done } = await reader.read();
          buf += dec.decode(value, { stream: true });
          const parts = buf.split(/\\n\\n/);
          buf = parts.pop() || '';
          for (const line of parts) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') break;
            try {
              const data = JSON.parse(payload);
              if (data && data.__error) {
                fullContent = data.__error || '请求失败';
                if (currentSessionId === streamSessionId) bubble.textContent = fullContent;
                break;
              }
              const delta = data.choices?.[0]?.delta?.content;
              if (delta) {
                fullContent += delta;
                if (currentSessionId === streamSessionId) {
                  bubble.textContent = fullContent;
                  chatArea.scrollTop = chatArea.scrollHeight;
                }
              }
            } catch (_) {}
          }
          if (done) break;
        }
        if (currentSessionId === streamSessionId && !fullContent) bubble.textContent = '(无回复)';
        if (currentSessionId === streamSessionId) {
          const s = await api('/v1/sessions/' + streamSessionId);
          renderMessages(s.messages || []);
        }
      } catch (e) {
        if (e.name !== 'AbortError' && currentSessionId === streamSessionId) setStatus('错误: ' + e.message);
      }
      currentAbortController = null;
      setStatus('');
      sendBtn.disabled = false;
    }

    async function deleteSession(sessionId) {
      if (!confirm('确定删除该对话？')) return;
      await api('/v1/sessions/' + sessionId, { method: 'DELETE' });
      if (currentSessionId === sessionId) {
        currentSessionId = null;
        mainEmpty.style.display = 'flex';
        chatArea.style.display = 'none';
        inputRow.style.display = 'none';
        historyEl.innerHTML = '';
      }
      await loadSessions();
    }

    async function createSession() {
      const s = await api('/v1/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      await loadSessions();
      switchSession(s.id);
    }

    async function send() {
      const text = userInput.value.trim();
      if (!text || !currentSessionId) return;
      if (streamPollTimer) {
        clearInterval(streamPollTimer);
        streamPollTimer = null;
      }
      if (currentAbortController) currentAbortController.abort();
      currentAbortController = new AbortController();
      const streamSessionId = currentSessionId;
      userInput.value = '';
      sendBtn.disabled = true;
      setStatus('正在生成…');
      const s = await api('/v1/sessions/' + currentSessionId);
      if (currentSessionId !== streamSessionId) { setStatus(''); sendBtn.disabled = false; currentAbortController = null; return; }
      const messages = (s.messages || []).concat([{ role: 'user', content: text }]);
      const bubbleWrap = document.createElement('div');
      bubbleWrap.className = 'msg user';
      bubbleWrap.innerHTML = '<div class="label">你</div><div class="bubble">' + escapeHtml(text) + '</div>';
      historyEl.appendChild(bubbleWrap);
      const assistantWrap = document.createElement('div');
      assistantWrap.className = 'msg assistant';
      assistantWrap.innerHTML = '<div class="label">助手</div><div class="bubble"></div>';
      const bubble = assistantWrap.querySelector('.bubble');
      historyEl.appendChild(assistantWrap);
      chatArea.scrollTop = chatArea.scrollHeight;
      try {
        const r = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: streamSessionId,
            messages: messages,
            max_tokens: MAX_TOKENS,
            stream: true
          }),
          signal: currentAbortController.signal
        });
        if (!r.ok) throw new Error(await r.text());
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          buf += dec.decode(value, { stream: true });
          const lines = buf.split(/\\n\\n/);
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') break;
            try {
              const data = JSON.parse(payload);
              if (data && data.__error) {
                if (currentSessionId === streamSessionId) bubble.textContent = data.__error || '请求失败';
                break;
              }
              const delta = data.choices?.[0]?.delta?.content;
              if (delta !== undefined && currentSessionId === streamSessionId) {
                if (delta === '') {
                  if (!bubble.textContent || bubble.textContent === '排队中…') bubble.textContent = '排队中…';
                } else {
                  if (bubble.textContent === '排队中…') bubble.textContent = delta;
                  else bubble.textContent = (bubble.textContent || '') + delta;
                  chatArea.scrollTop = chatArea.scrollHeight;
                }
              }
            } catch (_) {}
          }
          if (done) break;
        }
        if (currentSessionId === streamSessionId && !bubble.textContent.trim()) bubble.textContent = '(无回复)';
      } catch (e) {
        if (e.name !== 'AbortError') {
          setStatus('错误: ' + e.message);
          if (currentSessionId === streamSessionId) bubble.textContent = bubble.textContent || '(请求失败)';
        }
      }
      currentAbortController = null;
      setStatus('');
      sendBtn.disabled = false;
    }

    document.getElementById('btnNew').onclick = createSession;
    sendBtn.onclick = send;
    userInput.onkeydown = (e) => { if (e.key === 'Enter') send(); };
    loadSessions();
  </script>
</body>
</html>
"""


def _get_model():
    """返回已加载的 LLAISYS 模型；未加载时抛出 503。"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH and restart.")
    return _model


def _get_tokenizer():
    """返回已加载的分词器；未加载时抛出 503。"""
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded. Set MODEL_PATH and restart.")
    return _tokenizer


# ---------- 请求/响应模型（与 OpenAI Chat Completions 对齐） ----------

class ChatMessage(BaseModel):
    """单条对话消息。"""
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(default="", description="message content")


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions 的请求体。"""
    model: str = Field(default="default", description="模型名（当前忽略，使用服务端加载的模型）")
    messages: list[ChatMessage] = Field(..., description="多轮对话历史，最后一条一般为 user")
    session_id: Optional[str] = Field(default=None, description="可选；若提供，完成后将本轮 user+assistant 追加到该会话")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="本次最多生成的新 token 数")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="默认 0.3 减少胡言乱语；若回复太死板可试 0.5")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=100)
    stream: bool = Field(default=False, description="是否以 SSE 流式返回")
    seed: Optional[int] = Field(default=None, description="随机种子；None 表示非确定性")


# ---------- 会话存储与模型（内存存储，重启清空） ----------
_sessions: dict[str, dict] = {}  # session_id -> { id, title, messages, created_at, updated_at }
_sessions_lock = threading.Lock()

# ---------- KV-Cache 池（Phase 3）：key=(session_id, user_message_index), value={blob, prefix_len, last_used}, LRU ----------
_KV_POOL_MAX_ENTRIES = int(os.environ.get("LLAISYS_KV_POOL_MAX", "16"))
_kv_pool: dict[tuple[str, int], dict] = {}  # (session_id, user_idx) -> {"blob": bytes, "prefix_len": int, "last_used": float}
_kv_pool_lock = threading.Lock()
import time as _time_module

# ---------- 项目#4：请求队列与 worker ----------
_REQUEST_QUEUE_MAX = int(os.environ.get("LLAISYS_REQUEST_QUEUE_MAX", "64"))
_request_queue: queue.Queue = queue.Queue(maxsize=_REQUEST_QUEUE_MAX)
_inference_lock = threading.Lock()  # 推理互斥，worker 与 regenerate 共用模型时串行化
_STREAM_SENTINEL = None  # 流式结束标记，worker 放入 response_queue 表示结束


def _kv_pool_get(session_id: str, user_message_index: int):
    """若存在则返回 (blob, prefix_len) 并更新 last_used；否则返回 None。"""
    with _kv_pool_lock:
        key = (session_id, user_message_index)
        if key not in _kv_pool:
            return None
        entry = _kv_pool[key]
        entry["last_used"] = _time_module.perf_counter()
        return (entry["blob"], entry["prefix_len"])


def _kv_pool_put(session_id: str, user_message_index: int, blob: bytes, prefix_len: int) -> None:
    """写入池；若超过容量则 LRU 淘汰。"""
    with _kv_pool_lock:
        while len(_kv_pool) >= _KV_POOL_MAX_ENTRIES and _kv_pool:
            oldest_key = min(_kv_pool, key=lambda k: _kv_pool[k]["last_used"])
            del _kv_pool[oldest_key]
        key = (session_id, user_message_index)
        _kv_pool[key] = {"blob": blob, "prefix_len": prefix_len, "last_used": _time_module.perf_counter()}


def resolve_kv_prefix(session_id: Optional[str], request_messages: list, tokenizer, input_ids: list) -> tuple[int, Optional[bytes]]:
    """若 session 且可命中 KV 池则返回 (prefix_len, blob)；否则返回 (0, None)。供 Engine 与 worker 共用。"""
    if not session_id or not request_messages or tokenizer is None:
        return (0, None)
    user_count = _count_user_messages(request_messages)
    if user_count == 0:
        return (0, None)
    hit = _kv_pool_get(session_id, user_count - 1)
    if hit is None:
        return (0, None)
    blob, stored_prefix_len = hit
    need_prefix_len = _prefix_len_for_messages(request_messages, tokenizer)
    if stored_prefix_len != need_prefix_len or need_prefix_len <= 0 or need_prefix_len >= len(input_ids):
        return (0, None)
    return (stored_prefix_len, blob)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return _sessions[session_id]


class SessionCreate(BaseModel):
    """创建会话请求体。"""
    title: Optional[str] = Field(default=None, description="可选标题，默认用首条用户消息摘要")


class SessionUpdate(BaseModel):
    """更新会话请求体（PATCH）。"""
    title: Optional[str] = Field(default=None, description="新标题")


class SessionOut(BaseModel):
    """会话响应。"""
    id: str
    title: Optional[str]
    messages: list[dict]  # [{ role, content }]
    created_at: str
    updated_at: str


class RegenerateRequest(BaseModel):
    """从某条消息后重新生成请求体。"""
    from_message_index: int = Field(..., ge=0, description="保留到该条用户消息（含），之后全部删除并重新生成")
    new_content: Optional[str] = Field(default=None, description="若提供，替换该条用户消息内容")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=True)
    seed: Optional[int] = Field(default=None)


def _strip_think_tags(content: str) -> str:
    """去掉助手回复中的 <think>...</think> 推理块，只保留实际回答，避免存入会话后干扰后续对话。"""
    if not content or not isinstance(content, str):
        return content or ""
    import re
    content = re.sub(r"<think>[\s\S]*?<\s*/\s*think\s*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\u003cthink\u003e[\s\S]*?\u003c/think\u003e", "", content)
    return content.strip() or content


# 退化输出时返回给用户的提示（避免界面只显示 "1 1 1" 等）
_DEGENERATE_FALLBACK = "（回复异常，请重试。）"


def _is_degenerate_output(full_content: list[str], last_token_ids: list[int], max_recent: int = 20) -> bool:
    """
    检测是否陷入退化输出（如重复的 "1\\n0\\n0\\n"、纯数字、或同一词/ token 大量重复）。
    若最近内容仅包含数字/换行/空格且有一定长度，或同一词在最近片段中出现过多，则视为退化并建议停止。
    """
    if len(full_content) < 5:
        return False
    recent_text = "".join(full_content[-max_recent:])
    if not recent_text.strip():
        return True
    allowed = set(" \n\t\r0123456789")
    if not all(c in allowed for c in recent_text):
        # 非纯数字：检查是否同一词重复过多（如 "regards regards regards"）
        words = recent_text.split()
        if len(words) >= 5:
            from collections import Counter
            cnt = Counter(words)
            if cnt and cnt.most_common(1)[0][1] >= 4:
                return True
        return False
    # 纯数字/换行且长度>=4 即视为退化（如 "1\\n0\\n" 或 "12\\n13"）
    if len(recent_text.strip()) >= 4:
        return True
    return False


def _is_content_only_digits_and_whitespace(full_content: list[str]) -> bool:
    """判断整段内容是否仅包含数字、空格、换行（用于退化时是否用 fallback 替换）。"""
    if not full_content:
        return True
    text = "".join(full_content).strip()
    if not text:
        return True
    return all(c in " \n\t\r0123456789" for c in text)


# 系统提示：留空避免触发 DeepSeek-R1 的固定“自我介绍”；仅当明确需要时可设环境变量
def _get_system_prompt() -> str:
    return os.environ.get("LLAISYS_SYSTEM_PROMPT", "").strip()


def _messages_to_prompt(messages: list[ChatMessage], tokenizer) -> str:
    """将 OpenAI 风格 messages 转为带对话模板的输入文本（含 system/user/assistant 格式）。"""
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    system = _get_system_prompt()
    if system and not any(m.get("role") == "system" for m in conversation):
        conversation.insert(0, {"role": "system", "content": system})
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    if os.environ.get("LLAISYS_DEBUG"):
        import sys
        print(f"[LLAISYS] prompt len={len(prompt)} last_300={repr(prompt[-300:])}", file=sys.stderr)
    return prompt


def _count_user_messages(messages: list[ChatMessage]) -> int:
    return sum(1 for m in messages if m.role == "user")


def _prefix_len_for_messages(messages: list[ChatMessage], tokenizer) -> int:
    """编码「去掉最后一条」的 messages 得到的 token 长度，用于池 key 的 prefix_len 校验。"""
    if not messages:
        return 0
    conv = [{"role": m.role, "content": m.content} for m in messages[:-1]]
    system = _get_system_prompt()
    if system and not any(c.get("role") == "system" for c in conv):
        conv.insert(0, {"role": "system", "content": system})
    prompt = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    return len(tokenizer.encode(prompt))


# 当传入路径无效时尝试的备用模型目录（与 __main__.py 中 DEFAULT_MODEL_PATH 保持一致）
_FALLBACK_MODEL_PATH = "/home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B"


def create_app(model_path: Optional[str] = None, device: str = "cpu"):
    """
    创建 FastAPI 应用并（在路径有效时）加载模型与分词器。
    model_path 为空时从环境变量 MODEL_PATH 读取；路径无效时再尝试 _FALLBACK_MODEL_PATH；仍无效则不加载，请求返回 503。
    """
    global _tokenizer, _model, _device_type
    path = model_path or os.environ.get("MODEL_PATH")
    if not path or not os.path.isdir(path):
        path = _FALLBACK_MODEL_PATH if os.path.isdir(_FALLBACK_MODEL_PATH) else None
    if not path or not os.path.isdir(path):
        _tokenizer = None
        _model = None
        _device_type = device
        app = FastAPI(title="LLAISYS Chatbot", description="OpenAI chat-completion style API")
        _register_routes(app)
        return app

    # 在导入 transformers（会拉取 torch）之前检测 GPU；CUDA_VISIBLE_DEVICES="" 会导致 cudaGetDeviceCount() 返回 0
    if device == "nvidia":
        import sys
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if _cvd == "" or _cvd is None:
            # 空或未设置时强制至少可见 0 号卡，避免容器/DSW 默认传 "" 导致看不到 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if _cvd == "":
                print("Info: CUDA_VISIBLE_DEVICES was empty, set to 0 for this process.", file=sys.stderr)
        try:
            from ..libllaisys import DeviceType, LIB_LLAISYS
            nvidia_api = LIB_LLAISYS.llaisysGetRuntimeAPI(DeviceType.NVIDIA)
            _count = nvidia_api.contents.get_device_count()
            if _count == 0:
                print("Warning: no NVIDIA GPUs available (get_device_count()=0), using CPU. (CUDA_VISIBLE_DEVICES=%r)" % os.environ.get("CUDA_VISIBLE_DEVICES"), file=sys.stderr)
                device = "cpu"
            else:
                print("Info: using NVIDIA GPU(s), count=%s." % _count, file=sys.stderr)
        except Exception as e:
            print("Warning: NVIDIA check failed (%s), using CPU." % e, file=sys.stderr)
            device = "cpu"

    from transformers import AutoTokenizer
    from ..libllaisys import DeviceType
    from ..models.qwen2 import Qwen2

    _device_type = device
    _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dev = DeviceType.NVIDIA if device == "nvidia" else DeviceType.CPU
    use_engine = os.environ.get("LLAISYS_USE_ENGINE_LOOP", "").strip().lower() in ("1", "true", "yes")
    max_batch_size = int(os.environ.get("LLAISYS_MAX_BATCH_SIZE", "4" if use_engine else "1"))
    _model = Qwen2(path, device=dev, max_batch_size=max_batch_size)

    global _engine
    _engine = None
    if use_engine and max_batch_size >= 1:
        from .engine import Engine
        def _engine_get_kv(session_id, request_messages, input_ids):
            return resolve_kv_prefix(session_id, request_messages, _tokenizer, input_ids)
        def _engine_put_kv(session_id, request_messages, blob, prefix_len):
            _kv_pool_put(session_id, _count_user_messages(request_messages), blob, prefix_len)
        _engine = Engine(
            _model,
            max_batch_size,
            pending_maxsize=_REQUEST_QUEUE_MAX,
            get_kv=_engine_get_kv,
            put_kv=_engine_put_kv,
        )

    app = FastAPI(title="LLAISYS Chatbot", description="OpenAI chat-completion style API")
    _register_routes(app)
    return app


def _register_routes(app: FastAPI):
    """注册所有 HTTP 路由。"""

    @app.get("/")
    def root():
        """根路径：返回服务说明与常用链接。"""
        return {
            "message": "LLAISYS Chatbot Server",
            "docs": "/docs",
            "health": "/health",
            "chat_ui": "/chat",
            "agent_ui": "/agent",
            "sessions_ui": "/sessions",
            "chat_api": "POST /v1/chat/completions",
        }

    @app.get("/health")
    def health():
        """健康检查：是否存活及模型是否已加载。"""
        return {"status": "ok", "model_loaded": _model is not None}

    @app.get("/v1/metrics")
    def metrics():
        """项目#4 监控指标（4.5.3）：队列长度、Engine 状态、KV 池大小等。"""
        out = {
            "request_queue_size": _request_queue.qsize(),
            "request_queue_max": _REQUEST_QUEUE_MAX,
            "kv_pool_size": len(_kv_pool),
            "kv_pool_max": _KV_POOL_MAX_ENTRIES,
        }
        if _engine is not None:
            out["engine"] = _engine.get_metrics()
        return out

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page():
        """返回内联的 Web 聊天页（HTML+JS），支持多轮对话与流式显示。"""
        return _CHAT_HTML

    @app.get("/agent", response_class=HTMLResponse)
    def agent_page():
        """返回 Agent 风格页面：思考 + 回答分块展示，仍使用同一流式 API。"""
        return _AGENT_HTML

    @app.get("/sessions", response_class=HTMLResponse)
    def sessions_page():
        """多会话管理 UI：列表、新建/删除/切换、编辑消息并从此处重新生成。"""
        return _SESSIONS_HTML

    # ---------- 会话 API ----------
    @app.get("/v1/sessions")
    def list_sessions():
        """列出所有会话，按 updated_at 倒序。"""
        with _sessions_lock:
            out = []
            for sid, s in _sessions.items():
                out.append({
                    "id": s["id"],
                    "title": s.get("title"),
                    "updated_at": s["updated_at"],
                    "message_count": len(s.get("messages", [])),
                })
        out.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"sessions": out}

    @app.post("/v1/sessions")
    def create_session(body: Optional[SessionCreate] = None):
        """创建新会话。"""
        sid = str(uuid.uuid4())
        now = _now_iso()
        with _sessions_lock:
            _sessions[sid] = {
                "id": sid,
                "title": (body.title if body else None) or "新对话",
                "messages": [],
                "created_at": now,
                "updated_at": now,
            }
            return _sessions[sid]

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str):
        """获取会话详情。"""
        s = _get_session(session_id)
        return SessionOut(
            id=s["id"],
            title=s.get("title"),
            messages=s.get("messages", []),
            created_at=s["created_at"],
            updated_at=s["updated_at"],
        )

    @app.patch("/v1/sessions/{session_id}")
    def update_session(session_id: str, body: SessionUpdate):
        """更新会话（如标题）。"""
        with _sessions_lock:
            if session_id not in _sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            s = _sessions[session_id]
            if body.title is not None:
                s["title"] = body.title
            s["updated_at"] = _now_iso()
            return dict(s)

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str):
        """删除会话。"""
        with _sessions_lock:
            if session_id not in _sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            del _sessions[session_id]
        return {"ok": True}

    def _resolve_kv_prefix_regenerate(session_id: str, from_message_index: int, request_messages: list[ChatMessage], tokenizer, input_ids: list[int]):
        """Regenerate 专用：查池 key=(session_id, from_message_index)，命中则返回 (prefix_len, blob)。"""
        hit = _kv_pool_get(session_id, from_message_index)
        if hit is None:
            return (0, None)
        blob, stored_prefix_len = hit
        need_prefix_len = _prefix_len_for_messages(request_messages, tokenizer)
        if stored_prefix_len != need_prefix_len or need_prefix_len <= 0 or need_prefix_len >= len(input_ids):
            return (0, None)
        return (stored_prefix_len, blob)

    @app.post("/v1/sessions/{session_id}/regenerate")
    def regenerate_session(session_id: str, body: RegenerateRequest):
        """从某条用户消息后截断并重新生成；可选替换该条内容；命中 KV 池则 suffix prefill，并写回池。"""
        s = _get_session(session_id)
        messages = list(s.get("messages", []))
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if body.from_message_index >= len(user_indices):
            raise HTTPException(
                status_code=400,
                detail="from_message_index out of range (no such user message)",
            )
        cut_at = user_indices[body.from_message_index]
        messages = messages[: cut_at + 1]
        if body.new_content is not None:
            messages[-1] = {"role": "user", "content": body.new_content}
        s["messages"] = messages
        s["updated_at"] = _now_iso()

        tokenizer = _get_tokenizer()
        model = _get_model()
        seed = body.seed if body.seed is not None else 0
        request_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        prompt = _messages_to_prompt(request_messages, tokenizer)
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            raise HTTPException(status_code=400, detail="Empty input after encoding")

        prefix_len, kv_blob = _resolve_kv_prefix_regenerate(session_id, body.from_message_index, request_messages, tokenizer, input_ids)

        if body.stream:
            def gen():
                with _inference_lock:
                    if kv_blob is not None and prefix_len > 0:
                        model.import_kv_cache(kv_blob, prefix_len)
                    full_content = []
                    tokens = list(input_ids)
                    next_id = None
                    n_remaining = body.max_tokens
                    if prefix_len == 0:
                        model.reset_kv_cache()
                    if prefix_len > 0 and prefix_len < len(input_ids):
                        suffix = input_ids[prefix_len:]
                        next_id = model.next_token(
                            suffix,
                            temperature=body.temperature,
                            top_k=body.top_k,
                            top_p=body.top_p,
                            seed=seed,
                        )
                        tokens.append(next_id)
                        if next_id != model.end_token:
                            delta_text = tokenizer.decode([next_id], skip_special_tokens=True)
                            if delta_text:
                                full_content.append(delta_text)
                                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:24]}', 'choices': [{'index': 0, 'delta': {'content': delta_text}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                        n_remaining = body.max_tokens - 1
                    for _ in range(n_remaining):
                        if next_id == model.end_token:
                            break
                        next_id = model.next_token(
                            tokens[-1:] if len(tokens) > 1 else tokens,
                            temperature=body.temperature,
                            top_k=body.top_k,
                            top_p=body.top_p,
                            seed=seed,
                        )
                        tokens.append(next_id)
                        if next_id == model.end_token:
                            break
                        delta_text = tokenizer.decode([next_id], skip_special_tokens=True)
                        if not delta_text:
                            continue
                        full_content.append(delta_text)
                        yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:24]}', 'choices': [{'index': 0, 'delta': {'content': delta_text}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:24]}', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    content = "".join(full_content) or "(无回复)"
                    s["messages"].append({"role": "assistant", "content": _strip_think_tags(content)})
                    s["updated_at"] = _now_iso()
                    _kv_pool_put(session_id, body.from_message_index + 1, model.export_kv_cache(), len(input_ids))

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        with _inference_lock:
            if kv_blob is not None and prefix_len > 0:
                model.import_kv_cache(kv_blob, prefix_len)
            full_tokens = model.generate(
                input_ids,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                seed=seed,
                prefix_len=prefix_len,
            )
            new_tokens = full_tokens[len(input_ids):]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True) or "(无回复)"
            content_clean = _strip_think_tags(content)
            s["messages"].append({"role": "assistant", "content": content_clean})
            s["updated_at"] = _now_iso()
            _kv_pool_put(session_id, body.from_message_index + 1, model.export_kv_cache(), len(input_ids))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content_clean}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(new_tokens), "total_tokens": len(full_tokens)},
        }

    def _resolve_kv_prefix(session_id: Optional[str], request_messages: list[ChatMessage], tokenizer, input_ids: list[int]):
        """若 session 且可命中池则返回 (prefix_len, blob)；否则返回 (0, None)。"""
        return resolve_kv_prefix(session_id, request_messages, tokenizer, input_ids)

    def _worker_loop():
        """项目#4：单 worker 线程，从请求队列取任务并执行推理，结果放入各请求的 response_queue。"""
        while True:
            item = _request_queue.get()
            resp_queue = item["response_queue"]
            cancel_event = item.get("cancel_event")
            try:
                with _inference_lock:
                    tokenizer = _get_tokenizer()
                    model = _get_model()
                    if tokenizer is None or model is None:
                        resp_queue.put({"__error": "Model not loaded", "status_code": 503})
                        continue
                    session_id = item.get("session_id")
                    messages = item["messages"]
                    stream = item["stream"]
                    seed = item["seed"]
                    prompt = _messages_to_prompt(messages, tokenizer)
                    input_ids = tokenizer.encode(prompt)
                    if not input_ids:
                        resp_queue.put({"__error": "Empty input after encoding", "status_code": 400})
                        continue
                    prefix_len, kv_blob = _resolve_kv_prefix(session_id, messages, tokenizer, input_ids)
                    if kv_blob is not None and prefix_len > 0:
                        model.import_kv_cache(kv_blob, prefix_len)
                    if stream:
                        if session_id:
                            with _sessions_lock:
                                s = _sessions.get(session_id)
                                if s is not None:
                                    s["messages"] = [{"role": m.role, "content": m.content} for m in messages]
                                    s["messages"].append({"role": "assistant", "content": ""})
                                    s["updated_at"] = _now_iso()
                        cancel_check = (lambda: cancel_event.is_set()) if cancel_event else None
                        for chunk in _stream_chunks_generator(
                            model, tokenizer, input_ids, item["max_tokens"],
                            item["temperature"], item["top_k"], item["top_p"], seed,
                            prompt=prompt, session_id=session_id, request_messages=messages, prefix_len=prefix_len,
                            cancel_check=cancel_check,
                        ):
                            if cancel_event and cancel_event.is_set():
                                break
                            resp_queue.put(chunk)
                            if session_id and isinstance(chunk, str) and chunk.startswith("data: "):
                                try:
                                    payload = chunk[6:].strip().strip("\n")
                                    if payload and payload != "[DONE]":
                                        data = json.loads(payload)
                                        delta = (data.get("choices") or [{}])[0].get("delta") or {}
                                        delta_content = delta.get("content")
                                        if isinstance(delta_content, str) and delta_content:
                                            with _sessions_lock:
                                                s = _sessions.get(session_id)
                                                if s and s["messages"] and s["messages"][-1]["role"] == "assistant":
                                                    s["messages"][-1]["content"] += delta_content
                                                    s["updated_at"] = _now_iso()
                                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                                    pass
                        resp_queue.put(_STREAM_SENTINEL)
                    else:
                        full_tokens = model.generate(
                            input_ids,
                            max_new_tokens=item["max_tokens"],
                            temperature=item["temperature"],
                            top_k=item["top_k"],
                            top_p=item["top_p"],
                            seed=seed,
                            prefix_len=prefix_len,
                        )
                        new_tokens = full_tokens[len(input_ids):]
                        content = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        content_clean = _strip_think_tags(content)
                        if session_id:
                            with _sessions_lock:
                                s = _sessions.get(session_id)
                                if s is not None:
                                    s["messages"] = [{"role": m.role, "content": m.content} for m in messages]
                                    s["messages"].append({"role": "assistant", "content": content_clean})
                                    s["updated_at"] = _now_iso()
                                    if s.get("title") == "新对话" and messages:
                                        first = messages[0].content.strip()[:30]
                                        if first:
                                            s["title"] = first + ("…" if len(messages[0].content) > 30 else "")
                            _kv_pool_put(session_id, _count_user_messages(messages), model.export_kv_cache(), len(input_ids))
                        resp_queue.put({
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion",
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": content_clean}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(new_tokens), "total_tokens": len(full_tokens)},
                        })
            except Exception as e:
                resp_queue.put({"__error": str(e), "status_code": 500})

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        """OpenAI 风格对话补全。启用 Engine 时走连续批处理；否则入队由单 worker 顺序处理。"""
        if req.session_id:
            _get_session(req.session_id)  # 404 if not found
        seed = req.seed if req.seed is not None else 0

        # ---------- Engine 路径：连续批处理，Prefill + Batched Decode ----------
        if _engine is not None:
            from .engine import RequestState, _StreamError
            tokenizer = _get_tokenizer()
            model = _get_model()
            if tokenizer is None or model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            prompt = _messages_to_prompt(req.messages, tokenizer)
            input_ids = tokenizer.encode(prompt)
            if not input_ids:
                raise HTTPException(status_code=400, detail="Empty input after encoding")
            out_queue = queue.Queue()
            req_state = RequestState(
                request_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                prompt_tokens=input_ids,
                max_tokens=req.max_tokens,
                out_queue=out_queue,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=seed,
                session_id=req.session_id,
                request_messages=req.messages,
            )
            try:
                _engine.submit_request(req_state)
            except queue.Full:
                raise HTTPException(status_code=503, detail="Request queue full, try again later")

            if req.stream:
                def stream_from_engine():
                    full_content = []
                    while True:
                        item = req_state.out_queue.get(timeout=300)
                        if item is None:
                            break
                        if isinstance(item, _StreamError):
                            yield f"data: {json.dumps({'error': item.message})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        delta_text = tokenizer.decode([item], skip_special_tokens=True)
                        if delta_text:
                            full_content.append(delta_text)
                            yield f"data: {json.dumps({'id': req_state.request_id, 'choices': [{'index': 0, 'delta': {'content': delta_text}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'id': req_state.request_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    content = "".join(full_content) or "(无回复)"
                    if req.session_id and req.messages:
                        with _sessions_lock:
                            s = _sessions.get(req.session_id)
                            if s is not None:
                                s["messages"] = [{"role": m.role, "content": m.content} for m in req.messages]
                                s["messages"].append({"role": "assistant", "content": _strip_think_tags(content)})
                                s["updated_at"] = _now_iso()
                                if s.get("title") == "新对话" and req.messages:
                                    first = req.messages[0].content.strip()[:30]
                                    if first:
                                        s["title"] = first + ("…" if len(req.messages[0].content) > 30 else "")

                return StreamingResponse(
                    stream_from_engine(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            collected = []
            while True:
                item = req_state.out_queue.get(timeout=300)
                if item is None:
                    break
                if isinstance(item, _StreamError):
                    raise HTTPException(status_code=500, detail=item.message)
                collected.append(item)
            new_tokens = collected
            content = tokenizer.decode(new_tokens, skip_special_tokens=True) or "(无回复)"
            content_clean = _strip_think_tags(content)
            if req.session_id and req.messages:
                with _sessions_lock:
                    s = _sessions.get(req.session_id)
                    if s is not None:
                        s["messages"] = [{"role": m.role, "content": m.content} for m in req.messages]
                        s["messages"].append({"role": "assistant", "content": content_clean})
                        s["updated_at"] = _now_iso()
                        if s.get("title") == "新对话" and req.messages:
                            first = req.messages[0].content.strip()[:30]
                            if first:
                                s["title"] = first + ("…" if len(req.messages[0].content) > 30 else "")
            return {
                "id": req_state.request_id,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content_clean}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(new_tokens), "total_tokens": len(input_ids) + len(new_tokens)},
            }

        # ---------- 原有 worker 路径 ----------
        response_queue = queue.Queue()
        cancel_event = threading.Event()
        try:
            _request_queue.put_nowait({
                "response_queue": response_queue,
                "cancel_event": cancel_event,
                "session_id": req.session_id,
                "messages": req.messages,
                "stream": req.stream,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "seed": seed,
            })
        except queue.Full:
            raise HTTPException(status_code=503, detail="Request queue full, try again later")
        if req.stream:
            response_queue.put("data: " + json.dumps(
                {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]},
                ensure_ascii=False,
            ) + "\n\n")
            def stream_from_queue():
                try:
                    while True:
                        chunk = response_queue.get()
                        if chunk is _STREAM_SENTINEL:
                            return
                        if isinstance(chunk, dict):
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        yield chunk
                finally:
                    cancel_event.set()
            return StreamingResponse(
                stream_from_queue(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        resp = response_queue.get(timeout=300)
        if isinstance(resp, dict) and resp.get("__error"):
            raise HTTPException(status_code=resp.get("status_code", 500), detail=resp["__error"])
        return resp

    def _stream_chunks_generator(
        model, tokenizer, input_ids, max_tokens, temperature, top_k, top_p, seed,
        prompt: str = "",
        session_id: Optional[str] = None,
        request_messages: Optional[list[ChatMessage]] = None,
        prefix_len: int = 0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        """
        流式生成 SSE 块；供 _stream_response 与 worker 共用。yield 若干 "data: {...}\\n\\n" 字符串。
        cancel_check: 若提供且返回 True 则提前结束生成（用于客户端断开时取消推理）。
        """
        import sys
        _chat_debug = os.environ.get("LLAISYS_CHAT_DEBUG", "").strip() in ("1", "true", "yes")

        def generate():
            full_content = []
            recent_token_ids = []  # 用于退化检测
            stopped_degenerate = False  # 是否因退化输出而提前停止
            tokens = list(input_ids)
            next_id = None
            n_remaining = max_tokens
            if _chat_debug and prompt:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                print(f"[CHAT_DEBUG] prompt_len={len(prompt)} input_ids_len={len(input_ids)} prefix_len={prefix_len} tokenizer.eos_token_id={eos_id} model.end_token={model.end_token}", file=sys.stderr)
                print(f"[CHAT_DEBUG] prompt_tail(400)={repr(prompt[-400:])}", file=sys.stderr)
                try:
                    decoded_prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
                    print(f"[CHAT_DEBUG] decoded_input_ids={repr(decoded_prompt)}", file=sys.stderr)
                except Exception as e:
                    print(f"[CHAT_DEBUG] decode err: {e}", file=sys.stderr)
                if len(input_ids) <= 20:
                    print(f"[CHAT_DEBUG] input_ids={input_ids}", file=sys.stderr)
                else:
                    print(f"[CHAT_DEBUG] input_ids[:15]={input_ids[:15]} ... input_ids[-5:]={input_ids[-5:]}", file=sys.stderr)
            if prefix_len == 0:
                model.reset_kv_cache()
            # 每次采样使用不同 seed，避免 C 层每步用同一 seed 重设 RNG 导致重复采样同一 token（如出现 "0" 后立刻退化）
            sampling_step = 0
            if prefix_len > 0 and prefix_len < len(input_ids):
                suffix = input_ids[prefix_len:]
                next_id = model.next_token(
                    suffix,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed + sampling_step,
                )
                sampling_step += 1
                tokens.append(next_id)
                recent_token_ids.append(next_id)
                if _chat_debug:
                    print(f"[CHAT_DEBUG] token_id={next_id} end_token={model.end_token} delta={repr(tokenizer.decode([next_id], skip_special_tokens=True))}", file=sys.stderr)
                if next_id != model.end_token:
                    delta_text = tokenizer.decode([next_id], skip_special_tokens=True)
                    if delta_text:
                        full_content.append(delta_text)
                        chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                n_remaining = max_tokens - 1
            step = 0
            for _ in range(n_remaining):
                if cancel_check and cancel_check():
                    break
                if next_id == model.end_token:
                    break
                # 首步且 prefix_len==0 时必须传入完整 prompt 做 prefill，否则只传最后一个 token 做 decode
                if next_id is None and len(tokens) > 1:
                    next_id = model.next_token(
                        tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed + sampling_step,
                    )
                else:
                    next_id = model.next_token(
                        tokens[-1:] if len(tokens) > 1 else tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed + sampling_step,
                    )
                sampling_step += 1
                tokens.append(next_id)
                recent_token_ids.append(next_id)
                if len(recent_token_ids) > 30:
                    recent_token_ids.pop(0)
                if _chat_debug:
                    step += 1
                    dt = tokenizer.decode([next_id], skip_special_tokens=True)
                    print(f"[CHAT_DEBUG] step={step} token_id={next_id} end={next_id == model.end_token} delta={repr(dt)}", file=sys.stderr)
                if next_id == model.end_token:
                    break
                # 尽早检测：同一 token 连续或多次重复则视为退化，不输出当前 token 直接停
                if len(recent_token_ids) >= 2 and next_id == recent_token_ids[-2]:
                    stopped_degenerate = True
                    break
                if len(recent_token_ids) >= 5:
                    from collections import Counter
                    cnt = Counter(recent_token_ids[-6:])
                    if cnt and cnt.most_common(1)[0][1] >= 3:
                        stopped_degenerate = True
                        break
                delta_text = tokenizer.decode([next_id], skip_special_tokens=True)
                if not delta_text:
                    continue
                full_content.append(delta_text)
                # 退化输出检测：同一词重复、纯数字/换行等
                if _is_degenerate_output(full_content, recent_token_ids):
                    if _chat_debug:
                        print(f"[CHAT_DEBUG] stop: degenerate output detected", file=sys.stderr)
                    stopped_degenerate = True
                    break
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if _chat_debug:
                full_text = "".join(full_content)
                print(f"[CHAT_DEBUG] done steps={step} full_text_len={len(full_text)} full_text={repr(full_text[:500])}", file=sys.stderr)
            # 若因退化停止：仅当整段内容纯数字/空白等明显垃圾时才用 fallback 替换；否则保留已生成内容，只停止续写
            if stopped_degenerate:
                if _is_content_only_digits_and_whitespace(full_content):
                    fallback_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "choices": [{"index": 0, "delta": {"content": _DEGENERATE_FALLBACK}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(fallback_chunk, ensure_ascii=False)}\n\n"
                    full_content = [_DEGENERATE_FALLBACK]
                # 否则 full_content 保持不变，会话保存已生成的部分内容
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            if session_id and request_messages is not None:
                content = "".join(full_content) or "(无回复)"
                with _sessions_lock:
                    s = _sessions.get(session_id)
                    if s is not None:
                        s["messages"] = [{"role": m.role, "content": m.content} for m in request_messages]
                        s["messages"].append({"role": "assistant", "content": _strip_think_tags(content)})
                        s["updated_at"] = _now_iso()
                        if s.get("title") == "新对话" and request_messages:
                            first = request_messages[0].content.strip()[:30]
                            if first:
                                s["title"] = first + ("…" if len(request_messages[0].content) > 30 else "")
                _kv_pool_put(session_id, _count_user_messages(request_messages), model.export_kv_cache(), len(input_ids))

        yield from generate()

    def _stream_response(
        model, tokenizer, input_ids, max_tokens, temperature, top_k, top_p, seed,
        prompt: str = "",
        session_id: Optional[str] = None,
        request_messages: Optional[list[ChatMessage]] = None,
        prefix_len: int = 0,
    ):
        """SSE 流式响应；委托 _stream_chunks_generator 生成块。"""
        return StreamingResponse(
            _stream_chunks_generator(
                model, tokenizer, input_ids, max_tokens, temperature, top_k, top_p, seed,
                prompt=prompt,
                session_id=session_id,
                request_messages=request_messages,
                prefix_len=prefix_len,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.on_event("startup")
    def _start_request_worker():
        """启动请求处理：启用 Engine 时由其自带线程负责；否则启动单 worker 线程。"""
        if _engine is not None:
            return  # Engine 在 create_app 中已启动 _step_loop 线程
        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()
