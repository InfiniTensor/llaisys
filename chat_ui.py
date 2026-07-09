#!/usr/bin/env python3
"""
LLAISYS Chat UI — Gradio 6.x  豆包风格（全屏 + 历史侧边栏）

用法:
    python chat_ui.py --server http://localhost:8000 [--port 7860]
"""

import html as _html
import json
import os
import re
import uuid
import argparse

import gradio as gr
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 本地持久化路径（JSON 文件存放对话历史）
# ─────────────────────────────────────────────────────────────────────────────
_PERSIST_DIR = os.path.join(os.path.expanduser("~"), ".llaisys")
_PERSIST_FILE = os.path.join(_PERSIST_DIR, "chat_history.json")


def _save_conversations(conversations: list, session_id: str) -> None:
    """将 conversations 和 active session_id 写入磁盘 JSON。"""
    try:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        data = {"session_id": session_id, "conversations": conversations}
        with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _load_conversations():
    """从磁盘加载 conversations 和 session_id。返回 (conversations, session_id)。"""
    try:
        if os.path.exists(_PERSIST_FILE):
            with open(_PERSIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            sid = data.get("session_id", str(uuid.uuid4()))
            if convs:
                return convs, sid
    except Exception:
        pass
    return [], str(uuid.uuid4())

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

# 匹配 special token 变体：<|xxx|> / <｜xxx｜> / < | xxx | >（允许空格）
# 使用 [^\n|｜<>] 代替 [\w._-]，以匹配 ▁（U+2581，SentencePiece 词边界符）等非 ASCII 字符
_SPECIAL_RE = re.compile(
    r"<\s*[|｜]\s*[^\n|｜<>]{2,}\s*[|｜]\s*>",
    re.UNICODE,
)

# 显式匹配常见 EOS 变体（宽松空格、中英文竖线、▁ 词边界符），作为兜底清洗
# ▁ = U+2581（LOWER ONE EIGHTH BLOCK），Qwen EOS token 中使用，不是普通下划线
_EOS_RE = re.compile(
    r"<\s*[|｜]\s*"
    r"(?:end[▁_\-\s]*of[▁_\-\s]*(?:sentence|text|turn)"
    r"|endoftext"
    r"|im[▁_\-\s]*end"
    r"|eot[▁_\-\s]*id"
    r")\s*[|｜]\s*>",
    re.IGNORECASE,
)

# 尾部不完整的特殊 token 前缀（流式残留）
# [^\n>]* 可匹配含 ▁ 的任意字符
_PARTIAL_TAIL_RE = re.compile(r"<(?:\s{0,3}[|｜][^\n>]*)?$")

def _to_text(content) -> str:
    """
    兼容 Gradio Chatbot 的多种 content 形态：
      - str
      - list[dict|str|...]
      - dict（如多模态消息片段）
    统一抽取为字符串，避免 re.sub 收到 list/dict 报错。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # 常见字段：text / content / value
                txt = item.get("text") or item.get("content") or item.get("value")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        txt = content.get("text") or content.get("content") or content.get("value")
        return txt if isinstance(txt, str) else ""
    return ""

def _clean(text) -> str:
    s = _to_text(text)
    # 1) 显式移除 EOS 标记（最优先，避免被其他正则截断后残留）
    s = _EOS_RE.sub("", s)
    # 2) 去除特殊 token 形态
    s = _SPECIAL_RE.sub("", s)
    # 3) 去除尾部不完整的特殊 token 前缀（流式残留如 "< | end_of"）
    s = _PARTIAL_TAIL_RE.sub("", s)
    # 4) 常见乱码替换符 U+FFFD
    s = s.replace("\ufffd", "")
    # 5) 过滤大多数不可见控制字符（保留换行/制表）
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    return s.strip()

def _trim_repetition(text: str, min_chunk: int = 80) -> str:
    """
    截掉文本末尾与前文重复的大段内容。

    如果末尾 sz 个字符与紧邻前方 sz 个字符完全一致，说明模型陷入了
    重复循环，移除末尾的重复副本。作为服务端检测的第二道防线。
    """
    n = len(text)
    if n < min_chunk * 2:
        return text
    for sz in range(min_chunk, min(n // 2 + 1, 501), 30):
        if text[-sz:] == text[-2 * sz : -sz]:
            return text[:-sz].rstrip()
    return text

def _normalize_think(text: str) -> str:
    """
    确保 <think>...</think> 标签完整，让 Gradio reasoning_tags 正确渲染。
    - 有 </think> 但没有 <think>：补头
    - 有 <think> 但没有 </think>（流式中途）：补尾
    - 连续多个 think 块不影响
    """
    has_open = "<think>" in text
    has_close = "</think>" in text
    if has_close and not has_open:
        text = "<think>" + text
    elif has_open and not has_close:
        text = text + "\n</think>"
    return text

def _strip_think(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = re.sub(r"<think>.*$",         "", t,    flags=re.DOTALL)
    return _clean(t)

def _build_api_messages(history: list) -> list:
    msgs = []
    for item in history:
        role    = item["role"]
        content = item["content"] if isinstance(item["content"], str) else ""
        if role == "assistant":
            content = _strip_think(content)
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 历史对话管理
# ─────────────────────────────────────────────────────────────────────────────

def _conv_title(history: list) -> str:
    for m in history:
        if m.get("role") == "user":
            text = re.sub(r"<[^>]+>", "", _clean(m.get("content", ""))).strip()
            return (text[:22] + "…") if len(text) > 22 else text
    return "新对话"

def _get_title_by_id(conversations: list, sid: str) -> str:
    for c in conversations:
        if c["id"] == sid:
            return c.get("title") or "新对话"
    return "新对话"

def _update_conversations(conversations: list, sid: str, history: list) -> list:
    convs = [dict(c) for c in conversations]
    for c in convs:
        if c["id"] == sid:
            c["messages"] = history
            if not c.get("title"):
                c["title"] = _conv_title(history)
            return convs
    convs.append({"id": sid, "title": _conv_title(history), "messages": history})
    return convs

def render_sidebar(conversations: list, current_id: str) -> str:
    items = ""
    for conv in reversed(conversations):
        cls   = "hi active" if conv["id"] == current_id else "hi"
        title = _html.escape((conv.get("title") or "新对话")[:26])
        cid   = conv["id"]
        # 使用 data 属性存储 session id
        items += (
            f'<div class="{cls}" data-sid="{cid}" onclick="window.__llaHandleClick(event, \'{cid}\')">'
            f'<svg class="hico" viewBox="0 0 16 16"><path d="M8 1.5c-3.59 0-6.5 '
            f'2.69-6.5 6s2.91 6 6.5 6a6.4 6.4 0 002.8-.64l2.7.82-.83-2.56A5.84 '
            f'5.84 0 0014.5 7.5c0-3.31-2.91-6-6.5-6z" stroke="currentColor" '
            f'stroke-width="1.2" fill="none"/>'
            f'<span class="hti">{title}</span>'
            f'</div>\n'
        )
    if not items:
        items = '<div class="hempty">暂无历史对话</div>'
    return f'<div id="hsc">{items}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# 历史对话点击处理 JS
# ─────────────────────────────────────────────────────────────────────────────

_HEAD_JS = """
(function() {
    console.log('[LLA] JS loaded');
    
    // 历史对话点击处理函数
    window.__llaHandleClick = function(event, sid) {
        event.preventDefault();
        event.stopPropagation();
        console.log('[LLA] History click:', sid);
        var input = document.querySelector('#hcb input');
        if (!input) input = document.querySelector('#hcb textarea');
        if (input) {
            input.value = sid;
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
        }
    };
    
    // 只处理 Shift+Enter 换行，Enter 交给 Gradio 原生处理
    function initKeyHandler() {
        var ta = document.querySelector('#msg-input textarea');
        if (!ta) {
            setTimeout(initKeyHandler, 100);
            return;
        }
        console.log('[LLA] Found textarea:', ta);
        
        ta.addEventListener('keydown', function(e) {
            console.log('[LLA] Keydown:', e.key, 'Shift:', e.shiftKey);
            if (e.key === 'Enter' && e.shiftKey) {
                // Shift+Enter：允许默认换行行为，不做任何处理
                console.log('[LLA] Shift+Enter - allow newline');
            }
            // Enter (without Shift): 不处理，完全交给 Gradio 原生的 msg_box.submit()
        });
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initKeyHandler);
    } else {
        initKeyHandler();
    }
})();
"""

CSS = """
/* ══ RESET & FULL PAGE ═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
html, body {
    height: 100% !important; margin: 0 !important; padding: 0 !important;
    overflow: hidden !important;
    background: #f7f8fa !important;
    font-family: -apple-system,'PingFang SC','Microsoft YaHei','Segoe UI',sans-serif !important;
}
.gradio-container {
    height: 100vh !important; max-width: 100% !important; width: 100% !important;
    padding: 0 !important; margin: 0 !important;
    background: transparent !important; overflow: hidden !important;
}
.gradio-container .main,
.gradio-container > .main > .contain,
.gradio-container > .main > .contain > .wrap {
    height: 100vh !important; max-height: 100vh !important;
    padding: 0 !important; margin: 0 !important;
    overflow: hidden !important; max-width: 100% !important;
}
footer { display: none !important; }

/* ══ OUTER ROW ═══════════════════════════════════════════════════════════════ */
#outer-row {
    display: flex !important; flex-direction: row !important;
    flex-wrap: nowrap !important; align-items: stretch !important;
    height: 100vh !important; width: 100% !important;
    gap: 0 !important; overflow: hidden !important;
}
#outer-row > * { min-width: 0 !important; }

/* ══ SIDEBAR ═════════════════════════════════════════════════════════════════ */
#sidebar {
    flex: 0 0 220px !important; width: 220px !important;
    min-width: 220px !important; max-width: 220px !important;
    height: 100vh !important;
    background: #f7f8fa !important;
    border-right: 1px solid #e8e8e8 !important;
    display: flex !important; flex-direction: column !important;
    overflow: hidden !important; padding: 0 !important; gap: 0 !important;
}
/* Sidebar header */
#sb-hdr { padding: 18px 16px 12px !important; border-bottom: 1px solid #efefef !important;
    flex-shrink: 0 !important; gap: 0 !important; }
#sb-hdr .gr-markdown p,
#sb-hdr p { font-size: 16px !important; font-weight: 700 !important;
    color: #111 !important; margin: 0 !important; }

/* Sidebar new-chat button */
#sb-new { padding: 8px 10px !important; flex-shrink: 0 !important; gap: 0 !important; }
#sb-new button {
    width: 100% !important; background: #e8faf3 !important; color: #059669 !important;
    border: 1px solid #bbf7d0 !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 600 !important; height: 36px !important;
    padding: 0 14px !important; text-align: left !important;
    transition: all 0.15s !important;
}
#sb-new button:hover { background: #d1fae5 !important; border-color: #6ee7b7 !important; }

/* History label */
#sb-lbl { padding: 10px 16px 2px !important; flex-shrink: 0 !important; gap: 0 !important; }
#sb-lbl .gr-markdown p,
#sb-lbl p { font-size: 11px !important; color: #9ca3af !important;
    font-weight: 600 !important; letter-spacing: 0.7px !important;
    text-transform: uppercase !important; margin: 0 !important; }

/* History HTML scroll area */
#sb-hist {
    flex: 1 !important; min-height: 0 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    padding: 0 8px !important;
}
#sb-hist::-webkit-scrollbar { width: 4px; }
#sb-hist::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
#sb-hist > div { display: block !important; }
#hsc { padding: 4px 0 8px !important; }

/* History items */
.hi {
    display: flex !important; align-items: center !important; gap: 7px !important;
    padding: 7px 9px !important; border-radius: 7px !important;
    cursor: pointer !important; color: #374151 !important;
    font-size: 13px !important; user-select: none !important;
    margin: 1px 0 !important; transition: background 0.12s !important;
}
.hi:hover  { background: #eef0f3 !important; }
.hi.active { background: #e8faf3 !important; color: #065f46 !important; }
.hico { width: 14px !important; height: 14px !important; flex-shrink: 0 !important;
    color: #9ca3af !important; }
.hi.active .hico { color: #09b37b !important; }
.hti  { flex: 1 !important; overflow: hidden !important;
    text-overflow: ellipsis !important; white-space: nowrap !important; }
.hempty { color: #9ca3af !important; font-size: 12px !important;
    text-align: center !important; padding: 24px 0 !important; }

/* Hidden textbox for JS click events - keep in DOM but invisible */
#hcb { opacity: 0 !important; position: absolute !important; }

/* 侧边栏 & 右侧参数面板：不被全局 overflow:hidden 截断 */
#sidebar .wrap, #sidebar .contain,
#rp-col .wrap, #rp-col .contain {
    height: auto !important; max-height: none !important;
    overflow: visible !important;
}
#rp-col {
    overflow-y: auto !important;
}

/* Sidebar bottom */
#sb-bot { border-top: 1px solid #efefef !important; padding: 12px 16px !important;
    flex-shrink: 0 !important; gap: 0 !important; }
#sb-bot .gr-markdown p,
#sb-bot p { font-size: 12px !important; color: #6b7280 !important; margin: 0 !important; }

/* ══ MAIN CHAT COLUMN ════════════════════════════════════════════════════════ */
#main-col {
    flex: 1 1 0 !important; height: 100vh !important; min-width: 0 !important;
    display: flex !important; flex-direction: column !important;
    background: #ffffff !important; padding: 0 !important; gap: 0 !important;
    overflow: hidden !important;
}
/* Chat topbar */
#ct-bar { height: 52px !important; min-height: 52px !important; flex-shrink: 0 !important;
    border-bottom: 1px solid #f0f0f0 !important; padding: 0 24px !important;
    display: flex !important; align-items: center !important;
    background: #fff !important; gap: 0 !important; }
#ct-bar .gr-markdown p,
#ct-bar p { font-size: 15px !important; font-weight: 600 !important;
    color: #111 !important; margin: 0 !important; }

/* Chatbot fills remaining height */
#chatbot-box {
    flex: 1 !important; min-height: 0 !important;
    border: none !important; box-shadow: none !important;
    background: #fafbfc !important;
}
/* Override Gradio's inline height */
#chatbot-box > div { height: 100% !important; }
#chatbot-box .bubble-wrap { max-height: none !important; }

/* Message bubbles */
#chatbot-box .message-wrap { padding: 6px 24px !important; }
#chatbot-box .message.user .bubble-wrap,
#chatbot-box .user .bubble-wrap {
    background: #f0fdf7 !important; border: 1px solid #c6f0dc !important;
    border-radius: 14px 14px 4px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
#chatbot-box .message.bot .bubble-wrap,
#chatbot-box .bot .bubble-wrap,
#chatbot-box .message.assistant .bubble-wrap {
    background: #ffffff !important; border: 1px solid #eaecef !important;
    border-radius: 14px 14px 14px 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ══ INPUT ZONE ══════════════════════════════════════════════════════════════ */
#inp-zone {
    border-top: 1px solid #f0f0f0 !important; flex-shrink: 0 !important;
    padding: 12px 24px 8px !important; background: #fff !important; gap: 10px !important;
}
#msg-input { margin: 0 !important; }
#msg-input textarea {
    border: 1.5px solid #dde1e7 !important; border-radius: 12px !important;
    background: #f9fafb !important; font-size: 14px !important;
    line-height: 1.6 !important; padding: 10px 14px !important; color: #111 !important;
    transition: border-color .18s, box-shadow .18s !important; resize: none !important;
    font-family: inherit !important;
}
#msg-input textarea:focus {
    border-color: #09b37b !important; background: #fff !important;
    box-shadow: 0 0 0 3px rgba(9,179,123,.10) !important; outline: none !important;
}
#msg-input textarea::placeholder { color: #b0b7c0 !important; }
#send-btn { margin: 0 !important; }
#send-btn button {
    background: #09b37b !important; border: none !important;
    border-radius: 11px !important; color: #fff !important;
    font-size: 14px !important; font-weight: 600 !important; min-height: 50px !important;
    box-shadow: 0 2px 8px rgba(9,179,123,.28) !important;
    transition: background .15s, box-shadow .15s, transform .12s !important;
    font-family: inherit !important;
}
#send-btn button:hover {
    background: #07a06e !important; box-shadow: 0 5px 18px rgba(9,179,123,.38) !important;
    transform: translateY(-1px) !important;
}
#send-btn button:active { transform: translateY(0) !important; }

/* Action row */
#act-row { padding: 2px 24px 12px !important; background: #fff !important;
    gap: 10px !important; flex-shrink: 0 !important; }
#act-row button {
    background: transparent !important; border: 1px solid #e5e7eb !important;
    border-radius: 8px !important; color: #6b7280 !important;
    font-size: 12.5px !important; height: 32px !important; transition: all .15s !important;
}
#clear-btn button:hover {
    border-color: #fca5a5 !important; color: #dc2626 !important;
    background: #fff8f8 !important;
}
#stop-btn button {
    border-color: #fca5a5 !important; color: #dc2626 !important;
}
#stop-btn button:hover {
    background: #fff1f2 !important; border-color: #f87171 !important;
}

/* ══ RIGHT PANEL ═════════════════════════════════════════════════════════════ */
#rp-col {
    flex: 0 0 260px !important; width: 260px !important;
    min-width: 260px !important; max-width: 260px !important;
    height: 100vh !important;
    border-left: 1px solid #f0f0f0 !important; background: #fafbfc !important;
    overflow-y: auto !important; padding: 20px 16px !important;
    flex-shrink: 0 !important; gap: 10px !important;
}
#rp-col::-webkit-scrollbar { width: 4px; }
#rp-col::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
#rp-col .gr-markdown h3 { font-size: 13px !important; font-weight: 600 !important;
    color: #374151 !important; margin: 0 0 12px !important; }
#rp-col .gr-markdown p { font-size: 13px !important; color: #4b5563 !important;
    line-height: 1.6 !important; }
#rp-col label span { font-size: 13px !important; }

/* ══ REASONING (think) BLOCK ═════════════════════════════════════════════════ */
.thinking {
    background: linear-gradient(135deg,#f0fdf8,#ecfdf5) !important;
    border: 1px solid #a7f3d0 !important; border-left: 3px solid #09b37b !important;
    border-radius: 10px !important; margin: 6px 0 10px !important;
    font-size: 13px !important; overflow: hidden !important;
}
.thinking > summary {
    padding: 9px 13px !important; color: #065f46 !important; font-weight: 500 !important;
    cursor: pointer !important; list-style: none !important; user-select: none !important;
    background: rgba(9,179,123,.05) !important;
    display: flex !important; align-items: center !important; gap: 6px !important;
}
.thinking > summary::marker,
.thinking > summary::-webkit-details-marker { display: none !important; }
.thinking > summary::before {
    content: "▶" !important; font-size: 10px !important; color: #09b37b !important;
    transition: transform .2s !important; display: inline-block !important;
}
.thinking[open] > summary::before { transform: rotate(90deg) !important; }
.thinking > summary:hover { background: rgba(9,179,123,.10) !important; }
.thinking > div, .thinking > p {
    padding: 8px 14px 11px !important; color: #1a3a2e !important;
    line-height: 1.7 !important; border-top: 1px solid #bbf7d0 !important;
}

/* ══ CODE BLOCKS ═════════════════════════════════════════════════════════════ */
code, pre { font-family: 'JetBrains Mono','Fira Code',Consolas,monospace !important; }
pre { background: #1e1e2e !important; border-radius: 8px !important;
    padding: 14px 16px !important; overflow-x: auto !important; border: none !important; }
pre code { color: #cdd6f4 !important; font-size: 13px !important; line-height: 1.6 !important; }
:not(pre) > code { background: #f1f4f8 !important; color: #d63384 !important;
    border-radius: 4px !important; padding: 2px 5px !important; font-size: 13px !important; }

/* ── Catppuccin Mocha 暗色语法高亮（覆盖 Highlight.js 默认配色） ───────── */
pre code .hljs-keyword,
pre code .token.keyword        { color: #cba6f7 !important; }  /* 紫色: if/for/def/class/import */
pre code .hljs-built_in,
pre code .token.builtin         { color: #fab387 !important; }  /* 橘色: print/len/range */
pre code .hljs-string,
pre code .token.string          { color: #a6e3a1 !important; }  /* 绿色: "hello" */
pre code .hljs-number,
pre code .token.number          { color: #fab387 !important; }  /* 橘色: 42, 3.14 */
pre code .hljs-title,
pre code .hljs-title\\.function_,
pre code .token.function        { color: #89b4fa !important; }  /* 蓝色: 函数名 */
pre code .hljs-comment,
pre code .token.comment         { color: #7f849c !important; font-style: italic !important; }
pre code .hljs-variable,
pre code .token.variable        { color: #f38ba8 !important; }  /* 粉红: 变量 */
pre code .hljs-operator,
pre code .token.operator        { color: #89dceb !important; }  /* 青色: = + - * / % */
pre code .hljs-punctuation,
pre code .token.punctuation     { color: #bac2de !important; }  /* 浅灰: () [] {} , ; */
pre code .hljs-params           { color: #cdd6f4 !important; }  /* 白色: 函数参数 */
pre code .hljs-meta,
pre code .token.decorator       { color: #f38ba8 !important; }  /* 粉红: @decorator */
pre code .hljs-literal,
pre code .token.boolean         { color: #fab387 !important; }  /* 橘色: True/False/None */
pre code .hljs-type,
pre code .hljs-name             { color: #f9e2af !important; }  /* 黄色: 类型名/标签名 */
pre code .hljs-attr,
pre code .token.attr-name       { color: #f9e2af !important; }  /* 黄色: 属性名 */
pre code .hljs-symbol           { color: #f2cdcd !important; }
pre code .hljs-selector-class   { color: #f9e2af !important; }
pre code .hljs-selector-tag     { color: #cba6f7 !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# 流式生成
# ─────────────────────────────────────────────────────────────────────────────

def respond(user_msg, history, session_id, server_url,
            temperature, top_k, top_p, max_tokens, thinking_budget, conversations):
    if not user_msg.strip():
        yield history, "", conversations, render_sidebar(conversations, session_id), \
              _get_title_by_id(conversations, session_id)
        return

    api_msgs = _build_api_messages(history)
    api_msgs.append({"role": "user", "content": user_msg})

    history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": "▌"},
    ]
    # Pre-compute sidebar (won't change during streaming)
    pre_sidebar = render_sidebar(conversations, session_id)
    pre_title   = _get_title_by_id(conversations, session_id) or "新对话"
    yield history, "", conversations, pre_sidebar, pre_title

    full_text = ""
    try:
        with requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages":        api_msgs,
                "temperature":     float(temperature),
                "top_k":           int(top_k),
                "top_p":           float(top_p),
                "max_tokens":      int(max_tokens),
                "thinking_budget": int(thinking_budget),
                "stream":          True,
                "session_id":      session_id,
            },
            stream=True, timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode() if isinstance(line, bytes) else line
                if not s.startswith("data: "):
                    continue
                payload = s[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text += delta
                        # 实时更新：确保 think 标签完整，流式显示
                        normalized_text = _normalize_think(_clean(full_text))
                        history[-1]["content"] = normalized_text
                        yield history, "", conversations, pre_sidebar, pre_title
                except Exception:
                    pass
    except Exception as exc:
        history[-1]["content"] = f"❌ 连接错误：{exc}"
        yield history, "", conversations, pre_sidebar, pre_title
        return

    # 生成完毕：确保 think 标签闭合，去除重复内容
    final_text = _normalize_think(_trim_repetition(_clean(full_text)))
    history[-1]["content"] = final_text
    # 更新历史记录和侧边栏
    updated_convs = _update_conversations(conversations, session_id, history)
    new_title     = _conv_title(history)
    new_sidebar   = render_sidebar(updated_convs, session_id)
    # 持久化到磁盘
    _save_conversations(updated_convs, session_id)
    yield history, "", updated_convs, new_sidebar, new_title


def do_clear(session_id, server_url, conversations):
    try:
        requests.post(f"{server_url}/v1/sessions/{session_id}/clear", timeout=5)
    except Exception:
        pass
    # 清空消息，保留此会话 id
    # 同时更新 conversations 中该 session 的消息
    updated_convs = _update_conversations(conversations, session_id, [])
    _save_conversations(updated_convs, session_id)
    sidebar = render_sidebar(updated_convs, session_id)
    title   = _get_title_by_id(updated_convs, session_id) or "新对话"
    return [], "", updated_convs, sidebar, title


def stop_generation(session_id: str, server_url: str):
    """通知服务器立即停止当前会话的生成。"""
    try:
        requests.post(f"{server_url}/v1/sessions/{session_id}/abort", timeout=2)
    except Exception:
        pass


def do_new_session(conversations):
    new_id  = str(uuid.uuid4())
    _save_conversations(conversations, new_id)
    sidebar = render_sidebar(conversations, new_id)
    return [], "", new_id, sidebar, "新对话"


def on_history_click(conv_id, conversations, server_url):
    """处理历史对话点击切换事件。"""
    if not conv_id:
        return [], gr.update(), gr.update(), gr.update(), ""
    
    # 查找对应的会话
    for conv in conversations:
        if conv["id"] == conv_id:
            msgs    = conv.get("messages", [])
            title   = conv.get("title") or "新对话"
            sidebar = render_sidebar(conversations, conv_id)
            # 返回：chatbot 消息，session_id, sidebar HTML, 标题，清空 hist_click
            return msgs, conv_id, sidebar, title, ""
    
    # 未找到会话，返回空
    return [], conv_id, render_sidebar(conversations, conv_id), "新对话", ""


def _restore_chat_history(conversations: list, session_id: str) -> list:
    """从 conversations 列表中恢复指定 session 的 chatbot 消息列表。"""
    for conv in conversations:
        if conv["id"] == session_id:
            return conv.get("messages", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# UI 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_ui(server_url: str) -> gr.Blocks:
    # 启动时从磁盘加载历史对话
    saved_convs, saved_sid = _load_conversations()

    with gr.Blocks(title="LLAISYS模型聊天机器人") as demo:

        # ── State ─────────────────────────────────────────────────────────
        session_id_st    = gr.State(saved_sid)
        server_url_st    = gr.State(server_url)
        conversations_st = gr.State(saved_convs)

        with gr.Row(elem_id="outer-row"):

            # ══ LEFT SIDEBAR ═══════════════════════════════════════════════
            with gr.Column(scale=0, min_width=220, elem_id="sidebar"):

                with gr.Row(elem_id="sb-hdr"):
                    gr.Markdown("🤖 **LLAISYS Chat**")

                sb_new_btn = gr.Button("✦  新对话", elem_id="sb-new")

                with gr.Row(elem_id="sb-lbl"):
                    gr.Markdown("历史对话")

                history_html = gr.HTML(
                    render_sidebar(saved_convs, saved_sid),
                    elem_id="sb-hist",
                )

                # 隐藏的 textbox，接收侧边栏 JS 点击事件（用 CSS 隐藏而不是 visible=False）
                hist_click = gr.Textbox(value="", show_label=False, elem_id="hcb")

                with gr.Row(elem_id="sb-bot"):
                    gr.Markdown("👤 &nbsp;LLAISYS User")

            # ══ MAIN CHAT COLUMN ═══════════════════════════════════════════
            with gr.Column(scale=5, elem_id="main-col"):

                # 顶部标题栏
                with gr.Row(elem_id="ct-bar"):
                    conv_title_md = gr.Markdown(
                        _get_title_by_id(saved_convs, saved_sid)
                    )

                # 聊天区
                chatbot = gr.Chatbot(
                    elem_id="chatbot-box",
                    height=600,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                    reasoning_tags=[("<think>", "</think>")],
                    value=_restore_chat_history(saved_convs, saved_sid),
                    placeholder=(
                        "<div style='text-align:center;padding:100px 20px 40px'>"
                        "<div style='font-size:52px;margin-bottom:16px;opacity:.35'>✦</div>"
                        "<p style='color:#4b5563;font-size:16px;font-weight:500;margin:0 0 8px'>"
                        "有什么可以帮助你？</p>"
                        "<p style='color:#9ca3af;font-size:13px;margin:0'>"
                        "推理过程将以可折叠的思考块展示</p>"
                        "</div>"
                    ),
                )

                # 输入行
                with gr.Row(elem_id="inp-zone"):
                    msg_box = gr.Textbox(
                        placeholder="发消息给 LLAISYS（Enter 换行，shift + Enter 发送）",
                        show_label=False, lines=2, max_lines=8,
                        scale=9, container=False, autofocus=True,
                        elem_id="msg-input",
                    )
                    send_btn = gr.Button(
                        "发 送", variant="primary", scale=1, min_width=90,
                        elem_id="send-btn",
                    )

                # 操作行
                with gr.Row(elem_id="act-row"):
                    clear_btn = gr.Button("🗑  清空对话", size="sm", elem_id="clear-btn")
                    stop_btn  = gr.Button("⏹  停止生成", size="sm", elem_id="stop-btn")

            # ══ RIGHT SETTINGS PANEL ═══════════════════════════════════════
            with gr.Column(scale=0, min_width=260, elem_id="rp-col"):
                gr.Markdown("### ⚙️  生成参数")
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1, step=0.05,
                    label="温度 Temperature", info="↑ 更随机 · ↓ 更保守",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=200, value=30, step=1, label="Top-K",
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P  核采样",
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=4096, value=2048, step=64, label="最大 Token 数",
                )
                thinking_budget = gr.Slider(
                    minimum=0, maximum=3000, value=2000, step=100,
                    label="思考限额 (think 块最多字符数)",
                    info="0 = 不限制",
                )
                gr.Markdown(
                    "<hr style='border-color:#ebebeb;margin:14px 0'/>\n\n"
                    "### 💭  思考块\n\n"
                    "点击 **Reasoning** 折叠块\n"
                    "展开 / 收起推理过程。\n\n"
                    "适用于 **DeepSeek-R1**、\n"
                    "**Qwen3** 等推理模型。"
                )
                gr.Markdown("<hr style='border-color:#ebebeb;margin:14px 0'/>")
                gr.Markdown("### 🔄  模型")
                model_dropdown = gr.Dropdown(
                    choices=["Qwen2"],
                    value="Qwen2",
                    label="当前模型",
                    interactive=False,
                    info="服务端加载的模型",
                )

        # ── 事件绑定 ────────────────────────────────────────────────────────
        # gen_outputs: [chatbot, msg_box, conversations_st, history_html, conv_title_md]
        gen_inputs  = [msg_box, chatbot, session_id_st, server_url_st,
                       temperature, top_k, top_p, max_tokens, thinking_budget,
                       conversations_st]
        gen_outputs = [chatbot, msg_box, conversations_st, history_html, conv_title_md]

        gen_event = send_btn.click(respond, inputs=gen_inputs, outputs=gen_outputs)
        
        # 绑定 Enter 键发送（Gradio 原生支持）
        msg_box.submit(respond, inputs=gen_inputs, outputs=gen_outputs)

        stop_btn.click(
            stop_generation,
            inputs=[session_id_st, server_url_st],
            outputs=[],
            cancels=[gen_event],
        )

        clear_btn.click(
            do_clear,
            inputs=[session_id_st, server_url_st, conversations_st],
            outputs=[chatbot, msg_box, conversations_st, history_html, conv_title_md],
        )

        sb_new_btn.click(
            do_new_session,
            inputs=[conversations_st],
            outputs=[chatbot, msg_box, session_id_st, history_html, conv_title_md],
        )

        # 历史对话点击切换：使用 .change() 事件（比 .input 更可靠）
        hist_click.change(
            on_history_click,
            inputs=[hist_click, conversations_st, server_url_st],
            outputs=[chatbot, session_id_st, history_html, conv_title_md, msg_box],
        )

        # 每次浏览器连接/刷新时从磁盘重新加载历史，避免刷新后丢失会话
        def _on_page_load():
            convs, sid = _load_conversations()
            history = _restore_chat_history(convs, sid)
            sidebar = render_sidebar(convs, sid)
            title = _get_title_by_id(convs, sid) or "新对话"
            return convs, sid, history, sidebar, title

        demo.load(
            _on_page_load,
            inputs=[],
            outputs=[conversations_st, session_id_st, chatbot, history_html, conv_title_md],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chat UI (Gradio 6)")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args()

    print(f"[ui] 后端: {args.server}")
    print(f"[ui] 界面: http://localhost:{args.port}")
    demo = build_ui(args.server)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        favicon_path="assets/AI.ico",   # 新增这一行
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=CSS,
        head=f'<script>{_HEAD_JS}</script>',
    )


if __name__ == "__main__":
    main()
