"""
LLAISYS 交互式命令行聊天客户端

用法:
    python chat_cli.py [--server http://localhost:8000] [--session default]

内置命令:
    /quit          退出
    /new           新建对话（清空历史）
    /history       显示当前对话历史
    /sessions      列出所有本地会话
    /switch <id>   切换到指定会话
    /edit <N>      编辑第 N 条用户消息并重新生成（N 从 1 开始）
    /temp <val>    设置温度（0–2）
    /topk <val>    设置 Top-K
    /topp <val>    设置 Top-P
    /maxtok <val>  设置最大新 Token 数
"""

import argparse
import json
import sys
import os
import uuid
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    print("错误：请先安装 requests：pip install requests")
    sys.exit(1)

# prompt_toolkit 能正确处理 CJK 双宽字符，退格不会错位
try:
    from prompt_toolkit import prompt as _pt_prompt
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    _USE_PT = True
except ImportError:
    _USE_PT = False

import re as _re
_STRIP_ANSI = _re.compile(r'\x1b\[[0-9;]*[mK]')

# 清洗特殊 token 的正则
# 使用 [^\n|｜<>] 代替 [\w._-]，以匹配 ▁（U+2581，SentencePiece 词边界符）等非 ASCII 字符
_SPECIAL_TOKEN_RE = _re.compile(
    r"<\s*[|｜]\s*[^\n|｜<>]{2,}\s*[|｜]\s*>",
    _re.UNICODE,
)
# ▁ = U+2581（LOWER ONE EIGHTH BLOCK），Qwen EOS token 中使用，不是普通下划线
_EOS_TOKEN_RE = _re.compile(
    r"<\s*[|｜]\s*"
    r"(?:end[▁_\-\s]*of[▁_\-\s]*(?:sentence|text|turn)"
    r"|endoftext"
    r"|im[▁_\-\s]*end"
    r"|eot[▁_\-\s]*id"
    r")\s*[|｜]\s*>",
    _re.IGNORECASE,
)
# [^\n>]* 可匹配含 ▁ 的任意字符
_PARTIAL_TAIL_RE = _re.compile(r"<(?:\s{0,3}[|｜][^\n>]*)?$")

def _clean_reply(text: str) -> str:
    """清洗模型输出中的特殊 token 和 EOS 标记。"""
    text = _EOS_TOKEN_RE.sub("", text)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = _PARTIAL_TAIL_RE.sub("", text)
    text = text.replace("\ufffd", "")
    return text.rstrip()

def _input(prompt_ansi: str) -> str:
    """
    支持 CJK 宽字符退格的 input 封装。
    优先使用 prompt_toolkit（最佳）。
    未安装时降级为无颜色的标准 input。
    建议：pip install prompt-toolkit
    """
    if _USE_PT:
        return _pt_prompt(_PT_ANSI(prompt_ansi))
    # 降级：去掉颜色码，输出纯文本提示符
    return input(_STRIP_ANSI.sub('', prompt_ansi))


# ─────────────────────────────────────────────────────────────────────────────
# 会话数据结构
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str, title: str = "新对话"):
        self.id = session_id
        self.title = title
        self.messages: List[Dict] = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        if len(self.messages) == 1:
            self.title = content[:30]

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})


# ─────────────────────────────────────────────────────────────────────────────
# 核心聊天逻辑
# ─────────────────────────────────────────────────────────────────────────────

def stream_chat(
    server: str,
    session: Session,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    向服务器发送当前会话消息，流式打印响应，返回完整回复文本。
    """
    try:
        resp = requests.post(
            f"{server}/v1/chat/completions",
            json={
                "model": "llaisys",
                "messages": session.messages,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
                "session_id": session.id,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n[错误] 无法连接服务器 {server}，请确认服务器已启动。")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"\n[错误] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return ""

    print("\033[32mAssistant\033[0m: ", end="", flush=True)
    reply = ""
    buf = ""
    # hold-back 缓冲：缓存可能是特殊 token 不完整前缀的尾部
    _hold = ""
    # 思考标签状态跟踪
    _in_think = False

    for raw in resp.iter_content(chunk_size=None):
        if not raw:
            continue
        buf += raw.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                # 拼入 hold-back 缓冲
                _hold += delta
                
                # 跟踪 think 标签状态
                if "<think>" in _hold and not _in_think:
                    _in_think = True
                if "</think>" in _hold and _in_think:
                    _in_think = False
                
                # 检测是否包含完整 EOS 标记
                eos_m = _EOS_TOKEN_RE.search(_hold)
                if eos_m:
                    _hold = _hold[:eos_m.start()]
                # 检测尾部是否有未闭合的特殊 token 前缀
                partial_m = _PARTIAL_TAIL_RE.search(_hold)
                if partial_m:
                    safe = _hold[:partial_m.start()]
                    _hold = _hold[partial_m.start():]
                else:
                    safe = _hold
                    _hold = ""
                if safe:
                    # 清洗完整特殊 token（但保留 think 标签）
                    # 先移除 EOS 标记，保留 <think>...</think>
                    safe = _EOS_TOKEN_RE.sub("", safe)
                    # 移除其他特殊 token，但保留 <think> 和 </think>
                    safe = _SPECIAL_TOKEN_RE.sub("", safe)
                    print(safe, end="", flush=True)
                    reply += safe
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # 输出 hold-back 中残留的非特殊 token 文本
    if _hold:
        _hold = _EOS_TOKEN_RE.sub("", _hold)
        _hold = _SPECIAL_TOKEN_RE.sub("", _hold)
        _hold = _PARTIAL_TAIL_RE.sub("", _hold)
        if _hold:
            print(_hold, end="", flush=True)
            reply += _hold

    # 确保 think 标签闭合：如果流式结束时仍在 think 块内，自动闭合
    if _in_think:
        print("\n</think>", end="", flush=True)
        reply += "\n</think>"

    print()  # 换行
    return _clean_reply(reply)


def clear_server_cache(server: str, session_id: str):
    """通知服务器清空指定会话的 KV-Cache。"""
    try:
        requests.post(f"{server}/v1/sessions/{session_id}/clear", timeout=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────

def print_help(temp=0.8, topk=50, topp=0.9, maxtok=512):
    print(f"""
可用命令：
  /quit               退出程序
  /new                新建对话
  /clone [N]          新建对话并保留最近 N 轮上下文（默认 2 轮）
  /history            显示对话历史
  /sessions           列出所有会话
  /switch <id>        切换会话ﾈid 可由 /sessions 查看）
  /edit <N>           编辑第 N 条用户消息并重新生成
  /temp  <0.0–2.0>    设置 temperature（当前：{temp}）
  /topk  <1–500>      设置 Top-K（当前：{topk}）
  /topp  <0.0–1.0>    设置 Top-P（当前：{topp}）
  /maxtok <N>         设置最大新 Token 数（当前：{maxtok}）
""")


def chat_loop(server: str, default_session_id: str):
    sessions: Dict[str, Session] = {}
    current_id = default_session_id

    def get_session(sid: str) -> Session:
        if sid not in sessions:
            sessions[sid] = Session(sid)
        return sessions[sid]

    current = get_session(current_id)

    # 采样参数（运行时可调）
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    max_tokens = 512

    print(f"LLAISYS Chat CLI — 服务器: {server}")
    print("输入 /help 查看命令列表，Ctrl-C 或 /quit 退出。")
    print("提示：Enter 发送，Shift+Enter 换行（需要 prompt_toolkit 支持）\n")

    while True:
        # 提示符
        try:
            prompt_str = f"\033[34mYou\033[0m [{current.title[:18]}]: "
            user_input = _input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────────────────────────────────────

        if user_input == "/quit":
            print("再见！")
            break

        elif user_input == "/help":
            print_help(temperature, top_k, top_p, max_tokens)

        elif user_input == "/new":
            new_id = f"s_{uuid.uuid4().hex[:8]}"
            current = get_session(new_id)
            current_id = new_id
            print(f"[新建对话 {new_id}]")

        elif user_input.startswith("/clone"):
            parts_cmd = user_input.split()
            n_keep = 2
            if len(parts_cmd) > 1:
                try:
                    n_keep = int(parts_cmd[1])
                except ValueError:
                    pass
            # 保留最近 n_keep 轮对话作为示例上下文
            kept: List[Dict] = []
            rounds = 0
            for m in reversed(current.messages):
                if m["role"] == "user" and rounds >= n_keep:
                    break
                kept.insert(0, dict(m))
                if m["role"] == "user":
                    rounds += 1
            new_id = f"s_{uuid.uuid4().hex[:8]}"
            new_sess = get_session(new_id)
            new_sess.messages = kept
            new_sess.title = f"[续] {current.title[:20]}"
            current = new_sess
            current_id = new_id
            print(f"[新建对话 {new_id}，保留 {len(kept)} 条示例消息]")  # noqa: E501

        elif user_input == "/history":
            if not current.messages:
                print("[当前对话为空]")
            else:
                for i, m in enumerate(current.messages):
                    role_label = "You" if m["role"] == "user" else "AI "
                    preview = m["content"].replace("\n", " ")[:80]
                    print(f"  [{i+1}] {role_label}: {preview}")

        elif user_input == "/sessions":
            if not sessions:
                print("[暂无会话]")
            else:
                for sid, s in sessions.items():
                    marker = " ◀" if sid == current_id else ""
                    print(f"  {sid}  {s.title}{marker}")

        elif user_input.startswith("/switch "):
            sid = user_input[8:].strip()
            if sid in sessions:
                current_id = sid
                current = sessions[sid]
                # 切换会话时通知服务器清空 KV-Cache
                clear_server_cache(server, current_id)
                print(f"[切换到会话: {current.title or sid}]")
            else:
                print(f"[未找到会话 {sid}，可用: {list(sessions.keys())}]")

        elif user_input.startswith("/edit "):
            try:
                n = int(user_input[6:].strip())
            except ValueError:
                print("[用法: /edit <消息序号，从1开始>]")
                continue

            user_msgs = [(i, m) for i, m in enumerate(current.messages) if m["role"] == "user"]
            if n < 1 or n > len(user_msgs):
                print(f"[序号超范围，当前共 {len(user_msgs)} 条用户消息]")
                continue

            orig_idx, orig_msg = user_msgs[n - 1]
            print(f"  原内容: {orig_msg['content']}")
            try:
                new_content = input("  新内容: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not new_content:
                continue

            # 修改消息并截断后续历史
            current.messages[orig_idx]["content"] = new_content
            current.messages = current.messages[: orig_idx + 1]
            if n == 1:
                current.title = new_content[:30]

            # 重新生成
            reply = stream_chat(server, current, temperature, top_k, top_p, max_tokens)
            if reply:
                current.add_assistant(reply)

        elif user_input.startswith("/temp "):
            try:
                temperature = float(user_input[6:])
                print(f"[temperature = {temperature}]")
            except ValueError:
                print("[用法: /temp <0.0–2.0>]")

        elif user_input.startswith("/topk "):
            try:
                top_k = int(user_input[6:])
                print(f"[top_k = {top_k}]")
            except ValueError:
                print("[用法: /topk <整数>]")

        elif user_input.startswith("/topp "):
            try:
                top_p = float(user_input[6:])
                print(f"[top_p = {top_p}]")
            except ValueError:
                print("[用法: /topp <0.0–1.0>]")

        elif user_input.startswith("/maxtok "):
            try:
                max_tokens = int(user_input[8:])
                print(f"[max_tokens = {max_tokens}]")
            except ValueError:
                print("[用法: /maxtok <整数>]")

        elif user_input.startswith("/"):
            print(f"[未知命令: {user_input}，输入 /help 查看帮助]")

        # ── 正常聊天 ──────────────────────────────────────────────────────────

        else:
            current.add_user(user_input)
            reply = stream_chat(server, current, temperature, top_k, top_p, max_tokens)
            if reply:
                current.add_assistant(reply)
            else:
                # 发送失败，撤销用户消息
                current.messages.pop()


def main():
    parser = argparse.ArgumentParser(description="LLAISYS 交互式聊天 CLI")
    parser.add_argument(
        "--server", default="http://localhost:8000", help="服务器地址"
    )
    parser.add_argument(
        "--session", default=f"s_{uuid.uuid4().hex[:8]}", help="初始会话 ID"
    )
    args = parser.parse_args()
    chat_loop(args.server, args.session)


if __name__ == "__main__":
    main()
