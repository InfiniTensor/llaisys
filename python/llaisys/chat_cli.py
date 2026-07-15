import argparse
import json
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "HTTP client support is optional. Install with `pip install ./python[server]`."
    ) from exc


def _build_payload(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    stream: bool,
    session_id: str,
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "stream": stream,
        "session_id": session_id,
    }


def _stream_completion(client: httpx.Client, url: str, payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    with client.stream("POST", url, json=payload, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                parts.append(content)
    print()
    return "".join(parts)


def _non_stream_completion(client: httpx.Client, url: str, payload: Dict[str, Any]) -> str:
    response = client.post(url, json=payload, timeout=None)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print(content)
    return content


def _request_json(client: httpx.Client, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = client.request(method, url, json=payload, timeout=None)
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def _create_session(
    client: httpx.Client,
    base_url: str,
    session_id: str,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"session_id": session_id}
    if messages:
        payload["messages"] = messages
    return _request_json(client, "POST", f"{base_url}/v1/sessions", payload)


def _get_session(client: httpx.Client, base_url: str, session_id: str) -> Dict[str, Any]:
    return _request_json(client, "GET", f"{base_url}/v1/sessions/{session_id}")


def _list_sessions(client: httpx.Client, base_url: str) -> List[Dict[str, Any]]:
    return _request_json(client, "GET", f"{base_url}/v1/sessions").get("data", [])


def _delete_session(client: httpx.Client, base_url: str, session_id: str) -> Dict[str, Any]:
    return _request_json(client, "DELETE", f"{base_url}/v1/sessions/{session_id}")


def _ensure_session(
    client: httpx.Client,
    base_url: str,
    session_id: str,
    seed_messages: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    try:
        payload = _get_session(client, base_url, session_id)
        return session_id, payload.get("messages", [])
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
    _create_session(client, base_url, session_id, seed_messages)
    return session_id, list(seed_messages)


def _print_history(messages: List[Dict[str, str]]) -> None:
    if not messages:
        print("(empty session)")
        return
    for idx, message in enumerate(messages, start=1):
        print(f"{idx}. {message['role']}: {message['content']}")


def _print_sessions(sessions: List[Dict[str, Any]], current_session_id: Optional[str]) -> None:
    if not sessions:
        print("(no sessions)")
        return
    for session in sessions:
        marker = "*" if session["id"] == current_session_id else " "
        print(
            f"{marker} {session['id']}  "
            f"{session['title']}  "
            f"messages={session['message_count']}  "
            f"tokens={session['token_count']}"
        )


def _print_help() -> None:
    print("/help                 Show this help")
    print("/session              Show the current session id")
    print("/new [session_id]     Start a new session")
    print("/list                 List sessions on the server")
    print("/switch <session_id>  Switch to another session")
    print("/history              Show the current conversation")
    print("/regen                Regenerate the last assistant reply")
    print("/edit <idx> <text>    Edit a past user message and regenerate from there")
    print("/delete [session_id]  Delete a session")
    print("/quit                 Exit")


def _complete(
    client: httpx.Client,
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    stream: bool,
    session_id: str,
) -> str:
    payload = _build_payload(
        model,
        messages,
        max_tokens,
        top_k,
        top_p,
        temperature,
        stream,
        session_id,
    )
    return (
        _stream_completion(client, endpoint, payload)
        if stream
        else _non_stream_completion(client, endpoint, payload)
    )


def _handle_command(
    command_line: str,
    *,
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    stream: bool,
    session_id: str,
    messages: List[Dict[str, str]],
    initial_messages: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]], bool]:
    command, _, rest = command_line.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command == "/help":
        _print_help()
        return session_id, messages, False

    if command == "/session":
        print(session_id)
        return session_id, messages, False

    if command == "/list":
        _print_sessions(_list_sessions(client, base_url), session_id)
        return session_id, messages, False

    if command == "/history":
        _print_history(messages)
        return session_id, messages, False

    if command == "/new":
        next_session_id = rest or f"session-{uuid.uuid4().hex}"
        _create_session(client, base_url, next_session_id, initial_messages)
        print(f"Switched to {next_session_id}")
        return next_session_id, list(initial_messages), False

    if command == "/switch":
        if not rest:
            print("Usage: /switch <session_id>")
            return session_id, messages, False
        payload = _get_session(client, base_url, rest)
        print(f"Switched to {rest}")
        return rest, payload.get("messages", []), False

    if command == "/delete":
        target_session_id = rest or session_id
        _delete_session(client, base_url, target_session_id)
        print(f"Deleted {target_session_id}")
        if target_session_id == session_id:
            next_session_id = f"session-{uuid.uuid4().hex}"
            _create_session(client, base_url, next_session_id, initial_messages)
            print(f"Switched to {next_session_id}")
            return next_session_id, list(initial_messages), False
        return session_id, messages, False

    if command == "/regen":
        if not messages or messages[-1]["role"] != "assistant":
            print("No assistant reply to regenerate.")
            return session_id, messages, False
        next_messages = messages[:-1]
        reply = _complete(
            client,
            endpoint,
            model,
            next_messages,
            max_tokens,
            top_k,
            top_p,
            temperature,
            stream,
            session_id,
        )
        next_messages = next_messages + [{"role": "assistant", "content": reply}]
        return session_id, next_messages, False

    if command == "/edit":
        parts = rest.split(" ", 1)
        if len(parts) != 2:
            print("Usage: /edit <message_index> <new text>")
            return session_id, messages, False
        try:
            index = int(parts[0])
        except ValueError:
            print("Message index must be an integer.")
            return session_id, messages, False
        if index < 1 or index > len(messages):
            print("Message index out of range.")
            return session_id, messages, False
        if messages[index - 1]["role"] != "user":
            print("Only user messages can be edited.")
            return session_id, messages, False

        next_messages = [{"role": message["role"], "content": message["content"]} for message in messages[:index]]
        next_messages[-1]["content"] = parts[1]
        reply = _complete(
            client,
            endpoint,
            model,
            next_messages,
            max_tokens,
            top_k,
            top_p,
            temperature,
            stream,
            session_id,
        )
        next_messages.append({"role": "assistant", "content": reply})
        return session_id, next_messages, False

    if command in {"/quit", "/exit"}:
        return session_id, messages, True

    print("Unknown command. Use /help to list session commands.")
    return session_id, messages, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--model", default="llaisys-qwen2", type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--system-prompt", default=None, type=str)
    parser.add_argument("--max-tokens", default=128, type=int)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=0.8, type=float)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--session-id", default=None, type=str)
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--show-session", action="store_true")
    parser.add_argument("--create-session", action="store_true")
    parser.add_argument("--delete-session", action="store_true")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/v1/chat/completions"
    stream = not args.no_stream
    session_id = args.session_id or f"session-{uuid.uuid4().hex}"
    initial_messages: List[Dict[str, str]] = []
    if args.system_prompt:
        initial_messages.append({"role": "system", "content": args.system_prompt})

    if args.prompt is None and not sys.stdin.isatty() and not any(
        [args.list_sessions, args.show_session, args.create_session, args.delete_session]
    ):
        raise RuntimeError(
            "Interactive mode requires a TTY on stdin. "
            "Run this command directly inside the activated `llaisys` environment, "
            "or pass `--prompt` for one-shot mode."
        )

    with httpx.Client(trust_env=False) as client:
        if args.list_sessions:
            _print_sessions(_list_sessions(client, base_url), session_id)
            return

        if args.create_session:
            payload = _create_session(client, base_url, session_id, initial_messages)
            print(payload["id"])
            return

        if args.show_session:
            payload = _get_session(client, base_url, session_id)
            _print_history(payload.get("messages", []))
            return

        if args.delete_session:
            payload = _delete_session(client, base_url, session_id)
            print(json.dumps(payload, ensure_ascii=False))
            return

        session_id, messages = _ensure_session(client, base_url, session_id, initial_messages)
        prompt = args.prompt

        while True:
            if prompt is None:
                try:
                    prompt = input("user> ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    break

            if not prompt:
                break

            if prompt.startswith("/"):
                session_id, messages, should_exit = _handle_command(
                    prompt,
                    client=client,
                    base_url=base_url,
                    endpoint=endpoint,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    stream=stream,
                    session_id=session_id,
                    messages=messages,
                    initial_messages=initial_messages,
                )
                if should_exit:
                    break
                if args.prompt is not None:
                    break
                prompt = None
                continue

            messages.append({"role": "user", "content": prompt})
            reply = _complete(
                client,
                endpoint,
                args.model,
                messages,
                args.max_tokens,
                args.top_k,
                args.top_p,
                args.temperature,
                stream,
                session_id,
            )
            messages.append({"role": "assistant", "content": reply})

            if args.prompt is not None:
                break
            prompt = None


if __name__ == "__main__":
    main()
