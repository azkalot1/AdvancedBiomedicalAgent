from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import requests


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "".join(_extract_text(item) for item in payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return payload["text"]
        if "content" in payload:
            return _extract_text(payload["content"])
        if "chunk" in payload:
            return _extract_text(payload["chunk"])
        if "message" in payload:
            return _extract_text(payload["message"])
    return ""


def _normalize_stream_event(event_name: str, event_data: Any) -> tuple[str, Any]:
    """
    Normalize stream events across Agent Server variants.

    Some versions emit:
      event: messages
      data: <chunk>

    Others emit:
      event: message
      data: {"event": "messages", "data": <chunk>}
    """
    mode = event_name
    payload = event_data

    if event_name == "message" and isinstance(event_data, dict):
        nested_mode = event_data.get("event") or event_data.get("mode") or event_data.get("stream_mode")
        nested_payload = event_data.get("data", event_data.get("payload"))
        if isinstance(nested_mode, str):
            mode = nested_mode
        if nested_payload is not None:
            payload = nested_payload

    return mode, payload


def _iter_sse_events(response: requests.Response):
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _as_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("threads", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _thread_id_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("thread_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _http_error_details(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text
    try:
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            err = payload["error"]
            code = str(err.get("code", "error"))
            message = str(err.get("message", "Request failed"))
            request_id = err.get("request_id")
            suffix = f" (request_id={request_id})" if request_id else ""
            return f"{response.status_code} {response.reason}: [{code}] {message}{suffix}"
        if isinstance(payload, dict) and "detail" in payload:
            return f"{response.status_code} {response.reason}: {payload['detail']}"
        return f"{response.status_code} {response.reason}: {payload}"
    except Exception:
        return f"{response.status_code} {response.reason}: {body}"


@dataclass
class ContextItem:
    id: str
    type: str
    source: str
    content: str
    token_count: int
    line_range: tuple[int, int] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "content": self.content,
            "token_count": self.token_count,
        }
        if self.line_range:
            payload["line_range"] = [self.line_range[0], self.line_range[1]]
        return payload


class LangGraphServerClient:
    def __init__(
        self,
        base_url: str,
        assistant_id: str,
        user_id: str | None,
        timeout: float = 120.0,
        stream_tool_args: bool = False,
        api_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.assistant_id = assistant_id
        self.user_id = (user_id or "").strip()
        self.timeout = timeout
        self.stream_tool_args = stream_tool_args
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token.strip()}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def health(self) -> None:
        response = self.session.get(self._url("/v1/ok"), timeout=self.timeout)
        if response.status_code >= 400:
            response = self.session.get(self._url("/ok"), timeout=self.timeout)
        response.raise_for_status()

    def resolve_user_id(self) -> str:
        response = self.session.get(self._url("/v1/me"), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("GET /v1/me returned invalid payload")
        user_id = payload.get("user_id")
        if not isinstance(user_id, str) or not user_id.strip():
            raise RuntimeError("GET /v1/me did not return user_id")
        self.user_id = user_id
        return user_id

    def create_thread(self) -> str:
        metadata: dict[str, Any] = {"app": "co-scientist"}
        if self.user_id:
            metadata["user_id"] = self.user_id
        payload = {"metadata": metadata}
        response = self.session.post(self._url("/threads"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        thread_id = _thread_id_from_payload(response.json())
        if not thread_id:
            raise RuntimeError("Server did not return thread_id")
        return thread_id

    def list_threads(self, limit: int = 50) -> list[dict[str, Any]]:
        # Compatibility across LangGraph API variants:
        # older: GET /threads?limit=...
        # newer: POST /threads/search
        response = self.session.get(self._url(f"/threads?limit={limit}"), timeout=self.timeout)
        if response.status_code < 400:
            return _as_list(response.json())
        if response.status_code != 405:
            response.raise_for_status()

        search_payload = {"limit": limit}
        response = self.session.post(
            self._url("/threads/search"),
            json=search_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _as_list(response.json())

    def get_thread_state(self, thread_id: str) -> dict[str, Any]:
        response = self.session.get(self._url(f"/threads/{thread_id}/state"), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def list_reports(self, thread_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if thread_id:
            params["thread_id"] = thread_id
        response = self.session.get(self._url("/v1/reports"), params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        if isinstance(payload, list):  # legacy compatibility
            return [item for item in payload if isinstance(item, dict)]
        return []

    def get_report(self, report_id: str) -> dict[str, Any]:
        response = self.session.get(self._url(f"/v1/reports/{report_id}"), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def get_report_content(self, report_id: str, max_chars: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if max_chars is not None:
            params["max_chars"] = max_chars
        response = self.session.get(
            self._url(f"/v1/reports/{report_id}/content"),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def delete_report(self, report_id: str) -> None:
        response = self.session.delete(self._url(f"/v1/reports/{report_id}"), timeout=self.timeout)
        response.raise_for_status()

    def stream_run(self, thread_id: str, message: str, context_items: list[ContextItem]) -> None:
        if context_items:
            context_blob = "\n\n".join(
                f"[Context:{item.source}] {item.content.strip()}"
                for item in context_items
                if item.content.strip()
            )
            effective_message = (
                "Additional context for this request:\n"
                f"{context_blob}\n\n"
                f"User message:\n{message}"
            )
        else:
            effective_message = message

        input_payload = {
            "messages": [{"type": "human", "content": effective_message}],
            "context_items": [item.to_payload() for item in context_items],
        }
        payload = {
            "assistant_id": self.assistant_id,
            "input": input_payload,
            "config": {
                "configurable": {
                    "thread_id": thread_id,
                    "conversation_uuid": thread_id,
                    "stream_tool_args": self.stream_tool_args,
                }
            },
            "stream_mode": ["messages", "messages-tuple", "updates", "custom"],
        }
        if self.user_id:
            payload["config"]["configurable"]["user_id"] = self.user_id

        with self.session.post(
            self._url(f"/threads/{thread_id}/runs/stream"),
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            print("Assistant> ", end="", flush=True)
            printed_token = False

            for event_name, raw_data in _iter_sse_events(response):
                if raw_data == "[DONE]":
                    break
                try:
                    event_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    event_data = raw_data

                mode, payload_data = _normalize_stream_event(event_name, event_data)

                if mode in {"messages", "messages-tuple", "messages_tuple"}:
                    message_payload = payload_data[0] if isinstance(payload_data, list) and payload_data else payload_data
                    token = _extract_text(message_payload)
                    if token:
                        print(token, end="", flush=True)
                        printed_token = True
                    continue

                if mode in {"on_chat_model_stream", "chat_model_stream"}:
                    token = _extract_text(payload_data)
                    if token:
                        print(token, end="", flush=True)
                        printed_token = True
                    continue

                if mode == "custom" and isinstance(payload_data, dict):
                    custom_type = payload_data.get("type")
                    if custom_type == "tool_status":
                        status = payload_data.get("status", "unknown")
                        tool_name = payload_data.get("tool_name", "unknown_tool")
                        progress = payload_data.get("progress")
                        progress_label = f" {progress}%" if progress is not None else ""
                        invocation_id = payload_data.get("invocation_id")
                        suffix = f" ({invocation_id})" if invocation_id else ""
                        duration_ms = payload_data.get("duration_ms")
                        duration_label = f" [{duration_ms} ms]" if duration_ms is not None else ""
                        print(f"\n[tool] {tool_name}: {status}{progress_label}{duration_label}{suffix}")
                        args_preview = payload_data.get("args_preview")
                        if isinstance(args_preview, dict) and status in {"queued", "running"}:
                            preview = json.dumps(args_preview, ensure_ascii=False)
                            if len(preview) > 240:
                                preview = preview[:240] + "... <truncated>"
                            print(f"       args={preview}")
                        error_text = payload_data.get("error")
                        if status == "error" and isinstance(error_text, str) and error_text:
                            print(f"       error={error_text}")
                    elif custom_type == "report_generated":
                        report = payload_data.get("report", {})
                        filename = report.get("filename", "report.md")
                        report_id = report.get("id", "unknown")
                        print(f"\n[report] generated {filename} ({report_id})")
                    elif custom_type == "context_updated":
                        items = payload_data.get("items", [])
                        count = len(items) if isinstance(items, list) else 0
                        print(f"\n[context] {count} item(s) active")
                    elif custom_type == "plot_data":
                        print("\n[plot] plot_data event received")

            if printed_token:
                print()
            else:
                fallback = ""
                try:
                    fallback = _extract_last_assistant_text(self.get_thread_state(thread_id))
                except Exception:
                    fallback = ""
                print(fallback or "[no token stream]")


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _print_help() -> None:
    print("Commands:")
    print("  /new                              Create and switch to a new thread")
    print("  /threads                          List threads")
    print("  /switch <thread_id>               Switch active thread")
    print("  /messages [full]                  Pretty-print thread messages from state")
    print("  /reports                          List persistent reports")
    print("  /load_report <report_id>          Show full markdown report")
    print("  /delete-report <report_id>        Delete a report")
    print("  /add_to_context <text>            Add manual context for next messages")
    print("  /context                          List current context items")
    print("  /clear_context                    Clear context items")
    print("  /quit                             Exit")


def _print_threads(threads: list[dict[str, Any]], current_thread_id: str) -> None:
    if not threads:
        print("No threads found.")
        return
    print("Threads:")
    for item in threads:
        thread_id = str(item.get("thread_id") or item.get("id") or "<missing_id>")
        marker = "*" if thread_id == current_thread_id else " "
        metadata = item.get("metadata")
        suffix = f" metadata={metadata}" if metadata else ""
        print(f" {marker} {thread_id}{suffix}")


def _print_reports(reports: list[dict[str, Any]]) -> None:
    if not reports:
        print("No reports found.")
        return
    print("Reports:")
    for report in reports:
        created_at = str(report.get("created_at", ""))
        status = str(report.get("status", "complete"))
        filename = str(report.get("filename", "report.md"))
        report_id = str(report.get("id", "<missing_id>"))
        print(f" - {report_id} :: {filename} [{status}] {created_at}")


def _print_context(context_items: list[ContextItem]) -> None:
    if not context_items:
        print("No context items.")
        return
    print("Context items:")
    for item in context_items:
        preview = item.content.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f" - {item.id} [{item.source}] {preview}")


def _extract_last_assistant_text(state_payload: dict[str, Any]) -> str:
    values = state_payload.get("values", state_payload)
    if not isinstance(values, dict):
        return ""
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("type", message.get("role", ""))).lower()
        if role not in {"assistant", "ai"}:
            continue
        return _extract_text(message.get("content"))
    return ""


def _extract_ref_ids(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"\[ref:\s*([^\]]+)\]", text)]


def _messages_from_state(state_payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = state_payload.get("values", state_payload)
    if not isinstance(values, dict):
        return []
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [item for item in messages if isinstance(item, dict)]


def _message_content_as_text(content: Any, *, full: bool, max_chars: int) -> str:
    text = _extract_text(content).strip()
    if text:
        if not full and len(text) > max_chars:
            return text[:max_chars] + "... [truncated]"
        return text

    if isinstance(content, (dict, list)):
        rendered = json.dumps(content, ensure_ascii=False, indent=2 if full else None)
        if not full and len(rendered) > max_chars:
            return rendered[:max_chars] + "... [truncated]"
        return rendered

    if content is None:
        return ""
    rendered = str(content)
    if not full and len(rendered) > max_chars:
        return rendered[:max_chars] + "... [truncated]"
    return rendered


def _print_state_messages(state_payload: dict[str, Any], *, full: bool = False) -> None:
    messages = _messages_from_state(state_payload)
    if not messages:
        print("No messages in thread state.")
        return

    print(f"Messages ({len(messages)}):")
    for index, message in enumerate(messages, start=1):
        role = str(message.get("type", message.get("role", "unknown"))).lower()
        msg_id = message.get("id") or message.get("message_id")
        name = message.get("name")
        header_parts = [f"[{index}] {role}"]
        meta_parts: list[str] = []
        if name:
            meta_parts.append(f"name={name}")
        if msg_id:
            meta_parts.append(f"id={msg_id}")
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            meta_parts.append(f"tool_call_id={tool_call_id}")
        if meta_parts:
            header_parts.append(f"({', '.join(meta_parts)})")
        print(" ".join(header_parts))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            print("  tool_calls:")
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_name = str(tool_call.get("name", "unknown_tool"))
                call_id = str(tool_call.get("id", "")).strip()
                args = tool_call.get("args")
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                if not full and len(args_str) > 240:
                    args_str = args_str[:240] + "... [truncated]"
                suffix = f" id={call_id}" if call_id else ""
                print(f"    - {call_name}{suffix} args={args_str}")

        content_str = _message_content_as_text(
            message.get("content"),
            full=full,
            max_chars=5000 if full else 900,
        )
        if content_str:
            print("  content:")
            for line in content_str.splitlines() or [""]:
                print(f"    {line}")
        else:
            print("  content: <empty>")


def run_repl(
    *,
    base_url: str,
    assistant_id: str,
    user_id: str | None,
    initial_thread_id: str | None = None,
    stream_tool_args: bool = False,
    api_token: str | None = None,
) -> None:
    client = LangGraphServerClient(
        base_url=base_url,
        assistant_id=assistant_id,
        user_id=user_id,
        stream_tool_args=stream_tool_args,
        api_token=api_token,
    )
    try:
        client.health()
        resolved_user = client.resolve_user_id()
    except Exception as exc:
        raise RuntimeError(f"Could not reach server at {base_url}: {exc}") from exc

    thread_id = initial_thread_id or client.create_thread()
    context_items: list[ContextItem] = []

    print("AI Co-Scientist terminal client (LangGraph Server)")
    print(f"Server: {base_url}")
    print(f"User: {resolved_user}")
    print(f"Assistant: {assistant_id}")
    print(f"Thread: {thread_id}")
    _print_help()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            try:
                if command in {"/quit", "/exit"}:
                    print("Bye.")
                    return
                if command == "/new":
                    thread_id = client.create_thread()
                    print(f"Switched to new thread: {thread_id}")
                    continue
                if command == "/threads":
                    _print_threads(client.list_threads(), thread_id)
                    continue
                if command == "/switch":
                    if not arg:
                        print("Usage: /switch <thread_id>")
                        continue
                    thread_id = arg
                    print(f"Switched to thread: {thread_id}")
                    continue
                if command == "/reports":
                    _print_reports(client.list_reports(thread_id))
                    continue
                if command == "/messages":
                    full = arg.lower() in {"full", "all", "verbose"} if arg else False
                    _print_state_messages(client.get_thread_state(thread_id), full=full)
                    continue
                if command in {"/load_report", "/report"}:
                    if not arg:
                        print("Usage: /load_report <report_id>")
                        continue
                    report = client.get_report_content(arg)
                    filename = report.get("filename", "report.md")
                    content = str(report.get("content", ""))
                    print(f"[{filename}]")
                    print(content)
                    continue
                if command == "/delete-report":
                    if not arg:
                        print("Usage: /delete-report <report_id>")
                        continue
                    client.delete_report(arg)
                    print(f"Deleted report: {arg}")
                    continue
                if command in {"/add_to_context", "/context_add"}:
                    if not arg:
                        print("Usage: /add_to_context <text>")
                        continue
                    item = ContextItem(
                        id=f"ctx_{uuid4().hex[:8]}",
                        type="manual_paste",
                        source="Manual",
                        content=arg,
                        token_count=_token_estimate(arg),
                    )
                    context_items.append(item)
                    print(f"Added context item: {item.id}")
                    continue
                if command in {"/context", "/list_context"}:
                    _print_context(context_items)
                    continue
                if command in {"/clear_context", "/context_clear"}:
                    context_items = []
                    print("Context cleared.")
                    continue
                if command in {"/help", "/h", "/?"}:
                    _print_help()
                    continue
                print(f"Unknown command: {command}")
                _print_help()
                continue
            except requests.HTTPError as exc:
                print(f"Command failed: {_http_error_details(exc)}")
                continue
            except Exception as exc:
                print(f"Command error: {exc}")
                continue

        try:
            client.stream_run(thread_id=thread_id, message=user_input, context_items=context_items)
        except requests.HTTPError as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            print(f"Request failed: {details}")
        except Exception as exc:
            print(f"Chat error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat client for LangGraph Server.")
    parser.add_argument(
        "--server-url",
        default=os.getenv("LANGGRAPH_API_URL", "http://localhost:2024"),
        help="LangGraph server base URL (default: http://localhost:2024).",
    )
    parser.add_argument(
        "--assistant-id",
        default=os.getenv("BIOAGENT_ASSISTANT_ID", "co_scientist"),
        help="Assistant/graph id (default: co_scientist).",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("BIOAGENT_USER_ID"),
        help="Optional fallback user id (server-owned identity from /v1/me takes precedence).",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("BIOAGENT_API_TOKEN"),
        help="Bearer token for authenticated API access.",
    )
    parser.add_argument(
        "--thread-id",
        default=os.getenv("BIOAGENT_THREAD_ID"),
        help="Existing thread id to resume.",
    )
    parser.add_argument(
        "--stream-tool-args",
        action="store_true",
        help="Stream sanitized tool argument previews in tool_status events.",
    )
    args = parser.parse_args()

    try:
        run_repl(
            base_url=args.server_url,
            assistant_id=args.assistant_id,
            user_id=args.user_id,
            initial_thread_id=args.thread_id,
            stream_tool_args=args.stream_tool_args,
            api_token=args.api_token,
        )
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
