#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

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


def _normalize_stream_event(event_name: str, event_data: Any) -> tuple[str, Any]:
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


def _must(response: requests.Response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise RuntimeError(
            f"{label} failed: {response.status_code} {response.reason} -> {response.text[:800]}"
        )


def _create_thread(base_url: str, user_id: str, timeout: float, session: requests.Session) -> str:
    response = session.post(
        f"{base_url}/threads",
        json={"metadata": {"app": "co-scientist", "user_id": user_id}},
        timeout=timeout,
    )
    _must(response, 200, "POST /threads")
    payload = response.json()
    thread_id = payload.get("thread_id") or payload.get("id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise RuntimeError(f"POST /threads did not return thread_id. payload={payload}")
    return thread_id


def _list_threads(base_url: str, timeout: float, session: requests.Session) -> list[dict[str, Any]]:
    response = session.get(f"{base_url}/threads?limit=20", timeout=timeout)
    if response.status_code == 405:
        response = session.post(f"{base_url}/threads/search", json={"limit": 20}, timeout=timeout)
    _must(response, 200, "List threads")
    payload = response.json()
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("threads", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _resolve_user_id(base_url: str, timeout: float, session: requests.Session) -> str:
    response = session.get(f"{base_url}/v1/me", timeout=timeout)
    _must(response, 200, "GET /v1/me")
    payload = response.json()
    user_id = payload.get("user_id") if isinstance(payload, dict) else None
    if not isinstance(user_id, str) or not user_id.strip():
        raise RuntimeError(f"GET /v1/me returned invalid user_id. payload={payload}")
    return user_id.strip()


def _stream_run(
    *,
    base_url: str,
    assistant_id: str,
    user_id: str,
    thread_id: str,
    message: str,
    timeout: float,
    session: requests.Session,
) -> dict[str, Any]:
    payload = {
        "assistant_id": assistant_id,
        "input": {
            "messages": [{"type": "human", "content": message}],
        },
        "config": {
            "configurable": {
                "user_id": user_id,
                "thread_id": thread_id,
                "conversation_uuid": thread_id,
                "stream_tool_args": True,
            }
        },
        "stream_mode": ["messages", "messages-tuple", "updates", "custom"],
    }
    response = session.post(
        f"{base_url}/threads/{thread_id}/runs/stream",
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=timeout,
        stream=True,
    )
    _must(response, 200, "POST /threads/{id}/runs/stream")

    token_chars = 0
    custom_events = 0
    tool_status_events = 0
    report_ids: list[str] = []

    for event_name, raw_data in _iter_sse_events(response):
        if raw_data == "[DONE]":
            break
        try:
            event_data = json.loads(raw_data)
        except json.JSONDecodeError:
            event_data = raw_data

        mode, payload_data = _normalize_stream_event(event_name, event_data)
        if mode in {"messages", "messages-tuple", "messages_tuple", "on_chat_model_stream", "chat_model_stream"}:
            text = _extract_text(payload_data[0] if isinstance(payload_data, list) and payload_data else payload_data)
            token_chars += len(text)
            continue

        if mode == "custom" and isinstance(payload_data, dict):
            custom_events += 1
            if payload_data.get("type") == "tool_status":
                tool_status_events += 1
            if payload_data.get("type") == "report_generated":
                report = payload_data.get("report", {})
                if isinstance(report, dict):
                    rid = report.get("id") or report.get("ref_id")
                    if isinstance(rid, str) and rid:
                        report_ids.append(rid)

    return {
        "token_chars": token_chars,
        "custom_events": custom_events,
        "tool_status_events": tool_status_events,
        "report_ids": report_ids,
    }


def _get_thread_state(base_url: str, thread_id: str, timeout: float, session: requests.Session) -> dict[str, Any]:
    response = session.get(f"{base_url}/threads/{thread_id}/state", timeout=timeout)
    _must(response, 200, "GET /threads/{id}/state")
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _list_reports(
    base_url: str,
    user_id: str,
    thread_id: str,
    timeout: float,
    session: requests.Session,
) -> list[dict[str, Any]]:
    response = session.get(
        f"{base_url}/v1/reports",
        params={"thread_id": thread_id, "limit": 50, "offset": 0},
        timeout=timeout,
    )
    _must(response, 200, "GET /v1/reports")
    payload = response.json()
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI-oriented LangGraph API smoke test.")
    parser.add_argument("--base-url", default="http://localhost:2024")
    parser.add_argument("--assistant-id", default="co_scientist")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--api-token", default=os.getenv("BIOAGENT_API_TOKEN"))
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--message",
        default="Please list what you can do briefly and use tools if needed.",
        help="Message sent through streaming run endpoint.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--require-tool-status", action="store_true")
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    if args.api_token:
        session.headers.update({"Authorization": f"Bearer {args.api_token.strip()}"})

    print(f"[1/8] GET {base_url}/v1/ok")
    health = session.get(f"{base_url}/v1/ok", timeout=args.timeout)
    if health.status_code >= 400:
        health = session.get(f"{base_url}/ok", timeout=args.timeout)
    _must(health, 200, "GET /v1/ok")
    print("  ok")

    print("[2/8] GET /v1/me")
    resolved_user_id = _resolve_user_id(base_url, args.timeout, session)
    if args.user_id and args.user_id != resolved_user_id:
        print(f"  requested user_id={args.user_id}, server resolved user_id={resolved_user_id}")
    else:
        print(f"  user_id={resolved_user_id}")

    user_id = resolved_user_id
    thread_id = args.thread_id
    if thread_id:
        print(f"[3/8] Using existing thread: {thread_id}")
    else:
        print("[3/8] POST /threads")
        thread_id = _create_thread(base_url, user_id, args.timeout, session)
        print(f"  created thread_id={thread_id}")

    print("[4/8] GET/POST threads list")
    threads = _list_threads(base_url, args.timeout, session)
    print(f"  threads returned: {len(threads)}")

    print("[5/8] POST /threads/{id}/runs/stream")
    stream_stats = _stream_run(
        base_url=base_url,
        assistant_id=args.assistant_id,
        user_id=user_id,
        thread_id=thread_id,
        message=args.message,
        timeout=args.timeout,
        session=session,
    )
    print(
        "  stream stats: "
        f"token_chars={stream_stats['token_chars']} "
        f"custom_events={stream_stats['custom_events']} "
        f"tool_status_events={stream_stats['tool_status_events']} "
        f"report_generated={len(stream_stats['report_ids'])}"
    )

    print("[6/8] GET /threads/{id}/state")
    state = _get_thread_state(base_url, thread_id, args.timeout, session)
    values = state.get("values", state) if isinstance(state, dict) else {}
    message_count = len(values.get("messages", [])) if isinstance(values, dict) and isinstance(values.get("messages"), list) else 0
    print(f"  state messages: {message_count}")

    print("[7/8] GET /v1/reports?thread_id=...")
    reports = _list_reports(base_url, user_id, thread_id, args.timeout, session)
    print(f"  reports for thread: {len(reports)}")

    print("[8/8] GET report metadata/content (if any)")
    if reports:
        report_id = str(reports[0].get("id", ""))
        meta = session.get(f"{base_url}/v1/reports/{report_id}", timeout=args.timeout)
        _must(meta, 200, "GET /v1/reports/{report_id}")
        content = session.get(
            f"{base_url}/v1/reports/{report_id}/content",
            params={"max_chars": 5000},
            timeout=args.timeout,
        )
        _must(content, 200, "GET /v1/reports/{report_id}/content")
        content_payload = content.json()
        body = content_payload.get("content", "") if isinstance(content_payload, dict) else ""
        print(f"  report_id={report_id} content_chars={len(body)}")
    else:
        print("  no reports yet")

    if args.require_tool_status and int(stream_stats["tool_status_events"]) == 0:
        raise RuntimeError("Expected at least one tool_status custom event, got none.")
    if args.require_report and len(stream_stats["report_ids"]) == 0 and len(reports) == 0:
        raise RuntimeError("Expected at least one generated report, got none.")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
