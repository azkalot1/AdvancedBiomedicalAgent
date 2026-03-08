from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import requests

from .config import BenchmarkProfile
from .dataset import BenchmarkCase
from .scoring import build_benchmark_prompt, score_case_result


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _infer_role(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct_role = payload.get("role") or payload.get("type") or payload.get("message_type")
    if isinstance(direct_role, str) and direct_role.strip():
        return direct_role.strip().lower()
    for nested_key in ("message", "chunk"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            nested_role = _infer_role(nested)
            if nested_role:
                return nested_role
    return None


def _looks_like_selector_output(text: str) -> bool:
    compact = "".join((text or "").split()).lower()
    return compact.startswith('{"tools":[')


def _should_include_stream_chunk(payload: Any, metadata: Any = None) -> bool:
    if isinstance(metadata, dict):
        node = metadata.get("langgraph_node")
        if isinstance(node, str) and node.strip().lower() not in {"", "model"}:
            return False

    role = _infer_role(payload)
    if role and any(part in role for part in ("tool", "human", "user")):
        return False

    if isinstance(payload, dict):
        for key in ("tool_calls", "tool_call_chunks", "invalid_tool_calls"):
            if isinstance(payload.get(key), list) and payload.get(key):
                return False

    return not _looks_like_selector_output(_extract_text(payload))


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


def iter_sse_events(response: requests.Response):
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


def _must_status(response: requests.Response, expected_codes: set[int], label: str) -> None:
    if response.status_code not in expected_codes:
        raise RuntimeError(f"{label} failed: {response.status_code} {response.reason} -> {response.text[:800]}")


def _extract_last_assistant_text(state_payload: dict[str, Any]) -> str:
    values = state_payload.get("values", state_payload)
    messages = values.get("messages", []) if isinstance(values, dict) else []
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        message_type = getattr(message, "type", None) or (message.get("type") if isinstance(message, dict) else None)
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if str(message_type).lower() not in {"ai", "assistant"} and str(role).lower() not in {"assistant", "ai"}:
            continue
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        text = _extract_text(content)
        if text:
            return text
    return ""


def _extract_state_tool_calls(state_payload: dict[str, Any]) -> list[str]:
    values = state_payload.get("values", state_payload)
    messages = values.get("messages", []) if isinstance(values, dict) else []
    if not isinstance(messages, list):
        return []

    tool_calls: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            if message.get("type") == "tool" and isinstance(message.get("name"), str):
                tool_calls.append(message["name"])
            for call in message.get("tool_calls", []) or []:
                if isinstance(call, dict):
                    name = call.get("name") or call.get("function", {}).get("name")
                    if isinstance(name, str) and name:
                        tool_calls.append(name)
            additional_kwargs = message.get("additional_kwargs")
            if isinstance(additional_kwargs, dict):
                for call in additional_kwargs.get("tool_calls", []) or []:
                    if isinstance(call, dict):
                        name = call.get("name") or call.get("function", {}).get("name")
                        if isinstance(name, str) and name:
                            tool_calls.append(name)
    return list(dict.fromkeys(tool_calls))


def _extract_message_usage_metadata(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        usage_metadata = message.get("usage_metadata")
        if isinstance(usage_metadata, dict):
            return usage_metadata
        nested_message = message.get("message")
        if isinstance(nested_message, dict):
            nested_usage = nested_message.get("usage_metadata")
            if isinstance(nested_usage, dict):
                return nested_usage
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return usage_metadata
    return {}


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        if isinstance(call, dict):
            function_payload = call.get("function")
            function_name = function_payload.get("name") if isinstance(function_payload, dict) else None
            normalized.append(
                {
                    "id": call.get("id"),
                    "name": call.get("name") or function_name,
                    "arguments": call.get("args") or call.get("arguments") or (
                        function_payload.get("arguments") if isinstance(function_payload, dict) else None
                    ),
                    "type": call.get("type"),
                }
            )
    return normalized


def _normalize_single_message(message: Any, index: int) -> dict[str, Any]:
    if isinstance(message, dict):
        content = message.get("content")
        text_content = _extract_text(content)
        additional_kwargs = message.get("additional_kwargs") if isinstance(message.get("additional_kwargs"), dict) else {}
        response_metadata = message.get("response_metadata") if isinstance(message.get("response_metadata"), dict) else {}
        return {
            "index": index,
            "id": message.get("id"),
            "type": message.get("type"),
            "role": message.get("role"),
            "name": message.get("name"),
            "content_text": text_content,
            "content_raw": content,
            "tool_calls": _normalize_tool_calls(message.get("tool_calls")),
            "additional_tool_calls": _normalize_tool_calls(additional_kwargs.get("tool_calls")),
            "usage_metadata": _extract_message_usage_metadata(message),
            "response_metadata": response_metadata,
            "additional_kwargs": additional_kwargs,
        }
    return {
        "index": index,
        "id": None,
        "type": getattr(message, "type", None),
        "role": getattr(message, "role", None),
        "name": getattr(message, "name", None),
        "content_text": _extract_text(getattr(message, "content", None)),
        "content_raw": getattr(message, "content", None),
        "tool_calls": _normalize_tool_calls(getattr(message, "tool_calls", None)),
        "additional_tool_calls": [],
        "usage_metadata": _extract_message_usage_metadata(message),
        "response_metadata": getattr(message, "response_metadata", None),
        "additional_kwargs": getattr(message, "additional_kwargs", None),
    }


def _extract_normalized_messages(state_payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = state_payload.get("values", state_payload)
    messages = values.get("messages", []) if isinstance(values, dict) else []
    if not isinstance(messages, list):
        return []
    return [_normalize_single_message(message, index) for index, message in enumerate(messages)]


def _extract_state_token_usage(state_payload: dict[str, Any]) -> dict[str, Any]:
    values = state_payload.get("values", state_payload)
    messages = values.get("messages", []) if isinstance(values, dict) else []
    if not isinstance(messages, list):
        return {
            "source": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    for message in reversed(messages):
        usage_metadata = _extract_message_usage_metadata(message)
        if not usage_metadata:
            continue
        input_tokens = _safe_int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens"))
        output_tokens = _safe_int(usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens"))
        total_tokens = _safe_int(usage_metadata.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "source": "usage_metadata",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    return {
        "source": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def _normalize_tool_invocations(tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invocations: dict[str, dict[str, Any]] = {}
    fallback_index = 0
    for event in tool_events:
        tool_name = str(event.get("tool_name", "")).strip()
        if not tool_name:
            continue
        invocation_id = str(event.get("invocation_id", "")).strip()
        if not invocation_id:
            invocation_id = f"{tool_name}::{fallback_index}"
            fallback_index += 1

        entry = invocations.setdefault(
            invocation_id,
            {
                "invocation_id": invocation_id,
                "tool_name": tool_name,
                "started_at": None,
                "finished_at": None,
                "statuses": [],
                "args_preview": None,
                "duration_ms": None,
                "error": None,
            },
        )
        status = str(event.get("status", "")).strip() or None
        timestamp = str(event.get("timestamp", "")).strip() or None
        if status is not None:
            entry["statuses"].append(status)
            if status in {"queued", "running"} and entry["started_at"] is None:
                entry["started_at"] = timestamp
            if status in {"success", "error"}:
                entry["finished_at"] = timestamp
        if entry["args_preview"] is None and event.get("args_preview") is not None:
            entry["args_preview"] = event.get("args_preview")
        if event.get("duration_ms") is not None:
            entry["duration_ms"] = event.get("duration_ms")
        if event.get("error") is not None:
            entry["error"] = event.get("error")
    return list(invocations.values())


def _extract_report_snapshots(tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for event in tool_events:
        if event.get("type") != "report_generated":
            continue
        report = event.get("report")
        if isinstance(report, dict):
            snapshots.append(dict(report))
    return snapshots


def _summarize_tool_events(tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    invocations = _normalize_tool_invocations(tool_events)
    by_tool: dict[str, int] = {}
    successful = 0
    failed = 0
    for invocation in invocations:
        tool_name = str(invocation["tool_name"])
        by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
        statuses = set(invocation["statuses"])
        if "success" in statuses:
            successful += 1
        if "error" in statuses:
            failed += 1

    return {
        "total_calls": len(invocations),
        "successful_calls": successful,
        "failed_calls": failed,
        "by_tool": dict(sorted(by_tool.items())),
    }


@dataclass(frozen=True)
class BenchmarkRunResult:
    payload: dict[str, Any]
    detail_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_detail_dict(self) -> dict[str, Any]:
        return dict(self.detail_payload)


class ManagedAegraServer:
    def __init__(self, profile: BenchmarkProfile) -> None:
        self.profile = profile
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "ManagedAegraServer":
        if not self.profile.server.launch_enabled:
            return self

        env = os.environ.copy()
        env.update(self.profile.model.server_env())
        env.update(self.profile.server.launch_env)

        cwd = self.profile.server.launch_cwd or str(Path(__file__).resolve().parents[3])
        self.process = subprocess.Popen(  # noqa: S603
            self.profile.server.launch_command,
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        runner = BenchmarkHttpRunner(self.profile)
        deadline = time.time() + self.profile.server.readiness_timeout_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Managed Aegra server exited before becoming ready.")
            try:
                runner.check_health()
                return self
            except Exception:
                time.sleep(1.0)
        raise RuntimeError(
            f"Managed Aegra server did not become ready within {self.profile.server.readiness_timeout_seconds} seconds."
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)


class BenchmarkHttpRunner:
    def __init__(self, profile: BenchmarkProfile) -> None:
        self.profile = profile
        self.session = requests.Session()
        if self.profile.server.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.profile.server.api_token}"})

    @property
    def base_url(self) -> str:
        return self.profile.server.base_url.rstrip("/")

    def close(self) -> None:
        self.session.close()

    def check_health(self) -> tuple[str, requests.Response]:
        paths = [
            self.profile.server.readiness_path,
            "/v1/ok",
            "/ok",
            "/ready",
            "/live",
            "/health",
        ]
        seen: set[str] = set()
        results: list[tuple[str, int]] = []
        for raw_path in paths:
            path = raw_path.strip() or "/health"
            if not path.startswith("/"):
                path = "/" + path
            if path in seen:
                continue
            seen.add(path)
            response = self.session.get(f"{self.base_url}{path}", timeout=self.profile.server.timeout_seconds)
            results.append((path, response.status_code))
            if response.status_code < 400:
                return path, response
        joined = ", ".join(f"{path}={code}" for path, code in results)
        raise RuntimeError(f"Health probe failed for all endpoints: {joined}")

    def resolve_user_id(self) -> str:
        if self.profile.server.user_id:
            return self.profile.server.user_id
        response = self.session.get(f"{self.base_url}/v1/me", timeout=self.profile.server.timeout_seconds)
        _must_status(response, {200}, "GET /v1/me")
        payload = response.json()
        user_id = payload.get("user_id") if isinstance(payload, dict) else None
        if not isinstance(user_id, str) or not user_id.strip():
            raise RuntimeError(f"GET /v1/me returned invalid user_id. payload={payload}")
        return user_id.strip()

    def create_thread(self, user_id: str) -> str:
        response = self.session.post(
            f"{self.base_url}/threads",
            json={"metadata": {"app": "bioagent-benchmark", "user_id": user_id}},
            timeout=self.profile.server.timeout_seconds,
        )
        _must_status(response, {200, 201}, "POST /threads")
        payload = response.json()
        thread_id = payload.get("thread_id") or payload.get("id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise RuntimeError(f"POST /threads did not return thread_id. payload={payload}")
        return thread_id

    def get_thread_state(self, thread_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/threads/{thread_id}/state",
            timeout=self.profile.server.timeout_seconds,
        )
        _must_status(response, {200}, "GET /threads/{id}/state")
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _stream_case(
        self,
        *,
        case: BenchmarkCase,
        thread_id: str,
        user_id: str,
    ) -> tuple[str, list[dict[str, Any]], list[str], int]:
        prompt = build_benchmark_prompt(case)
        run_id = f"benchmark_{case.case_id}_{uuid4().hex[:10]}"
        allowed_tools = case.allowed_tools or self.profile.run_policy.allowed_tools

        configurable: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "conversation_uuid": thread_id,
            "stream_tool_args": self.profile.run_policy.stream_tool_args,
            "model_name": self.profile.model.model_name,
            "benchmark_run_id": run_id,
        }
        if allowed_tools:
            configurable["benchmark_allowed_tools"] = allowed_tools
        if self.profile.run_policy.disallowed_tools:
            configurable["benchmark_disallowed_tools"] = self.profile.run_policy.disallowed_tools
        if self.profile.run_policy.max_total_tool_calls is not None:
            configurable["benchmark_max_total_tool_calls"] = self.profile.run_policy.max_total_tool_calls
        if self.profile.run_policy.max_tools_per_step is not None:
            configurable["benchmark_max_tools_per_step"] = self.profile.run_policy.max_tools_per_step

        payload = {
            "assistant_id": self.profile.server.assistant_id,
            "input": {"messages": [{"type": "human", "content": prompt}]},
            "config": {"configurable": configurable},
            "stream_mode": ["messages", "messages-tuple", "updates", "custom"],
        }
        response = self.session.post(
            f"{self.base_url}/threads/{thread_id}/runs/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=self.profile.run_policy.per_case_timeout_seconds,
            stream=True,
        )
        _must_status(response, {200}, "POST /threads/{id}/runs/stream")

        answer_chunks: list[str] = []
        tool_events: list[dict[str, Any]] = []
        report_ids: list[str] = []
        token_chars = 0

        for event_name, raw_data in iter_sse_events(response):
            if raw_data == "[DONE]" or event_name.lower() in {"done", "end"}:
                break
            try:
                event_data = json.loads(raw_data)
            except json.JSONDecodeError:
                event_data = raw_data

            mode, payload_data = _normalize_stream_event(event_name, event_data)
            if mode in {"messages", "messages-tuple", "messages_tuple", "on_chat_model_stream", "chat_model_stream"}:
                message_payload = payload_data[0] if isinstance(payload_data, list) and payload_data else payload_data
                metadata = payload_data[1] if isinstance(payload_data, list) and len(payload_data) > 1 else None
                if not _should_include_stream_chunk(message_payload, metadata):
                    continue
                text = _extract_text(message_payload)
                if text:
                    answer_chunks.append(text)
                    token_chars += len(text)
                continue

            if mode == "custom" and isinstance(payload_data, dict):
                custom_type = payload_data.get("type")
                if custom_type == "tool_status":
                    tool_events.append(payload_data)
                elif custom_type == "report_generated":
                    report = payload_data.get("report")
                    if isinstance(report, dict):
                        report_id = report.get("id") or report.get("ref_id")
                        if isinstance(report_id, str) and report_id:
                            report_ids.append(report_id)

        return "".join(answer_chunks), tool_events, report_ids, token_chars

    def run_case(self, case: BenchmarkCase) -> BenchmarkRunResult:
        started = perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        prompt = build_benchmark_prompt(case)
        result_payload: dict[str, Any] = {
            "case": case.to_dict(),
            "profile": self.profile.to_dict(),
            "prompt": prompt,
            "tool_budget": self.profile.run_policy.max_total_tool_calls,
            "run_started_at": started_at,
        }
        detail_payload: dict[str, Any] = {
            "case": case.to_dict(),
            "profile": self.profile.to_dict(),
            "prompt": prompt,
            "run_started_at": started_at,
        }
        try:
            user_id = self.resolve_user_id()
            thread_id = self.create_thread(user_id)
            streamed_answer, tool_events, report_ids, token_chars = self._stream_case(
                case=case,
                thread_id=thread_id,
                user_id=user_id,
            )
            state = self.get_thread_state(thread_id)
            latest_assistant_message_text = _extract_last_assistant_text(state).strip()
            streamed_answer_text = streamed_answer.strip()
            # Score against the latest assistant message from persisted state.
            # The streamed text may contain concatenated chunks from multiple assistant turns.
            final_answer_text = latest_assistant_message_text or streamed_answer_text
            score = score_case_result(case, final_answer_text)
            tool_summary = _summarize_tool_events(tool_events)
            token_usage = _extract_state_token_usage(state)
            normalized_messages = _extract_normalized_messages(state)
            tool_invocations = _normalize_tool_invocations(tool_events)
            report_snapshots = _extract_report_snapshots(tool_events)

            result_payload.update(
                {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "final_answer_text": final_answer_text,
                    "latest_assistant_message_text": latest_assistant_message_text,
                    "streamed_answer_text": streamed_answer_text,
                    "report_ids": report_ids,
                    "report_snapshots": report_snapshots,
                    "token_chars": token_chars,
                    "token_usage": token_usage,
                    "tool_events": tool_events,
                    "tool_invocations": tool_invocations,
                    "tool_summary": tool_summary,
                    "state_tool_calls": _extract_state_tool_calls(state),
                    **score,
                }
            )
            detail_payload.update(
                {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "final_answer_text": final_answer_text,
                    "latest_assistant_message_text": latest_assistant_message_text,
                    "streamed_answer_text": streamed_answer_text,
                    "report_ids": report_ids,
                    "report_snapshots": report_snapshots,
                    "token_chars": token_chars,
                    "token_usage": token_usage,
                    "tool_events": tool_events,
                    "tool_invocations": tool_invocations,
                    "tool_summary": tool_summary,
                    "state_tool_calls": _extract_state_tool_calls(state),
                    "thread_state": state,
                    "normalized_messages": normalized_messages,
                    **score,
                }
            )
        except Exception as exc:
            result_payload.update(
                {
                    "answer_status": "error",
                    "selected_option": None,
                    "is_correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tool_events": [],
                    "tool_invocations": [],
                    "tool_summary": {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "by_tool": {}},
                }
            )
            detail_payload.update(
                {
                    "answer_status": "error",
                    "selected_option": None,
                    "is_correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tool_events": [],
                    "tool_invocations": [],
                    "tool_summary": {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "by_tool": {}},
                }
            )
        latency_seconds = perf_counter() - started
        result_payload["latency_seconds"] = latency_seconds
        detail_payload["latency_seconds"] = latency_seconds
        return BenchmarkRunResult(result_payload, detail_payload)
