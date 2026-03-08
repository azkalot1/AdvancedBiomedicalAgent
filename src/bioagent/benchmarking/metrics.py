from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _tool_invocations_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    tool_events = result.get("tool_events")
    if not isinstance(tool_events, list):
        return []

    invocations: dict[str, dict[str, Any]] = {}
    anonymous_index = 0
    for event in tool_events:
        if not isinstance(event, dict):
            continue
        tool_name = str(event.get("tool_name", "")).strip()
        if not tool_name:
            continue
        invocation_id = str(event.get("invocation_id", "")).strip()
        if not invocation_id:
            invocation_id = f"{tool_name}::{anonymous_index}"
            anonymous_index += 1
        record = invocations.setdefault(
            invocation_id,
            {
                "invocation_id": invocation_id,
                "tool_name": tool_name,
                "statuses": [],
                "duration_ms": None,
            },
        )
        status = str(event.get("status", "")).strip()
        if status:
            record["statuses"].append(status)
        if event.get("duration_ms") is not None:
            record["duration_ms"] = event.get("duration_ms")
    return list(invocations.values())


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "summary": {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "invalid_format": 0,
                "error": 0,
                "accuracy": 0.0,
                "invalid_format_rate": 0.0,
                "error_rate": 0.0,
                "average_latency_seconds": 0.0,
                "average_tool_calls": 0.0,
                "average_successful_tool_calls": 0.0,
                "token_usage_available_runs": 0,
                "sum_input_tokens": 0,
                "sum_output_tokens": 0,
                "sum_total_tokens": 0,
                "average_input_tokens": None,
                "average_output_tokens": None,
                "average_total_tokens": None,
                "sum_streamed_output_chars": 0,
                "average_streamed_output_chars": 0.0,
            },
            "by_category": {},
            "tool_usage": {},
        }

    counts = Counter()
    latency_sum = 0.0
    tool_call_sum = 0
    successful_tool_call_sum = 0
    input_tokens_sum = 0
    output_tokens_sum = 0
    total_tokens_sum = 0
    token_usage_available = 0
    token_chars_sum = 0
    within_budget_correct = 0
    budget_applicable = 0
    category_stats: dict[str, Counter] = defaultdict(Counter)
    tool_usage = Counter()

    for result in results:
        answer_status = str(result.get("answer_status", "error"))
        if answer_status not in {"correct", "incorrect", "invalid_format", "ambiguous"}:
            counts["error"] += 1
            category_status = "error"
        else:
            category_status = "invalid_format" if answer_status == "ambiguous" else answer_status
            counts[category_status] += 1

        category = str(result.get("case", {}).get("category", "general"))
        category_stats[category]["total"] += 1
        category_stats[category][category_status] += 1

        latency_sum += _safe_float(result.get("latency_seconds"))
        token_chars_sum += _safe_int(result.get("token_chars"))

        token_usage = result.get("token_usage")
        if isinstance(token_usage, dict) and token_usage.get("source") == "usage_metadata":
            token_usage_available += 1
            input_tokens_sum += _safe_int(token_usage.get("input_tokens"))
            output_tokens_sum += _safe_int(token_usage.get("output_tokens"))
            total_tokens_sum += _safe_int(token_usage.get("total_tokens"))

        invocations = _tool_invocations_from_result(result)
        tool_call_sum += len(invocations)
        successful_calls = 0
        for invocation in invocations:
            tool_usage[str(invocation["tool_name"])] += 1
            if "success" in invocation["statuses"]:
                successful_calls += 1
        successful_tool_call_sum += successful_calls

        tool_budget = result.get("tool_budget")
        if isinstance(tool_budget, int) and tool_budget > 0:
            budget_applicable += 1
            if bool(result.get("is_correct")) and len(invocations) <= tool_budget:
                within_budget_correct += 1

    correct = counts["correct"]
    invalid_format = counts["invalid_format"]
    error_count = counts["error"]
    accuracy = correct / total

    by_category: dict[str, Any] = {}
    for category, stats in sorted(category_stats.items()):
        category_total = stats["total"]
        by_category[category] = {
            "total": category_total,
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "invalid_format": stats["invalid_format"],
            "error": stats["error"],
            "accuracy": (stats["correct"] / category_total) if category_total else 0.0,
        }

    summary = {
        "total": total,
        "correct": correct,
        "incorrect": counts["incorrect"],
        "invalid_format": invalid_format,
        "error": error_count,
        "accuracy": accuracy,
        "invalid_format_rate": invalid_format / total,
        "error_rate": error_count / total,
        "average_latency_seconds": latency_sum / total,
        "average_tool_calls": tool_call_sum / total,
        "average_successful_tool_calls": successful_tool_call_sum / total,
        "token_usage_available_runs": token_usage_available,
        "sum_input_tokens": input_tokens_sum,
        "sum_output_tokens": output_tokens_sum,
        "sum_total_tokens": total_tokens_sum,
        "average_input_tokens": (input_tokens_sum / token_usage_available) if token_usage_available else None,
        "average_output_tokens": (output_tokens_sum / token_usage_available) if token_usage_available else None,
        "average_total_tokens": (total_tokens_sum / token_usage_available) if token_usage_available else None,
        "sum_streamed_output_chars": token_chars_sum,
        "average_streamed_output_chars": token_chars_sum / total,
        "correct_within_budget_rate": (within_budget_correct / budget_applicable) if budget_applicable else None,
    }
    return {
        "summary": summary,
        "by_category": by_category,
        "tool_usage": dict(tool_usage.most_common()),
    }
