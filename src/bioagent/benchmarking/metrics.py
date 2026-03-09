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
                "pass_rate": 0.0,
                "overall_pass_rate": 0.0,
                "mcq_accuracy": None,
                "open_ended_pass_rate": None,
                "average_score": None,
                "judge_error_rate": None,
            },
            "by_category": {},
            "by_type": {},
            "tool_usage": {},
            "score_dimensions": {},
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
    type_stats: dict[str, Counter] = defaultdict(Counter)
    tool_usage = Counter()
    score_sum = 0.0
    scored_runs = 0
    mcq_total = 0
    mcq_correct = 0
    open_ended_total = 0
    open_ended_passed = 0
    judge_error_count = 0
    score_dimension_totals: dict[str, float] = defaultdict(float)
    score_dimension_counts: dict[str, int] = defaultdict(int)

    for result in results:
        answer_status = str(result.get("answer_status", "error"))
        case_type = str(result.get("case", {}).get("type", "mcq") or "mcq")
        if answer_status in {"correct", "passed"}:
            category_status = "correct"
            counts["correct"] += 1
        elif answer_status in {"incorrect", "failed"}:
            category_status = "incorrect"
            counts["incorrect"] += 1
        elif answer_status in {"invalid_format", "ambiguous"}:
            category_status = "invalid_format"
            counts["invalid_format"] += 1
        else:
            counts["error"] += 1
            category_status = "error"

        category = str(result.get("case", {}).get("category", "general"))
        category_stats[category]["total"] += 1
        category_stats[category][category_status] += 1
        type_stats[case_type]["total"] += 1
        type_stats[case_type][category_status] += 1

        latency_sum += _safe_float(result.get("latency_seconds"))
        token_chars_sum += _safe_int(result.get("token_chars"))

        score_value = result.get("score_value")
        if isinstance(score_value, (int, float)):
            score_sum += float(score_value)
            scored_runs += 1
        score_dimensions = result.get("score_dimensions")
        if isinstance(score_dimensions, list):
            for dimension in score_dimensions:
                if not isinstance(dimension, dict):
                    continue
                name = str(dimension.get("name", "")).strip()
                if not name:
                    continue
                score_dimension_totals[name] += _safe_float(dimension.get("score"))
                score_dimension_counts[name] += 1

        if case_type == "mcq":
            mcq_total += 1
            if category_status == "correct":
                mcq_correct += 1
        elif case_type == "open_ended":
            open_ended_total += 1
            if category_status == "correct":
                open_ended_passed += 1
            if category_status == "error":
                judge_error_count += 1

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

    by_type: dict[str, Any] = {}
    for case_type, stats in sorted(type_stats.items()):
        type_total = stats["total"]
        by_type[case_type] = {
            "total": type_total,
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "invalid_format": stats["invalid_format"],
            "error": stats["error"],
            "accuracy": (stats["correct"] / type_total) if type_total else 0.0,
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
        "pass_rate": correct / total,
        "overall_pass_rate": correct / total,
        "mcq_accuracy": (mcq_correct / mcq_total) if mcq_total else None,
        "open_ended_pass_rate": (open_ended_passed / open_ended_total) if open_ended_total else None,
        "average_score": (score_sum / scored_runs) if scored_runs else None,
        "judge_error_rate": (judge_error_count / open_ended_total) if open_ended_total else None,
    }
    return {
        "summary": summary,
        "by_category": by_category,
        "by_type": by_type,
        "tool_usage": dict(tool_usage.most_common()),
        "score_dimensions": {
            name: score_dimension_totals[name] / score_dimension_counts[name]
            for name in sorted(score_dimension_totals)
            if score_dimension_counts[name]
        },
    }
