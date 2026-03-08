from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from bioagent.benchmarking.cli import main as benchmark_cli_main
from bioagent.benchmarking.config import load_profile
from bioagent.benchmarking.dataset import load_benchmark_suite
from bioagent.benchmarking.http_runner import _extract_last_assistant_text, iter_sse_events
from bioagent.benchmarking.metrics import summarize_results
from bioagent.benchmarking.scoring import parse_final_answer, score_case_result


TESTS_DIR = Path(__file__).resolve().parent


def test_load_benchmark_suite_sample() -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mcq")
    assert suite.name == "basic_biomed_mcq"
    assert len(suite.cases) == 3
    assert suite.cases[0].correct_option in suite.cases[0].options


def test_load_profile_sample() -> None:
    profile = load_profile(TESTS_DIR / "benchmarks" / "profiles.yaml", "local_vllm_quick")
    assert profile.model.provider == "local"
    assert profile.model.base_url == "http://localhost:8080/v1"
    assert profile.server.launch_command == ["./scripts/run_aegra.sh", "dev-quick"]
    assert profile.run_policy.max_total_tool_calls == 4


def test_parse_final_answer_strict_contract() -> None:
    result = parse_final_answer("Reasoning here.\nFINAL_ANSWER: B", ["A", "B", "C", "D"])
    assert result.status == "parsed"
    assert result.selected_option == "B"


def test_parse_final_answer_common_fallback() -> None:
    result = parse_final_answer("After checking the data, the correct option is C.", ["A", "B", "C", "D"])
    assert result.status == "parsed"
    assert result.selected_option == "C"


def test_parse_final_answer_ambiguous() -> None:
    result = parse_final_answer("Answer: A\nBut maybe Answer: B", ["A", "B", "C", "D"])
    assert result.status == "ambiguous"
    assert result.selected_option is None


def test_score_case_result_invalid_format() -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mcq")
    payload = score_case_result(suite.cases[0], "I refuse to pick.")
    assert payload["answer_status"] == "invalid_format"
    assert payload["is_correct"] is False


def test_extract_last_assistant_text_uses_latest_ai_message_only() -> None:
    state_payload = {
        "values": {
            "messages": [
                {"type": "human", "content": "Question"},
                {"type": "ai", "content": "Reasoning about the problem."},
                {"type": "tool", "name": "think", "content": "internal"},
                {"type": "ai", "content": "FINAL_ANSWER: C"},
            ]
        }
    }
    assert _extract_last_assistant_text(state_payload) == "FINAL_ANSWER: C"


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = True):
        del decode_unicode
        yield from self._lines


def test_iter_sse_events_parses_multiline_payload() -> None:
    response = _FakeResponse(
        [
            "event: custom",
            'data: {"type":"tool_status",',
            'data: "status":"running"}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    events = list(iter_sse_events(response))
    assert events[0][0] == "custom"
    assert events[0][1] == '{"type":"tool_status",\n"status":"running"}'
    assert events[1] == ("message", "[DONE]")


def test_summarize_results_counts_statuses() -> None:
    results = [
        {
            "case": {"category": "alpha"},
            "answer_status": "correct",
            "is_correct": True,
            "latency_seconds": 1.2,
            "token_chars": 120,
            "token_usage": {"source": "usage_metadata", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "tool_budget": 3,
            "tool_events": [
                {"tool_name": "search_target_drugs", "invocation_id": "a", "status": "running"},
                {"tool_name": "search_target_drugs", "invocation_id": "a", "status": "success"},
            ],
        },
        {
            "case": {"category": "alpha"},
            "answer_status": "invalid_format",
            "is_correct": False,
            "latency_seconds": 0.8,
            "token_chars": 80,
            "token_usage": {"source": "unavailable", "input_tokens": None, "output_tokens": None, "total_tokens": None},
            "tool_events": [],
        },
        {
            "case": {"category": "beta"},
            "answer_status": "error",
            "is_correct": False,
            "latency_seconds": 0.5,
            "token_chars": 0,
            "tool_events": [],
        },
    ]
    summary = summarize_results(results)
    assert summary["summary"]["total"] == 3
    assert summary["summary"]["correct"] == 1
    assert summary["summary"]["invalid_format"] == 1
    assert summary["summary"]["error"] == 1
    assert summary["by_category"]["alpha"]["accuracy"] == 0.5
    assert summary["tool_usage"]["search_target_drugs"] == 1
    assert summary["summary"]["token_usage_available_runs"] == 1
    assert summary["summary"]["sum_total_tokens"] == 15
    assert summary["summary"]["average_total_tokens"] == 15.0
    assert summary["summary"]["sum_streamed_output_chars"] == 200


async def _dummy_tool(query: str) -> str:
    return f"ok:{query}"


def test_with_tool_status_streaming_emits_tool_output(monkeypatch) -> None:
    tools_module = pytest.importorskip("bioagent.agent.tools")
    structured_tool_module = pytest.importorskip("langchain_core.tools")
    StructuredTool = structured_tool_module.StructuredTool

    events: list[dict[str, object]] = []
    base_tool = StructuredTool.from_function(coroutine=_dummy_tool, name="dummy_tool", description="dummy")
    wrapped_tool = tools_module.with_tool_status_streaming(base_tool)

    monkeypatch.setattr(tools_module, "_emit_tool_status", lambda **kwargs: events.append(kwargs))

    result = asyncio.run(wrapped_tool.ainvoke({"query": "first"}))
    assert result == "ok:first"
    assert events
    assert events[0]["tool_name"] == "dummy_tool"
    assert events[0]["status"] == "running"
    assert events[-1]["status"] == "success"


def test_benchmark_cli_summarize(tmp_path: Path, capsys) -> None:
    raw_runs = tmp_path / "raw_runs.jsonl"
    raw_runs.write_text(
        json.dumps(
            {
                "case": {"category": "alpha"},
                "answer_status": "correct",
                "is_correct": True,
                "latency_seconds": 1.0,
                "tool_events": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = benchmark_cli_main(["summarize", "--input", str(raw_runs)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"correct": 1' in output
