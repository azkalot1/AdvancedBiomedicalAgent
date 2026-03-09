from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from bioagent.llm_judge import LLMJudge
from bioagent.benchmarking.cli import main as benchmark_cli_main
from bioagent.benchmarking.config import BenchmarkProfile, BenchmarkRunPolicy, BenchmarkServerConfig, ModelProfile, load_profile
from bioagent.benchmarking.dataset import load_benchmark_suite
from bioagent.benchmarking.http_runner import BenchmarkHttpRunner, BenchmarkRunResult, _extract_last_assistant_text, iter_sse_events
from bioagent.benchmarking.metrics import summarize_results
from bioagent.benchmarking.scoring import parse_final_answer, score_case_result


TESTS_DIR = Path(__file__).resolve().parent


class _FakeJudgeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeJudgeModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def ainvoke(self, _messages):
        if not self._responses:
            raise AssertionError("No fake judge responses remaining.")
        return _FakeJudgeResponse(self._responses.pop(0))


def _load_prepare_curebench():
    module_path = TESTS_DIR.parent / "scripts" / "benchmarks" / "prepare_curebench_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_curebench_dataset", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CureBench preparation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_curebench


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


def test_load_benchmark_suite_mixed_sample() -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mixed")
    assert suite.name == "basic_biomed_mixed"
    assert len(suite.cases) == 3
    assert suite.cases[0].is_mcq
    assert suite.cases[1].is_open_ended
    assert suite.cases[1].reference_answer == "Inform their healthcare provider immediately and seek emergency medical care."
    assert suite.cases[1].judge["threshold"] == 0.7


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


def test_score_open_ended_case_without_rubric() -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mixed")
    judge = LLMJudge(
        _FakeJudgeModel(
            [
                json.dumps(
                    {
                        "score_value": 0.82,
                        "dimensions": [],
                        "missing_points": [],
                        "incorrect_statements": [],
                        "reasoning": "The answer recommends immediately contacting a clinician and getting emergency help.",
                    }
                )
            ]
        )
    )
    payload = score_case_result(
        suite.cases[1],
        "They should tell their care team right away and seek emergency medical care.",
        judge=judge,
    )
    assert payload["answer_status"] == "passed"
    assert payload["is_correct"] is True
    assert payload["scoring_type"] == "llm_judge_reference"
    assert payload["score_value"] == pytest.approx(0.82)
    assert payload["score_dimensions"] == []


def test_score_open_ended_case_with_rubric() -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mixed")
    judge = LLMJudge(
        _FakeJudgeModel(
            [
                json.dumps(
                    {
                        "score_value": 0.1,
                        "dimensions": [
                            {
                                "name": "clinical_correctness",
                                "score": 0.8,
                                "weight": 0.7,
                                "reasoning": "The answer correctly recommends stopping the suspected antibiotic.",
                            },
                            {
                                "name": "safety",
                                "score": 1.0,
                                "weight": 0.3,
                                "reasoning": "It clearly recommends urgent evaluation and avoids unsafe delay.",
                            },
                        ],
                        "missing_points": [],
                        "incorrect_statements": [],
                        "reasoning": "The answer is clinically sound and safety-conscious.",
                    }
                )
            ]
        )
    )
    payload = score_case_result(
        suite.cases[2],
        "Stop the antibiotic right away and seek urgent medical evaluation.",
        judge=judge,
    )
    assert payload["answer_status"] == "passed"
    assert payload["is_correct"] is True
    assert payload["scoring_type"] == "llm_judge_rubric"
    assert payload["score_value"] == pytest.approx(0.86)
    assert len(payload["score_dimensions"]) == 2


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


def test_summarize_results_open_ended_metrics() -> None:
    results = [
        {
            "case": {"category": "supportive_care", "type": "open_ended"},
            "answer_status": "passed",
            "is_correct": True,
            "score_value": 0.9,
            "score_dimensions": [
                {"name": "clinical_correctness", "score": 0.8},
                {"name": "safety", "score": 1.0},
            ],
            "latency_seconds": 1.0,
            "tool_events": [],
        },
        {
            "case": {"category": "supportive_care", "type": "open_ended"},
            "answer_status": "failed",
            "is_correct": False,
            "score_value": 0.4,
            "score_dimensions": [
                {"name": "clinical_correctness", "score": 0.4},
                {"name": "safety", "score": 0.5},
            ],
            "latency_seconds": 2.0,
            "tool_events": [],
        },
        {
            "case": {"category": "target_to_drug", "type": "mcq"},
            "answer_status": "correct",
            "is_correct": True,
            "latency_seconds": 0.5,
            "tool_events": [],
        },
    ]
    summary = summarize_results(results)
    assert summary["summary"]["pass_rate"] == pytest.approx(2 / 3)
    assert summary["summary"]["average_score"] == pytest.approx(0.65)
    assert summary["summary"]["open_ended_pass_rate"] == pytest.approx(0.5)
    assert summary["summary"]["mcq_accuracy"] == pytest.approx(1.0)
    assert summary["score_dimensions"]["clinical_correctness"] == pytest.approx(0.6)
    assert summary["score_dimensions"]["safety"] == pytest.approx(0.75)


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


def test_prepare_curebench_generates_mixed_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "curebench.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "mcq_001",
                        "question_type": "multi_choice",
                        "question": "Which option is correct?",
                        "correct_answer": "B",
                        "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                    }
                ),
                json.dumps(
                    {
                        "id": "open_001",
                        "question_type": "open_ended",
                        "question": "What should the patient do next?",
                        "correct_answer": "A",
                        "options": {"A": "Seek urgent medical care", "B": "Wait at home"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepare_curebench = _load_prepare_curebench()
    metadata = prepare_curebench(input_path, tmp_path / "prepared")

    assert metadata["counts"]["mcq"] == 1
    assert metadata["counts"]["open_ended"] == 1

    open_suite = load_benchmark_suite(tmp_path / "prepared" / "suites" / "open_ended.yaml")
    assert len(open_suite.cases) == 1
    assert open_suite.cases[0].is_open_ended
    assert open_suite.cases[0].reference_answer == "Seek urgent medical care"
    assert open_suite.cases[0].judge == {}


def test_benchmark_runner_retries_invalid_format(monkeypatch) -> None:
    suite = load_benchmark_suite(TESTS_DIR / "benchmarks" / "mcq")
    profile = BenchmarkProfile(
        name="retry_test",
        model=ModelProfile(provider="local", model_name="dummy", base_url="http://localhost:8080/v1"),
        server=BenchmarkServerConfig(base_url="http://localhost:8000"),
        run_policy=BenchmarkRunPolicy(retry_attempts=1),
    )
    runner = BenchmarkHttpRunner(profile)
    attempts = [
        BenchmarkRunResult(
            {
                "case": suite.cases[0].to_dict(),
                "answer_status": "invalid_format",
                "is_correct": False,
            },
            {
                "case": suite.cases[0].to_dict(),
                "answer_status": "invalid_format",
                "is_correct": False,
            },
        ),
        BenchmarkRunResult(
            {
                "case": suite.cases[0].to_dict(),
                "answer_status": "correct",
                "is_correct": True,
            },
            {
                "case": suite.cases[0].to_dict(),
                "answer_status": "correct",
                "is_correct": True,
            },
        ),
    ]

    monkeypatch.setattr(runner, "_run_case_once", lambda _case: attempts.pop(0))

    result = runner.run_case(suite.cases[0]).to_dict()
    assert result["answer_status"] == "correct"
    assert result["attempt_number"] == 2
    assert result["max_attempts"] == 2
    assert result["retry_count"] == 1
    assert result["retry_history"][0]["answer_status"] == "invalid_format"
