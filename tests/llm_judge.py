from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain.messages import HumanMessage


@dataclass
class GroundTruthCase:
    test_id: str
    tool: str
    input: dict[str, Any]
    expected_facts: dict[str, list[str]]
    source: str | None = None
    confidence: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundTruthCase":
        return cls(
            test_id=payload["id"],
            tool=payload["tool"],
            input=payload.get("input", {}),
            expected_facts=payload.get("expected_facts", {}),
            source=payload.get("source"),
            confidence=payload.get("confidence"),
        )


@dataclass
class JudgeEvaluation:
    test_id: str
    correctness_score: float
    required_facts_found: list[str]
    required_facts_missing: list[str]
    incorrect_statements: list[str]
    optional_facts_found: list[str]
    judge_reasoning: str
    raw_response: str


def load_ground_truth_cases(directory: str | Path) -> list[GroundTruthCase]:
    """Load ground-truth YAML test cases from a directory."""
    directory = Path(directory)
    cases: list[GroundTruthCase] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for payload in data.get("test_cases", []):
            cases.append(GroundTruthCase.from_dict(payload))
    return cases


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON object from a text blob."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


class LLMJudge:
    """Uses an LLM to evaluate tool output against ground truth facts."""

    JUDGE_PROMPT = """You are evaluating a biomedical search tool's output for correctness.

## Input Query
{input_description}

## Tool Output
{tool_output}

## Expected Facts (Ground Truth)
Required facts that MUST be present:
{required_facts}

Optional facts that MAY be present:
{optional_facts}

## Task
1. Check if each REQUIRED fact is correctly represented in the output.
2. Note any INCORRECT statements in the output.
3. Check which OPTIONAL facts are present.
4. Provide a correctness score from 0.0 to 1.0.

Respond in JSON format with keys:
correctness_score, required_facts_found, required_facts_missing,
incorrect_statements, optional_facts_found, reasoning
"""

    def __init__(self, model) -> None:
        self.model = model

    async def evaluate(self, test_case: GroundTruthCase, tool_output: str) -> JudgeEvaluation:
        required = test_case.expected_facts.get("required", [])
        optional = test_case.expected_facts.get("optional", [])

        prompt = self.JUDGE_PROMPT.format(
            input_description=json.dumps(test_case.input),
            tool_output=tool_output,
            required_facts="\n".join(f"- {fact}" for fact in required) or "None",
            optional_facts="\n".join(f"- {fact}" for fact in optional) or "None",
        )

        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        raw_text = getattr(response, "content", "") or ""
        parsed = _extract_json_block(raw_text)

        if not parsed:
            return JudgeEvaluation(
                test_id=test_case.test_id,
                correctness_score=0.0,
                required_facts_found=[],
                required_facts_missing=required,
                incorrect_statements=[],
                optional_facts_found=[],
                judge_reasoning="Failed to parse judge JSON response.",
                raw_response=raw_text,
            )

        return JudgeEvaluation(
            test_id=test_case.test_id,
            correctness_score=float(parsed.get("correctness_score", 0.0)),
            required_facts_found=list(parsed.get("required_facts_found", [])),
            required_facts_missing=list(parsed.get("required_facts_missing", [])),
            incorrect_statements=list(parsed.get("incorrect_statements", [])),
            optional_facts_found=list(parsed.get("optional_facts_found", [])),
            judge_reasoning=str(parsed.get("reasoning", "")),
            raw_response=raw_text,
        )
