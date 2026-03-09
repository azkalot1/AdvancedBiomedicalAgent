from __future__ import annotations

import json
import re
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain.messages import HumanMessage


DEFAULT_PASS_THRESHOLD = 0.7


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


@dataclass(frozen=True)
class AnswerJudgeResult:
    score_value: float
    passed: bool
    score_threshold: float
    score_dimensions: list[dict[str, Any]]
    missing_points: list[str]
    incorrect_statements: list[str]
    judge_reasoning: str
    raw_response: str
    parse_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_value": self.score_value,
            "passed": self.passed,
            "score_threshold": self.score_threshold,
            "score_dimensions": list(self.score_dimensions),
            "judge_missing_points": list(self.missing_points),
            "judge_incorrect_statements": list(self.incorrect_statements),
            "judge_notes": self.judge_reasoning,
            "judge_raw_response": self.raw_response,
            "judge_parse_mode": self.parse_mode,
        }


def load_ground_truth_cases(directory: str | Path) -> list[GroundTruthCase]:
    directory = Path(directory)
    cases: list[GroundTruthCase] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for payload in data.get("test_cases", []):
            cases.append(GroundTruthCase.from_dict(payload))
    return cases


def _extract_json_block(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _string_list(raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    values: list[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value:
            values.append(value)
    return values


def _normalize_rubric(raw_rubric: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rubric, list):
        return []
    rubric: list[dict[str, Any]] = []
    for raw_item in raw_rubric:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name", "")).strip()
        if not name:
            continue
        rubric.append(
            {
                "name": name,
                "description": str(raw_item.get("description", "")).strip(),
                "weight": _safe_float(raw_item.get("weight"), 1.0) or 1.0,
            }
        )
    return rubric


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "includes",
    "is",
    "label",
    "of",
    "on",
    "or",
    "section",
    "the",
    "to",
    "with",
    "used",
    "indicated",
    "patients",
}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _significant_tokens(text: str) -> list[str]:
    return [token for token in _word_tokens(text) if len(token) > 2 and token not in _STOPWORDS]


def _normalize_compact_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _section_variants(section_name: str) -> set[str]:
    base = section_name.strip().lower()
    variants = {base, base.replace("_", " "), base.replace(" ", "_"), base.replace("-", " ")}
    if "indications" in base:
        variants.update({"indications_and_usage", "indications and usage"})
    if "adverse reactions" in base:
        variants.update({"adverse_reactions"})
    return variants


def _fact_present_deterministically(fact: str, tool_output: str) -> bool:
    output_lower = tool_output.lower()
    output_tokens = set(_word_tokens(tool_output))

    section_match = re.match(r"^\s*(.+?)\s+label includes\s+(.+?)\s+section\s*$", fact, flags=re.IGNORECASE)
    if section_match:
        drug = section_match.group(1).strip().lower()
        section = section_match.group(2).strip().lower()
        has_drug = drug in output_lower
        has_section = any(variant in output_lower for variant in _section_variants(section))
        return has_drug and has_section

    indication_match = re.match(
        r"^\s*(.+?)\s+is\s+(?:indicated|used)\s+for\s+(.+?)\s*$",
        fact,
        flags=re.IGNORECASE,
    )
    if indication_match:
        drug = indication_match.group(1).strip().lower()
        condition = indication_match.group(2).strip().lower()
        if drug not in output_lower:
            return False
        condition_tokens = _significant_tokens(condition)
        if not condition_tokens:
            return condition in output_lower
        matched = sum(1 for token in condition_tokens if token in output_tokens)
        return (matched / len(condition_tokens)) >= 0.6

    structure_match = re.match(r"^\s*structure matches\s+(.+?)\s*$", fact, flags=re.IGNORECASE)
    if structure_match:
        molecule = structure_match.group(1).strip().lower()
        molecule_tokens = _significant_tokens(molecule)
        if not molecule_tokens:
            return molecule in output_lower
        matched = sum(1 for token in molecule_tokens if token in output_tokens)
        return (matched / len(molecule_tokens)) >= 0.7

    target_match = re.match(r"^\s*(.+?)\s+targets\s+(.+?)\s*$", fact, flags=re.IGNORECASE)
    if target_match:
        drug = target_match.group(1).strip().lower()
        target = target_match.group(2).strip()
        if drug not in output_lower:
            return False

        target_tokens = _significant_tokens(target)
        if target_tokens:
            matched = sum(1 for token in target_tokens if token in output_tokens)
            if (matched / len(target_tokens)) >= 0.6:
                return True

        compact_target = _normalize_compact_token(target)
        compact_output = _normalize_compact_token(tool_output)
        return bool(compact_target and compact_target in compact_output)

    pathway_match = re.match(
        r"^\s*(.+?)\s+is\s+associated\s+with\s+a\s+signaling\s+pathway\s*$",
        fact,
        flags=re.IGNORECASE,
    )
    if pathway_match:
        gene = pathway_match.group(1).strip().lower()
        has_gene = gene in output_lower
        has_pathway = ("pathway" in output_lower) or ("protein class" in output_lower)
        return has_gene and has_pathway

    fact_tokens = _significant_tokens(fact)
    if not fact_tokens:
        return fact.lower() in output_lower
    matched = sum(1 for token in fact_tokens if token in output_tokens)
    return (matched / len(fact_tokens)) >= 0.7


def _deterministic_required_checks(required_facts: list[str], tool_output: str) -> tuple[list[str], list[str]]:
    found: list[str] = []
    missing: list[str] = []
    for fact in required_facts:
        if _fact_present_deterministically(fact, tool_output):
            found.append(fact)
        else:
            missing.append(fact)
    return found, missing


def _reference_answer_overlap_score(reference_answer: str, candidate_answer: str) -> float:
    reference_tokens = _significant_tokens(reference_answer)
    if not reference_tokens:
        return 1.0 if reference_answer.strip().lower() == candidate_answer.strip().lower() else 0.0
    candidate_token_set = set(_significant_tokens(candidate_answer))
    if not candidate_token_set:
        return 0.0
    matched = sum(1 for token in reference_tokens if token in candidate_token_set)
    return matched / len(reference_tokens)


class LLMJudge:
    TOOL_JUDGE_PROMPT = """You are evaluating a biomedical search tool's output for correctness.

## Input Query
{input_description}

## Tool Output
{tool_output}

## Expected Facts (Ground Truth)
Required facts that MUST be present:
{required_facts}

Optional facts that MAY be present:
{optional_facts}

## Deterministic Required-Fact Precheck (high precision lexical check)
Required facts flagged as PRESENT before judging:
{precheck_found}

Required facts flagged as MISSING before judging:
{precheck_missing}

## Task
1. Check if each REQUIRED fact is correctly represented in the output.
2. Note any INCORRECT statements in the output.
3. Check which OPTIONAL facts are present.
4. Provide a correctness score from 0.0 to 1.0.

Return ONLY a single valid JSON object (no markdown/code fences/explanatory text) with keys:
correctness_score, required_facts_found, required_facts_missing,
incorrect_statements, optional_facts_found, reasoning
"""

    ANSWER_JUDGE_PROMPT = """You are evaluating a biomedical agent answer against a reference answer.

## Question
{question}

## Candidate Answer
{candidate_answer}

## Reference Answer
{reference_answer}

## Pass Threshold
{threshold}

## Rubric
{rubric_text}

## Task
1. Judge how well the candidate answer matches the medically correct reference answer.
2. Penalize materially incorrect, unsafe, or incomplete advice.
3. Return a normalized score between 0.0 and 1.0.
4. If a rubric is provided, score each rubric dimension separately. If no rubric is provided, return an empty dimensions list.

Return ONLY a single valid JSON object with keys:
score_value, dimensions, missing_points, incorrect_statements, reasoning

Rules for `dimensions`:
- It must be a JSON array.
- When rubric items are provided, include one item per rubric criterion with keys: name, score, weight, reasoning.
- When no rubric is provided, return an empty array.
"""

    REPAIR_PROMPT = """Reformat the following model output into STRICT JSON.
Return only JSON.

Model output:
{raw_output}
"""

    def __init__(self, model, *, default_threshold: float = DEFAULT_PASS_THRESHOLD) -> None:
        self.model = model
        self.default_threshold = default_threshold

    async def evaluate(self, test_case: GroundTruthCase, tool_output: str) -> JudgeEvaluation:
        required = test_case.expected_facts.get("required", [])
        optional = test_case.expected_facts.get("optional", [])
        pre_found, pre_missing = _deterministic_required_checks(required, tool_output)

        prompt = self.TOOL_JUDGE_PROMPT.format(
            input_description=json.dumps(test_case.input),
            tool_output=tool_output,
            required_facts="\n".join(f"- {fact}" for fact in required) or "None",
            optional_facts="\n".join(f"- {fact}" for fact in optional) or "None",
            precheck_found="\n".join(f"- {fact}" for fact in pre_found) or "None",
            precheck_missing="\n".join(f"- {fact}" for fact in pre_missing) or "None",
        )

        raw_text, parsed = await self._ainvoke_json(prompt)

        if not parsed:
            required_found = pre_found
            required_missing = [fact for fact in required if fact not in required_found]
            coverage = (len(required_found) / len(required)) if required else 1.0
            return JudgeEvaluation(
                test_id=test_case.test_id,
                correctness_score=coverage,
                required_facts_found=required_found,
                required_facts_missing=required_missing,
                incorrect_statements=[],
                optional_facts_found=[],
                judge_reasoning="Judge JSON parse failed; used deterministic required-fact fallback scoring.",
                raw_response=raw_text,
            )

        parsed_found = list(parsed.get("required_facts_found", []))
        for fact in pre_found:
            if fact not in parsed_found:
                parsed_found.append(fact)

        required_missing = [fact for fact in required if fact not in parsed_found]
        required_coverage = (len(parsed_found) / len(required)) if required else 1.0
        parsed_score = _clamp_score(parsed.get("correctness_score", 0.0))

        return JudgeEvaluation(
            test_id=test_case.test_id,
            correctness_score=max(parsed_score, required_coverage),
            required_facts_found=parsed_found,
            required_facts_missing=required_missing,
            incorrect_statements=list(parsed.get("incorrect_statements", [])),
            optional_facts_found=list(parsed.get("optional_facts_found", [])),
            judge_reasoning=str(parsed.get("reasoning", "")),
            raw_response=raw_text,
        )

    async def evaluate_answer(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        threshold: float | None = None,
        rubric: Any = None,
    ) -> AnswerJudgeResult:
        parsed_threshold = _clamp_score(self.default_threshold if threshold is None else threshold)
        normalized_rubric = _normalize_rubric(rubric)
        rubric_text = "No explicit rubric provided. Judge overall medical correctness, completeness, and safety."
        if normalized_rubric:
            rubric_text = "\n".join(
                f"- {item['name']} (weight={item['weight']}): {item['description'] or 'No description provided.'}"
                for item in normalized_rubric
            )

        prompt = self.ANSWER_JUDGE_PROMPT.format(
            question=question.strip(),
            candidate_answer=candidate_answer.strip(),
            reference_answer=reference_answer.strip(),
            threshold=parsed_threshold,
            rubric_text=rubric_text,
        )
        raw_text, parsed = await self._ainvoke_json(prompt)

        if not parsed:
            fallback_score = _reference_answer_overlap_score(reference_answer, candidate_answer)
            return AnswerJudgeResult(
                score_value=fallback_score,
                passed=fallback_score >= parsed_threshold,
                score_threshold=parsed_threshold,
                score_dimensions=[],
                missing_points=[],
                incorrect_statements=[],
                judge_reasoning="Judge JSON parse failed; used lexical overlap fallback against the reference answer.",
                raw_response=raw_text,
                parse_mode="fallback_overlap",
            )

        score_dimensions = self._normalize_answer_dimensions(parsed.get("dimensions"), normalized_rubric)
        score_value = self._compute_answer_score(parsed, score_dimensions)
        return AnswerJudgeResult(
            score_value=score_value,
            passed=score_value >= parsed_threshold,
            score_threshold=parsed_threshold,
            score_dimensions=score_dimensions,
            missing_points=_string_list(parsed.get("missing_points")),
            incorrect_statements=_string_list(parsed.get("incorrect_statements")),
            judge_reasoning=str(parsed.get("reasoning", "")).strip(),
            raw_response=raw_text,
            parse_mode="json",
        )

    def evaluate_answer_sync(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        threshold: float | None = None,
        rubric: Any = None,
    ) -> AnswerJudgeResult:
        return asyncio.run(
            self.evaluate_answer(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                threshold=threshold,
                rubric=rubric,
            )
        )

    def _normalize_answer_dimensions(
        self,
        raw_dimensions: Any,
        rubric: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not rubric or not isinstance(raw_dimensions, list):
            return []

        rubric_by_name = {item["name"]: item for item in rubric}
        dimensions: list[dict[str, Any]] = []
        for raw_item in raw_dimensions:
            if not isinstance(raw_item, dict):
                continue
            name = str(raw_item.get("name", "")).strip()
            if name not in rubric_by_name:
                continue
            spec = rubric_by_name[name]
            dimensions.append(
                {
                    "name": name,
                    "score": _clamp_score(raw_item.get("score")),
                    "weight": _safe_float(raw_item.get("weight"), spec["weight"]) or spec["weight"],
                    "reasoning": str(raw_item.get("reasoning", "")).strip(),
                }
            )
        return dimensions

    def _compute_answer_score(self, parsed: dict[str, Any], score_dimensions: list[dict[str, Any]]) -> float:
        if score_dimensions:
            total_weight = sum(max(0.0, _safe_float(item.get("weight"), 0.0)) for item in score_dimensions)
            if total_weight > 0:
                weighted_sum = sum(
                    _clamp_score(item.get("score")) * max(0.0, _safe_float(item.get("weight"), 0.0))
                    for item in score_dimensions
                )
                return weighted_sum / total_weight
        return _clamp_score(parsed.get("score_value"))

    async def _ainvoke_json(self, prompt: str) -> tuple[str, dict[str, Any] | None]:
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        raw_text = getattr(response, "content", "") or ""
        parsed = _extract_json_block(raw_text)
        if parsed:
            return raw_text, parsed

        repair_response = await self.model.ainvoke([HumanMessage(content=self.REPAIR_PROMPT.format(raw_output=raw_text))])
        repair_text = getattr(repair_response, "content", "") or ""
        repaired = _extract_json_block(repair_text)
        if repaired:
            return f"{raw_text}\n\n[REPAIRED_JSON]\n{repair_text}", repaired
        return raw_text, None

