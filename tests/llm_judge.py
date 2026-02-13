from __future__ import annotations

import json
import re
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
    # Fast path: already pure JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Common path: fenced markdown JSON
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


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "in", "includes", "is", "label",
    "of", "on", "or", "section", "the", "to", "with", "used", "indicated", "patients",
}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _significant_tokens(text: str) -> list[str]:
    return [t for t in _word_tokens(text) if len(t) > 2 and t not in _STOPWORDS]


def _normalize_compact_token(text: str) -> str:
    """Lowercase and remove non-alphanumerics for symbol-style matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _section_variants(section_name: str) -> set[str]:
    base = section_name.strip().lower()
    variants = {base, base.replace("_", " "), base.replace(" ", "_"), base.replace("-", " ")}
    # Common label convention aliases
    if "indications" in base:
        variants.update({"indications_and_usage", "indications and usage"})
    if "adverse reactions" in base:
        variants.update({"adverse_reactions"})
    return variants


def _fact_present_deterministically(fact: str, tool_output: str) -> bool:
    output_lower = tool_output.lower()
    output_tokens = set(_word_tokens(tool_output))

    # Pattern 1: "<drug> label includes <section> section"
    section_match = re.match(r"^\s*(.+?)\s+label includes\s+(.+?)\s+section\s*$", fact, flags=re.IGNORECASE)
    if section_match:
        drug = section_match.group(1).strip().lower()
        section = section_match.group(2).strip().lower()
        has_drug = drug in output_lower
        has_section = any(v in output_lower for v in _section_variants(section))
        return has_drug and has_section

    # Pattern 2: "<drug> is indicated/used for <condition>"
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
        matched = sum(1 for t in condition_tokens if t in output_tokens)
        # 60% token overlap handles variants like "Non-Small-Cell Lung" vs phrase in fact.
        return (matched / len(condition_tokens)) >= 0.6

    # Pattern 3: "Structure matches <molecule>"
    structure_match = re.match(r"^\s*structure matches\s+(.+?)\s*$", fact, flags=re.IGNORECASE)
    if structure_match:
        molecule = structure_match.group(1).strip().lower()
        molecule_tokens = _significant_tokens(molecule)
        if not molecule_tokens:
            return molecule in output_lower
        matched = sum(1 for t in molecule_tokens if t in output_tokens)
        return (matched / len(molecule_tokens)) >= 0.7

    # Pattern 4: "<drug> targets <gene/target>"
    target_match = re.match(
        r"^\s*(.+?)\s+targets\s+(.+?)\s*$",
        fact,
        flags=re.IGNORECASE,
    )
    if target_match:
        drug = target_match.group(1).strip().lower()
        target = target_match.group(2).strip()
        if drug not in output_lower:
            return False

        # First pass: token overlap
        target_tokens = _significant_tokens(target)
        if target_tokens:
            matched = sum(1 for t in target_tokens if t in output_tokens)
            if (matched / len(target_tokens)) >= 0.6:
                return True

        # Second pass: compact symbol overlap (ERBB2 ~= erbB-2)
        compact_target = _normalize_compact_token(target)
        compact_output = _normalize_compact_token(tool_output)
        return bool(compact_target and compact_target in compact_output)

    # Pattern 5: "<gene> is associated with a signaling pathway"
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

    # Generic fallback
    fact_tokens = _significant_tokens(fact)
    if not fact_tokens:
        return fact.lower() in output_lower
    matched = sum(1 for t in fact_tokens if t in output_tokens)
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

    REPAIR_PROMPT = """Reformat the following model output into STRICT JSON.
Return only JSON with keys:
correctness_score, required_facts_found, required_facts_missing, incorrect_statements, optional_facts_found, reasoning

Model output:
{raw_output}
"""

    def __init__(self, model) -> None:
        self.model = model

    async def evaluate(self, test_case: GroundTruthCase, tool_output: str) -> JudgeEvaluation:
        required = test_case.expected_facts.get("required", [])
        optional = test_case.expected_facts.get("optional", [])
        pre_found, pre_missing = _deterministic_required_checks(required, tool_output)

        prompt = self.JUDGE_PROMPT.format(
            input_description=json.dumps(test_case.input),
            tool_output=tool_output,
            required_facts="\n".join(f"- {fact}" for fact in required) or "None",
            optional_facts="\n".join(f"- {fact}" for fact in optional) or "None",
            precheck_found="\n".join(f"- {fact}" for fact in pre_found) or "None",
            precheck_missing="\n".join(f"- {fact}" for fact in pre_missing) or "None",
        )

        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        raw_text = getattr(response, "content", "") or ""
        parsed = _extract_json_block(raw_text)

        if not parsed:
            repair = self.REPAIR_PROMPT.format(raw_output=raw_text)
            repair_response = await self.model.ainvoke([HumanMessage(content=repair)])
            repair_text = getattr(repair_response, "content", "") or ""
            parsed = _extract_json_block(repair_text)
            if parsed:
                raw_text = f"{raw_text}\n\n[REPAIRED_JSON]\n{repair_text}"

        if not parsed:
            # Deterministic fallback instead of hard 0.0 on parser failures.
            required_found = pre_found
            required_missing = [f for f in required if f not in required_found]
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
        # Merge deterministic required-fact positives to avoid false negatives from judge formatting drift.
        for fact in pre_found:
            if fact not in parsed_found:
                parsed_found.append(fact)

        required_missing = [fact for fact in required if fact not in parsed_found]
        required_coverage = (len(parsed_found) / len(required)) if required else 1.0
        parsed_score = float(parsed.get("correctness_score", 0.0))

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
