from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bioagent.llm_judge import DEFAULT_PASS_THRESHOLD, LLMJudge
from .dataset import BenchmarkCase


FINAL_ANSWER_PREFIX = "FINAL_ANSWER:"


@dataclass(frozen=True)
class AnswerExtractionResult:
    status: Literal["parsed", "invalid_format", "ambiguous"]
    selected_option: str | None
    matched_text: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "selected_option": self.selected_option,
            "matched_text": self.matched_text,
            "reason": self.reason,
        }


def build_benchmark_prompt(case: BenchmarkCase) -> str:
    if case.is_open_ended:
        return build_open_ended_prompt(case)
    option_lines = [f"{letter}. {text}" for letter, text in case.options.items()]
    options_block = "\n".join(option_lines)
    return (
        "Answer the following biomedical multiple-choice question.\n"
        "You may use tools if needed.\n"
        "Reason however you want, but your final line must be exactly in this format:\n"
        "FINAL_ANSWER: <LETTER>\n"
        "Do not include anything after the final answer line.\n\n"
        f"Question:\n{case.question}\n\n"
        f"Options:\n{options_block}"
    )


def build_open_ended_prompt(case: BenchmarkCase) -> str:
    return (
        "Answer the following biomedical question.\n"
        "You may use tools if needed.\n"
        "Provide a concise, directly useful answer grounded in the evidence you find.\n"
        "Do not use multiple-choice notation unless the question explicitly asks for it.\n\n"
        f"Question:\n{case.question}"
    )


def _candidate_matches(text: str, option_letters: set[str]) -> list[tuple[str, str]]:
    option_pattern = "|".join(sorted(option_letters))
    patterns = [
        re.compile(rf"^\s*{re.escape(FINAL_ANSWER_PREFIX)}\s*({option_pattern})\s*$", flags=re.IGNORECASE | re.MULTILINE),
        re.compile(rf"^\s*answer\s*[:\-]?\s*({option_pattern})\s*$", flags=re.IGNORECASE | re.MULTILINE),
        re.compile(rf"^\s*option\s*[:\-]?\s*({option_pattern})\s*$", flags=re.IGNORECASE | re.MULTILINE),
        re.compile(
            rf"\b(?:the\s+correct\s+answer|the\s+correct\s+option|correct\s+answer|correct\s+option)\s*(?:is|:)?\s*({option_pattern})\b",
            flags=re.IGNORECASE,
        ),
    ]

    candidates: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidates.append((match.group(1).upper(), match.group(0)))
    return candidates


def parse_final_answer(text: str, option_letters: list[str] | tuple[str, ...]) -> AnswerExtractionResult:
    letters = {letter.strip().upper() for letter in option_letters if letter.strip()}
    if not letters:
        raise ValueError("parse_final_answer requires at least one valid option letter.")

    stripped = text.strip()
    if not stripped:
        return AnswerExtractionResult(
            status="invalid_format",
            selected_option=None,
            matched_text=None,
            reason="Assistant produced no final answer text.",
        )

    if stripped.upper() in letters:
        return AnswerExtractionResult(
            status="parsed",
            selected_option=stripped.upper(),
            matched_text=stripped,
            reason="Parsed standalone option letter.",
        )

    candidates = _candidate_matches(stripped, letters)
    unique_options = sorted({option for option, _ in candidates})
    if len(unique_options) == 1:
        selected = unique_options[0]
        matched_text = next(raw for option, raw in candidates if option == selected)
        return AnswerExtractionResult(
            status="parsed",
            selected_option=selected,
            matched_text=matched_text,
            reason="Parsed from answer-format pattern.",
        )

    if len(unique_options) > 1:
        return AnswerExtractionResult(
            status="ambiguous",
            selected_option=None,
            matched_text=", ".join(unique_options),
            reason=f"Multiple answer letters detected: {', '.join(unique_options)}.",
        )

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if last_line.upper() in letters:
            return AnswerExtractionResult(
                status="parsed",
                selected_option=last_line.upper(),
                matched_text=last_line,
                reason="Parsed standalone option letter from final line.",
            )

    return AnswerExtractionResult(
        status="invalid_format",
        selected_option=None,
        matched_text=None,
        reason="No parseable final answer letter found.",
    )


def _judge_threshold(case: BenchmarkCase) -> float:
    threshold = case.judge.get("threshold") if isinstance(case.judge, dict) else None
    try:
        return float(DEFAULT_PASS_THRESHOLD if threshold is None else threshold)
    except (TypeError, ValueError):
        return DEFAULT_PASS_THRESHOLD


def score_case_result(case: BenchmarkCase, answer_text: str, judge: LLMJudge | None = None) -> dict[str, object]:
    if case.is_open_ended:
        if judge is None:
            raise ValueError("Open-ended benchmark scoring requires an LLMJudge instance.")
        judge_result = judge.evaluate_answer_sync(
            question=case.question,
            reference_answer=case.reference_answer or "",
            candidate_answer=answer_text,
            threshold=_judge_threshold(case),
            rubric=case.judge.get("rubric") if isinstance(case.judge, dict) else None,
        )
        scoring_type = "llm_judge_rubric" if case.judge.get("rubric") else "llm_judge_reference"
        return {
            "answer_status": "passed" if judge_result.passed else "failed",
            "selected_option": None,
            "is_correct": judge_result.passed,
            "expected_option": None,
            "reference_answer": case.reference_answer,
            "scoring_type": scoring_type,
            **judge_result.to_dict(),
        }

    extraction = parse_final_answer(answer_text, case.option_letters())
    if extraction.status != "parsed" or extraction.selected_option is None:
        return {
            "answer_status": extraction.status,
            "selected_option": extraction.selected_option,
            "is_correct": False,
            "expected_option": case.correct_option,
            "scoring_type": "mcq_exact_match",
            "extraction": extraction.to_dict(),
        }

    is_correct = extraction.selected_option == case.correct_option
    return {
        "answer_status": "correct" if is_correct else "incorrect",
        "selected_option": extraction.selected_option,
        "is_correct": is_correct,
        "expected_option": case.correct_option,
        "scoring_type": "mcq_exact_match",
        "extraction": extraction.to_dict(),
    }
