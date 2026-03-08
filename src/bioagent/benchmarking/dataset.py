from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _normalize_option_key(raw_key: Any) -> str:
    key = str(raw_key).strip().upper()
    if len(key) != 1 or not key.isalpha():
        raise ValueError(f"Invalid option key '{raw_key}'. Expected a single letter such as A/B/C/D.")
    return key


def _ensure_text_map(raw_options: Any) -> dict[str, str]:
    if not isinstance(raw_options, dict) or not raw_options:
        raise ValueError("Benchmark case options must be a non-empty mapping of option letter to text.")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in raw_options.items():
        key = _normalize_option_key(raw_key)
        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"Benchmark option '{key}' must have non-empty text.")
        normalized[key] = value
    return dict(sorted(normalized.items()))


def _normalize_str_list(raw_values: Any, *, field_name: str) -> list[str]:
    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError(f"Benchmark field '{field_name}' must be a list when provided.")
    values: list[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in values:
            values.append(value)
    return values


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    options: dict[str, str]
    correct_option: str
    category: str
    expected_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, source_file: str | None = None) -> "BenchmarkCase":
        case_id = str(payload.get("id", "")).strip()
        if not case_id:
            raise ValueError("Benchmark case is missing required field 'id'.")

        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError(f"Benchmark case '{case_id}' is missing required field 'question'.")

        options = _ensure_text_map(payload.get("options"))
        correct_option = _normalize_option_key(payload.get("correct_option", ""))
        if correct_option not in options:
            raise ValueError(
                f"Benchmark case '{case_id}' has correct_option='{correct_option}', "
                f"which is not present in options {sorted(options)}."
            )

        category = str(payload.get("category", "general")).strip() or "general"
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        return cls(
            case_id=case_id,
            question=question,
            options=options,
            correct_option=correct_option,
            category=category,
            expected_tools=_normalize_str_list(payload.get("expected_tools"), field_name="expected_tools"),
            allowed_tools=_normalize_str_list(payload.get("allowed_tools"), field_name="allowed_tools"),
            metadata=dict(metadata),
            source_file=source_file,
        )

    def option_letters(self) -> list[str]:
        return list(self.options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "question": self.question,
            "options": dict(self.options),
            "correct_option": self.correct_option,
            "category": self.category,
            "expected_tools": list(self.expected_tools),
            "allowed_tools": list(self.allowed_tools),
            "metadata": dict(self.metadata),
            "source_file": self.source_file,
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cases: list[BenchmarkCase]
    source_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cases": [case.to_dict() for case in self.cases],
            "source_paths": list(self.source_paths),
        }


def _load_suite_file(path: Path) -> tuple[str | None, list[BenchmarkCase]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark suite file '{path}' must contain a mapping at the top level.")

    suite_name = str(payload.get("suite", "")).strip() or None
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"Benchmark suite file '{path}' must contain a top-level 'cases' list.")

    cases = [BenchmarkCase.from_dict(item, source_file=str(path)) for item in raw_cases]
    return suite_name, cases


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Benchmark suite path does not exist: {root}")

    if root.is_file():
        suite_name, cases = _load_suite_file(root)
        return BenchmarkSuite(
            name=suite_name or root.stem,
            cases=cases,
            source_paths=[str(root)],
        )

    suite_name = root.name
    cases: list[BenchmarkCase] = []
    source_paths: list[str] = []
    for file_path in sorted(root.glob("*.yaml")):
        file_suite_name, file_cases = _load_suite_file(file_path)
        if file_suite_name and suite_name == root.name:
            suite_name = file_suite_name
        cases.extend(file_cases)
        source_paths.append(str(file_path))

    if not cases:
        raise ValueError(f"No benchmark cases found in directory '{root}'.")

    return BenchmarkSuite(name=suite_name, cases=cases, source_paths=source_paths)
