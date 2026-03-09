from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_INPUT = Path("benchmarks/curebench/raw/curebench_valset_pharse1.jsonl")
DEFAULT_OUTPUT_ROOT = Path("benchmarks/curebench")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": "curebench",
        "source_question_type": record.get("question_type"),
    }


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    question_type = str(record.get("question_type", "")).strip()
    correct_answer = str(record.get("correct_answer", "")).strip().upper() or None
    options = record.get("options") if isinstance(record.get("options"), dict) else {}
    normalized: dict[str, Any] = {
        "id": str(record.get("id", "")).strip(),
        "question": str(record.get("question", "")).strip(),
        "question_type": question_type,
        "options": options,
        "correct_answer": correct_answer,
        "metadata": _base_metadata(record),
    }
    if question_type == "multi_choice":
        normalized["benchmark_case"] = {
            "id": normalized["id"],
            "type": "mcq",
            "category": "general",
            "question": normalized["question"],
            "options": options,
            "correct_option": correct_answer,
            "metadata": _base_metadata(record),
        }
    elif question_type in {"open_ended", "open_ended_multi_choice"}:
        normalized["reference_answer"] = options.get(correct_answer) if correct_answer and isinstance(options, dict) else None
        normalized["benchmark_case"] = {
            "id": normalized["id"],
            "type": "open_ended",
            "category": "general",
            "question": normalized["question"],
            "reference_answer": normalized["reference_answer"],
            "metadata": {
                **_base_metadata(record),
                "raw_correct_option": correct_answer,
                "raw_options": options,
            },
        }
    return normalized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def prepare_curebench(input_path: Path, output_root: Path) -> dict[str, Any]:
    records = _read_jsonl(input_path)
    normalized_records = [_normalize_record(record) for record in records]
    mcq_cases = [
        record["benchmark_case"]
        for record in normalized_records
        if record.get("question_type") == "multi_choice" and isinstance(record.get("benchmark_case"), dict)
    ]
    open_ended_cases = [
        record["benchmark_case"]
        for record in normalized_records
        if record.get("question_type") in {"open_ended", "open_ended_multi_choice"}
        and isinstance(record.get("benchmark_case"), dict)
    ]
    mixed_cases = [*mcq_cases, *open_ended_cases]

    normalized_dir = output_root / "normalized"
    suites_dir = output_root / "suites"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    suites_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(normalized_dir / "source_records.jsonl", normalized_records)
    _write_yaml(suites_dir / "mcq.yaml", {"suite": "curebench_mcq", "cases": mcq_cases})
    _write_yaml(suites_dir / "open_ended.yaml", {"suite": "curebench_open_ended", "cases": open_ended_cases})
    _write_yaml(suites_dir / "mixed.yaml", {"suite": "curebench_mixed", "cases": mixed_cases})

    metadata = {
        "dataset": "curebench",
        "raw_input": str(input_path),
        "normalized_output": "normalized/source_records.jsonl",
        "suite_outputs": {
            "mcq": "suites/mcq.yaml",
            "open_ended": "suites/open_ended.yaml",
            "mixed": "suites/mixed.yaml",
        },
        "counts": {
            "total": len(normalized_records),
            "mcq": len(mcq_cases),
            "open_ended": len(open_ended_cases),
        },
        "question_types": sorted({str(record.get("question_type", "")).strip() for record in records if record.get("question_type")}),
    }
    _write_json(output_root / "metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the local CureBench raw dataset into benchmark suite YAML files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to the raw CureBench JSONL file")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory for normalized files and suites")
    args = parser.parse_args(argv)

    metadata = prepare_curebench(Path(args.input), Path(args.output_root))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
