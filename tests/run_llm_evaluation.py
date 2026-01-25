#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bioagent.agent import get_chat_model
from bioagent.agent.tools.dbsearch import DBSEARCH_TOOLS
from bioagent.agent.tools.target_search import TARGET_SEARCH_TOOLS, pharmacology_search


TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from llm_judge import LLMJudge, load_ground_truth_cases  # noqa: E402


def _get_tool_function(tool_name: str):
    for tool in DBSEARCH_TOOLS:
        if tool.name == tool_name:
            return tool
    for tool in TARGET_SEARCH_TOOLS:
        if tool.name == tool_name:
            return tool
    if tool_name == "pharmacology_search":
        return pharmacology_search
    return None


def _normalize_tool_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "search_clinical_trials" and isinstance(tool_input.get("nct_ids"), list):
        tool_input = dict(tool_input)
        tool_input["nct_ids"] = ", ".join(tool_input["nct_ids"])
    return tool_input


async def _execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    tool = _get_tool_function(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    normalized = _normalize_tool_input(tool_name, tool_input)
    return await tool.ainvoke(normalized)


async def run_evaluations(ground_truth_dir: Path, model_name: str) -> dict[str, Any]:
    model = get_chat_model(model_name, "openrouter", model_parameters={"temperature": 0.0})
    judge = LLMJudge(model)

    cases = load_ground_truth_cases(ground_truth_dir)
    results: list[dict[str, Any]] = []

    for case in cases:
        tool_output = await _execute_tool(case.tool, case.input)
        evaluation = await judge.evaluate(case, tool_output)
        results.append(
            {
                "case": asdict(case),
                "tool_output": tool_output,
                "evaluation": asdict(evaluation),
            }
        )

    passed = sum(1 for r in results if r["evaluation"]["correctness_score"] >= 0.7)
    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "results": results,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluations for search tools.")
    parser.add_argument(
        "--ground-truth-dir",
        default=str(TESTS_DIR / "ground_truth"),
        help="Directory containing YAML ground truth files",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash",
        help="Model name for LLM judge",
    )
    parser.add_argument("--output", default=None, help="Write JSON output to file")
    args = parser.parse_args()

    payload = await run_evaluations(Path(args.ground_truth_dir), args.model)

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote evaluation results to {args.output}")
    else:
        print(json.dumps(payload["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
