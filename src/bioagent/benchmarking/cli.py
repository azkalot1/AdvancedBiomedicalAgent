from __future__ import annotations

import argparse
import json
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import BenchmarkProfile, BenchmarkRunPolicy, BenchmarkServerConfig, ModelProfile, load_profile, load_profiles
from .dataset import BenchmarkSuite, load_benchmark_suite
from .http_runner import BenchmarkHttpRunner, BenchmarkRunResult, ManagedAegraServer
from .metrics import summarize_results

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run automated HTTP benchmarks against the biomedical agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite")
    run_parser.add_argument("--suite", required=True, help="Path to a benchmark YAML file or directory")
    run_parser.add_argument("--output-dir", required=True, help="Directory for raw_runs.jsonl and summaries")
    run_parser.add_argument("--profile-file", default=None, help="YAML file containing named benchmark profiles")
    run_parser.add_argument("--profile", default=None, help="Named profile in --profile-file")
    run_parser.add_argument("--base-url", default="http://localhost:8000", help="Existing Aegra server base URL")
    run_parser.add_argument("--assistant-id", default="co_scientist", help="Assistant graph id")
    run_parser.add_argument("--api-token", default=None, help="Optional API bearer token")
    run_parser.add_argument("--user-id", default=None, help="Optional fixed user id instead of GET /v1/me")
    run_parser.add_argument("--provider", default="openrouter", help="Model provider for launched server profiles")
    run_parser.add_argument("--model", default="google/gemini-3-flash-preview", help="Model name")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Primary model temperature")
    run_parser.add_argument("--model-base-url", default=None, help="Model base URL for local/OpenAI-compatible providers")
    run_parser.add_argument("--model-api-key", default=None, help="Model API key for launched server profiles")
    run_parser.add_argument(
        "--tool-mode",
        default="all",
        choices=["all", "selective", "none"],
        help="Tool exposure mode: all tools, selective subset, or no tools bound",
    )
    run_parser.add_argument("--allow-tool", action="append", default=[], help="Allow only the named tool (repeatable)")
    run_parser.add_argument("--deny-tool", action="append", default=[], help="Disallow the named tool (repeatable)")
    run_parser.add_argument("--max-total-tool-calls", type=int, default=None, help="Global tool-call budget per case")
    run_parser.add_argument("--max-tools-per-step", type=int, default=None, help="Max tools exposed per model step")
    run_parser.add_argument(
        "--retry-attempts",
        type=int,
        default=0,
        help="Number of extra attempts for invalid-format or error case results",
    )
    run_parser.add_argument("--stream-tool-args", action="store_true", help="Capture sanitized tool arg previews")
    run_parser.add_argument("--timeout", type=float, default=180.0, help="Per-case timeout and default server timeout")
    run_parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent cases to run")
    run_parser.add_argument("--run-label", default=None, help="Optional human-readable label for this benchmark run")
    run_parser.add_argument(
        "--launch-command",
        default=None,
        help="Optional server launch command. Example: './scripts/run_aegra.sh dev-quick'",
    )
    run_parser.add_argument("--launch-cwd", default=None, help="Working directory for --launch-command")
    run_parser.add_argument(
        "--launch-env",
        action="append",
        default=[],
        help="Extra env override for launched server in KEY=VALUE format (repeatable)",
    )
    run_parser.add_argument("--readiness-path", default="/health", help="Health endpoint for launched server")
    run_parser.add_argument("--readiness-timeout", type=float, default=90.0, help="Seconds to wait for launched server")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a raw_runs.jsonl file")
    summarize_parser.add_argument("--input", required=True, help="Path to raw_runs.jsonl")
    summarize_parser.add_argument("--output", default=None, help="Optional path to summary.json")

    list_profiles_parser = subparsers.add_parser("list-profiles", help="List profiles in a profile file")
    list_profiles_parser.add_argument("--profile-file", required=True, help="YAML file containing named benchmark profiles")
    return parser


def _parse_key_value_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        key, separator, parsed = value.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE for launch env override, got '{value}'.")
        key = key.strip()
        if not key:
            raise ValueError(f"Launch env key cannot be empty in '{value}'.")
        env[key] = parsed
    return env


def _profile_from_args(args: argparse.Namespace) -> BenchmarkProfile:
    if args.profile_file and args.profile:
        return load_profile(args.profile_file, args.profile)
    if args.profile_file and not args.profile:
        profiles = load_profiles(args.profile_file)
        if len(profiles) != 1:
            available = ", ".join(sorted(profiles))
            raise ValueError(f"--profile is required when the profile file has multiple profiles: {available}")
        return next(iter(profiles.values()))

    launch_command = shlex.split(args.launch_command) if args.launch_command else []
    server = BenchmarkServerConfig(
        base_url=args.base_url.rstrip("/"),
        assistant_id=args.assistant_id,
        api_token=args.api_token,
        user_id=args.user_id,
        timeout_seconds=args.timeout,
        launch_command=launch_command,
        launch_cwd=args.launch_cwd,
        launch_env=_parse_key_value_pairs(args.launch_env),
        readiness_path=args.readiness_path,
        readiness_timeout_seconds=args.readiness_timeout,
    )
    model = ModelProfile(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.model_base_url,
        api_key=args.model_api_key,
    )
    run_policy = BenchmarkRunPolicy(
        tool_mode=args.tool_mode,
        allowed_tools=list(dict.fromkeys(args.allow_tool)),
        disallowed_tools=list(dict.fromkeys(args.deny_tool)),
        max_total_tool_calls=args.max_total_tool_calls,
        max_tools_per_step=args.max_tools_per_step,
        retry_attempts=max(0, args.retry_attempts),
        stream_tool_args=args.stream_tool_args,
        per_case_timeout_seconds=args.timeout,
        concurrency=max(1, args.concurrency),
    )
    return BenchmarkProfile(name="adhoc", model=model, server=server, run_policy=run_policy)


def _write_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [json.dumps(result, ensure_ascii=False) for result in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_summary_markdown(suite: BenchmarkSuite, summary: dict[str, Any]) -> str:
    summary_block = summary["summary"]
    lines = [
        f"# Benchmark Summary: {suite.name}",
        "",
        f"- total: {summary_block['total']}",
        f"- correct: {summary_block['correct']}",
        f"- incorrect: {summary_block['incorrect']}",
        f"- invalid_format: {summary_block['invalid_format']}",
        f"- error: {summary_block['error']}",
        f"- accuracy: {summary_block['accuracy']:.3f}",
        f"- pass_rate: {summary_block.get('pass_rate', summary_block['accuracy']):.3f}",
        f"- mcq_accuracy: {summary_block.get('mcq_accuracy')}",
        f"- open_ended_pass_rate: {summary_block.get('open_ended_pass_rate')}",
        f"- average_score: {summary_block.get('average_score')}",
        f"- invalid_format_rate: {summary_block['invalid_format_rate']:.3f}",
        f"- error_rate: {summary_block['error_rate']:.3f}",
        f"- average_latency_seconds: {summary_block['average_latency_seconds']:.3f}",
        f"- average_tool_calls: {summary_block['average_tool_calls']:.3f}",
        f"- token_usage_available_runs: {summary_block['token_usage_available_runs']}",
        f"- sum_input_tokens: {summary_block['sum_input_tokens']}",
        f"- sum_output_tokens: {summary_block['sum_output_tokens']}",
        f"- sum_total_tokens: {summary_block['sum_total_tokens']}",
        f"- average_input_tokens: {summary_block['average_input_tokens']}",
        f"- average_output_tokens: {summary_block['average_output_tokens']}",
        f"- average_total_tokens: {summary_block['average_total_tokens']}",
        f"- sum_streamed_output_chars: {summary_block['sum_streamed_output_chars']}",
        f"- average_streamed_output_chars: {summary_block['average_streamed_output_chars']:.3f}",
        "",
        "## By Category",
    ]
    for category, stats in summary["by_category"].items():
        lines.extend(
            [
                f"",
                f"### {category}",
                f"- total: {stats['total']}",
                f"- correct: {stats['correct']}",
                f"- incorrect: {stats['incorrect']}",
                f"- invalid_format: {stats['invalid_format']}",
                f"- error: {stats['error']}",
                f"- accuracy: {stats['accuracy']:.3f}",
            ]
        )

    if summary["tool_usage"]:
        lines.extend(["", "## Tool Usage"])
        for tool_name, count in summary["tool_usage"].items():
            lines.append(f"- {tool_name}: {count}")
    if summary.get("score_dimensions"):
        lines.extend(["", "## Judge Dimensions"])
        for dimension_name, value in summary["score_dimensions"].items():
            lines.append(f"- {dimension_name}: {value:.3f}")
    return "\n".join(lines) + "\n"


def _run_single_case(profile: BenchmarkProfile, case) -> BenchmarkRunResult:
    runner = BenchmarkHttpRunner(profile)
    try:
        return runner.run_case(case)
    finally:
        runner.close()


def _print_case_progress(index: int, total: int, result: BenchmarkRunResult) -> None:
    payload = result.to_dict()
    case = payload.get("case", {}) if isinstance(payload.get("case"), dict) else {}
    case_id = case.get("id", f"case_{index}")
    status = payload.get("answer_status", "unknown")
    latency = payload.get("latency_seconds")
    latency_label = f"{float(latency):.1f}s" if isinstance(latency, (int, float)) else "n/a"
    print(f"[{index}/{total}] {case_id} -> {status} ({latency_label})")


def _run_suite(profile: BenchmarkProfile, suite: BenchmarkSuite) -> list[BenchmarkRunResult]:
    bootstrap_runner = BenchmarkHttpRunner(profile)
    try:
        bootstrap_runner.check_health()
    finally:
        bootstrap_runner.close()

    total_cases = len(suite.cases)
    progress = tqdm(total=total_cases, desc="Benchmark", unit="case") if tqdm is not None else None

    if profile.run_policy.concurrency <= 1:
        results: list[BenchmarkRunResult] = []
        for index, case in enumerate(suite.cases, start=1):
            result = _run_single_case(profile, case)
            results.append(result)
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(str(result.to_dict().get("answer_status", "unknown")))
            else:
                _print_case_progress(index, total_cases, result)
        if progress is not None:
            progress.close()
        return results

    indexed_results: list[BenchmarkRunResult | None] = [None] * len(suite.cases)
    with ThreadPoolExecutor(max_workers=profile.run_policy.concurrency) as executor:
        future_map = {executor.submit(_run_single_case, profile, case): index for index, case in enumerate(suite.cases)}
        completed = 0
        for future in as_completed(future_map):
            index = future_map[future]
            result = future.result()
            indexed_results[index] = result
            completed += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(str(result.to_dict().get("answer_status", "unknown")))
            else:
                _print_case_progress(completed, total_cases, result)
    if progress is not None:
        progress.close()
    return [result for result in indexed_results if result is not None]


def _sanitize_file_stem(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "case"


def _attach_run_metadata(
    *,
    run_result: BenchmarkRunResult,
    suite: BenchmarkSuite,
    run_id: str,
    run_started_at: str,
    run_label: str | None,
    dataset_path: str,
) -> BenchmarkRunResult:
    payload = run_result.to_dict()
    detail_payload = run_result.to_detail_dict()
    metadata = {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_label": run_label,
        "suite_name": suite.name,
        "dataset_path": dataset_path,
        "case_id": payload.get("case", {}).get("id") if isinstance(payload.get("case"), dict) else None,
        "model_provider": payload.get("profile", {}).get("model", {}).get("provider")
        if isinstance(payload.get("profile"), dict)
        else None,
        "model_name": payload.get("profile", {}).get("model", {}).get("model_name")
        if isinstance(payload.get("profile"), dict)
        else None,
    }
    payload.update(metadata)
    detail_payload.update(metadata)
    return BenchmarkRunResult(payload, detail_payload)


def _write_case_detail_files(output_dir: Path, results: list[BenchmarkRunResult]) -> list[dict[str, Any]]:
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    summarized_results: list[dict[str, Any]] = []

    for index, result in enumerate(results, start=1):
        payload = result.to_dict()
        detail_payload = result.to_detail_dict()
        case = payload.get("case", {}) if isinstance(payload.get("case"), dict) else {}
        case_id = str(case.get("id", f"case_{index}"))
        detail_filename = f"{index:04d}_{_sanitize_file_stem(case_id)}.json"
        detail_path = cases_dir / detail_filename
        _write_json(detail_path, detail_payload)
        payload["detail_path"] = str(detail_path.relative_to(output_dir))
        summarized_results.append(payload)

    return summarized_results


def _build_manifest(
    *,
    run_id: str,
    run_started_at: str,
    run_label: str | None,
    suite: BenchmarkSuite,
    profile: BenchmarkProfile,
    output_dir: Path,
    raw_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_label": run_label,
        "suite_name": suite.name,
        "dataset_paths": list(suite.source_paths),
        "profile": profile.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "files": {
            "manifest": "manifest.json",
            "raw_runs": "raw_runs.jsonl",
            "summary_json": "summary.json",
            "summary_markdown": "summary.md",
            "cases_dir": "cases",
        },
        "counts": {
            "cases": len(raw_results),
            "correct": summary["summary"]["correct"],
            "incorrect": summary["summary"]["incorrect"],
            "invalid_format": summary["summary"]["invalid_format"],
            "error": summary["summary"]["error"],
        },
        "models": [
            {
                "provider": profile.model.provider,
                "model_name": profile.model.model_name,
            }
        ],
    }


def _command_run(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    suite = load_benchmark_suite(args.suite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"benchmark_run_{uuid4().hex[:12]}"
    run_started_at = datetime.now(timezone.utc).isoformat()

    with ManagedAegraServer(profile):
        results = _run_suite(profile, suite)

    enriched_results = [
        _attach_run_metadata(
            run_result=result,
            suite=suite,
            run_id=run_id,
            run_started_at=run_started_at,
            run_label=args.run_label,
            dataset_path=str(args.suite),
        )
        for result in results
    ]
    raw_results = _write_case_detail_files(output_dir, enriched_results)
    summary = summarize_results(raw_results)
    manifest = _build_manifest(
        run_id=run_id,
        run_started_at=run_started_at,
        run_label=args.run_label,
        suite=suite,
        profile=profile,
        output_dir=output_dir,
        raw_results=raw_results,
        summary=summary,
    )
    raw_path = output_dir / "raw_runs.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    _write_jsonl(raw_path, raw_results)
    _write_json(manifest_path, manifest)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(suite, summary), encoding="utf-8")

    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote raw results to {raw_path}")
    print(f"Wrote summary JSON to {summary_json_path}")
    print(f"Wrote summary Markdown to {summary_md_path}")
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    results = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            results.append(json.loads(stripped))
    summary = summarize_results(results)
    rendered = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Wrote summary to {args.output}")
    else:
        print(rendered)
    return 0


def _command_list_profiles(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.profile_file)
    for name in sorted(profiles):
        profile = profiles[name]
        print(
            f"{name}: model={profile.model.model_name} provider={profile.model.provider} "
            f"base_url={profile.server.base_url} launch={'yes' if profile.server.launch_enabled else 'no'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _command_run(args)
    if args.command == "summarize":
        return _command_summarize(args)
    if args.command == "list-profiles":
        return _command_list_profiles(args)
    raise ValueError(f"Unsupported benchmark command: {args.command}")
