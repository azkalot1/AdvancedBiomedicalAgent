from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as exc:
    missing_package = exc.name or "a plotting dependency"
    raise SystemExit(
        f"Missing dependency: {missing_package}. Install plotting dependencies with "
        "'pip install matplotlib seaborn' and rerun the script."
    ) from exc


DEFAULT_RUNS_ROOT = Path("benchmarks/curebench/runs")
DEFAULT_OUTPUT = Path("benchmarks/curebench/runs/manual_mcq_accuracy_barplot.png")
DEFAULT_RUN_IDS = [
    "gemini31flashlitepreview",
    "gemini31propreviewonlythinking",
    "glm5",
    "kimik25",
    "minmaxm27",
    "qwen3535ba3b",
    "qwen35397ba17b",
]
DISPLAY_NAMES = {
    "gemini31flashlitepreview": "Gemini 3.1 Flash Lite",
    "gemini31propreviewonlythinking": "Gemini 3.1 Pro - No Tools",
    "glm5": "GLM-5",
    "kimik25": "Kimi K2.5",
    "minmaxm25": "MiniMax-M2.5",
    "minmaxm27": "MiniMax-M2.7",
    "qwen3535ba3b": "Qwen 3.5 35B A3B",
    "qwen35397ba17b": "Qwen 3.5 397B A17B",
}


@dataclass(frozen=True)
class RunMetric:
    run_id: str
    display_name: str
    accuracy: float
    correct: int
    total: int
    summary_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return payload


def _extract_metric(summary_path: Path, run_id: str) -> RunMetric:
    payload = _load_json(summary_path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing 'summary' object in {summary_path}")

    accuracy = summary.get("mcq_accuracy")
    if accuracy is None:
        accuracy = summary.get("accuracy")
    if not isinstance(accuracy, (int, float)):
        raise ValueError(f"Missing numeric mcq accuracy in {summary_path}")

    correct = summary.get("correct")
    total = summary.get("total")
    if not isinstance(correct, int) or not isinstance(total, int):
        raise ValueError(f"Missing integer correct/total values in {summary_path}")

    return RunMetric(
        run_id=run_id,
        display_name=DISPLAY_NAMES.get(run_id, run_id),
        accuracy=float(accuracy),
        correct=correct,
        total=total,
        summary_path=summary_path,
    )


def collect_metrics(runs_root: Path, run_ids: list[str]) -> list[RunMetric]:
    metrics: list[RunMetric] = []
    for run_id in run_ids:
        summary_path = runs_root / run_id / "manual_mcq" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        metrics.append(_extract_metric(summary_path, run_id))
    return metrics


def plot_metrics(metrics: list[RunMetric], output_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")

    labels = [metric.display_name for metric in metrics]
    values = [metric.accuracy * 100 for metric in metrics]

    fig, ax = plt.subplots(figsize=(14, 8))
    palette = sns.color_palette("Blues_d", n_colors=len(metrics))
    bars = sns.barplot(x=labels, y=values, hue=labels, palette=palette, legend=False, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)

    for patch, metric in zip(bars.patches, metrics, strict=True):
        height = patch.get_height()
        ax.annotate(
            f"{metric.accuracy * 100:.1f}%\n{metric.correct}/{metric.total}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a seaborn bar plot from CureBench manual_mcq summary.json files.")
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root directory containing CureBench run folders",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output image path",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Run folder name to include. Pass multiple times to override the defaults.",
    )
    parser.add_argument(
        "--title",
        default="CureBench Manual MCQ Accuracy by Model",
        help="Chart title",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_ids = args.run_ids or DEFAULT_RUN_IDS
    metrics = collect_metrics(Path(args.runs_root), run_ids)
    plot_metrics(metrics, Path(args.output), args.title)

    for metric in metrics:
        print(f"{metric.run_id}\t{metric.accuracy:.6f}\t{metric.correct}/{metric.total}\t{metric.summary_path}")
    print(f"\nWrote bar plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
