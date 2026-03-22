#!/usr/bin/env python3
"""Append or update an entry in the shared chat model catalog (Python server + Next.js web).

Edits src/bioagent/config/chat_models.json. Restart the Python server after changes; rerun or
rebuild the web app so Next.js picks up the updated JSON.

Examples:
  PYTHONPATH=src python scripts/add_chat_model.py --id "vendor/model-id" --label "Display name"
  PYTHONPATH=src python scripts/add_chat_model.py --id "vendor/model-id" --label "Name" --context-window 128000
  PYTHONPATH=src python scripts/add_chat_model.py --id "vendor/model-id" --label "Renamed" --replace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from bioagent.config.model_catalog import (  # noqa: E402
    default_chat_models_path,
    reload_chat_models_catalog,
    validate_catalog,
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    models = data.get("models")
    if not isinstance(models, list):
        raise SystemExit(f"Expected 'models' array in {path}")
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="OpenRouter-style model id, e.g. minimax/minimax-m2.7")
    parser.add_argument("--label", required=True, help="Short label shown in the UI")
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        metavar="N",
        help="Optional context length in tokens (defaults to default_context_window_tokens)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Update label/context_window_tokens if the id already exists",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override catalog path (otherwise BIOAGENT_CHAT_MODELS_PATH or package default)",
    )
    args = parser.parse_args(argv)

    model_id = args.id.strip()
    label = args.label.strip()
    if not model_id or not label:
        raise SystemExit("--id and --label must be non-empty")

    path = Path(args.path).expanduser().resolve() if args.path else default_chat_models_path()
    if not path.is_file():
        raise SystemExit(f"Catalog file not found: {path}")

    data = _load(path)
    models: list[Any] = data["models"]
    found: int | None = None
    for i, item in enumerate(models):
        if isinstance(item, dict) and item.get("id") == model_id:
            found = i
            break

    if args.context_window is not None and args.context_window <= 0:
        raise SystemExit("--context-window must be positive")

    if found is not None:
        if not args.replace:
            raise SystemExit(f"Model id already exists: {model_id} (use --replace to update)")
        existing = models[found]
        if not isinstance(existing, dict):
            raise SystemExit(f"Invalid models[{found}] entry")
        updated: dict[str, Any] = {"id": model_id, "label": label}
        if args.context_window is not None:
            updated["context_window_tokens"] = args.context_window
        else:
            cwt = existing.get("context_window_tokens")
            if isinstance(cwt, int) and cwt > 0:
                updated["context_window_tokens"] = cwt
        models[found] = updated
        action = "Updated"
    else:
        new_entry: dict[str, Any] = {"id": model_id, "label": label}
        if args.context_window is not None:
            new_entry["context_window_tokens"] = args.context_window
        models.append(new_entry)
        action = "Added"

    _save(path, data)
    reload_chat_models_catalog()
    try:
        validate_catalog()
    except Exception as exc:
        raise SystemExit(f"Catalog validation failed after write: {exc}") from exc

    print(f"{action} {model_id!r} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
