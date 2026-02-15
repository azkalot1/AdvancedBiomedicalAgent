from __future__ import annotations

import asyncio
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

DEFAULT_REPORTS_ROOT = Path(os.getenv("BIOAGENT_RESEARCH_OUTPUT_DIR", "./research_outputs"))
DEFAULT_USER_ID = os.getenv("BIOAGENT_DEFAULT_USER_ID", "anonymous")


def _safe_segment(value: str | None, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    return raw.replace("/", "_").replace("\\", "_")


def thread_tool_outputs_namespace(user_id: str, thread_id: str) -> tuple[str, ...]:
    return ("users", user_id, "threads", thread_id, "tool_outputs")


def user_reports_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "reports")


def _thread_output_dir(root: Path, user_id: str, thread_id: str) -> Path:
    return root / "users" / user_id / "threads" / thread_id / "tool_outputs"


def _user_index_path(root: Path, user_id: str) -> Path:
    return root / "users" / user_id / "reports_index.json"


def thread_tool_outputs_dir(
    *,
    user_id: str,
    thread_id: str,
    root: Path = DEFAULT_REPORTS_ROOT,
) -> Path:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    scoped_thread = _safe_segment(thread_id, "default")
    return _thread_output_dir(root, scoped_user, scoped_thread)


def _get_mapping_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def resolve_scope(
    *,
    runtime: Any | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> tuple[str, str]:
    runtime_context = getattr(runtime, "context", None) if runtime is not None else None
    runtime_config = getattr(runtime, "config", None) if runtime is not None else None

    resolved_user = user_id or _get_mapping_value(runtime_context, "user_id")
    resolved_thread = thread_id or _get_mapping_value(runtime_context, "thread_id")

    configurable = _get_mapping_value(runtime_config, "configurable") if runtime_config is not None else None
    if not resolved_user:
        resolved_user = _get_mapping_value(configurable, "user_id")
    if not resolved_thread:
        resolved_thread = _get_mapping_value(configurable, "thread_id")

    if not resolved_thread:
        resolved_thread = os.getenv("BIOAGENT_DEFAULT_THREAD_ID", "default")

    return (
        _safe_segment(resolved_user, DEFAULT_USER_ID),
        _safe_segment(resolved_thread, "default"),
    )


async def _store_call(
    store: Any | None,
    async_name: str,
    sync_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if store is None:
        return None

    async_method = getattr(store, async_name, None)
    if async_method is not None:
        try:
            sig = inspect.signature(async_method)
            accepted = {key: value for key, value in kwargs.items() if key in sig.parameters}
        except (TypeError, ValueError):
            accepted = kwargs
        result = async_method(*args, **accepted)
        if inspect.isawaitable(result):
            return await result
        return result

    sync_method = getattr(store, sync_name, None)
    if sync_method is not None:
        try:
            sig = inspect.signature(sync_method)
            accepted = {key: value for key, value in kwargs.items() if key in sig.parameters}
        except (TypeError, ValueError):
            accepted = kwargs
        return await asyncio.to_thread(sync_method, *args, **accepted)

    return None


def _item_value(item: Any) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        if "value" in item:
            return item["value"]
        return item
    return getattr(item, "value", None)


def _item_key(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, dict):
        if "key" in item:
            return str(item["key"])
        if "id" in item:
            return str(item["id"])
        return None
    value = getattr(item, "key", None)
    if value is not None:
        return str(value)
    value = getattr(item, "id", None)
    if value is not None:
        return str(value)
    return None


def _parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    if not markdown.startswith("---"):
        return {}, markdown
    parts = markdown.split("---", 2)
    if len(parts) < 3:
        return {}, markdown
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, parts[2].lstrip()


def _write_user_index(root: Path, user_id: str, records: list[dict[str, Any]]) -> None:
    index_path = _user_index_path(root, user_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_user_index(root: Path, user_id: str) -> list[dict[str, Any]]:
    index_path = _user_index_path(root, user_id)
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _upsert_user_index_record(root: Path, user_id: str, record: dict[str, Any]) -> None:
    existing = _load_user_index(root, user_id)
    keyed = {str(item.get("id")): item for item in existing if item.get("id")}
    keyed[str(record["id"])] = record
    _write_user_index(root, user_id, list(keyed.values()))


def _remove_user_index_record(root: Path, user_id: str, report_id: str) -> None:
    existing = _load_user_index(root, user_id)
    filtered = [item for item in existing if str(item.get("id")) != report_id]
    _write_user_index(root, user_id, filtered)


def _build_report_metadata(
    *,
    report_id: str,
    tool_name: str,
    user_id: str,
    thread_id: str,
    size_chars: int,
    one_line: str,
    file_path: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = created_at or datetime.now(timezone.utc).isoformat()
    return {
        "id": report_id,
        "ref_id": report_id,
        "tool_name": tool_name,
        "user_id": user_id,
        "thread_id": thread_id,
        "filename": file_path.name,
        "path": str(file_path),
        "status": "complete",
        "size_chars": int(size_chars),
        "one_line": one_line.strip(),
        "created_at": now,
    }


def _write_report_markdown(file_path: Path, metadata: dict[str, Any], content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()
    payload = f"---\n{frontmatter}\n---\n\n{content.rstrip()}\n"
    file_path.write_text(payload)


def _list_report_stems(output_dir: Path, limit: int = 10) -> list[str]:
    return [item.stem for item in output_dir.glob("*.md")][:limit]


def _read_file_text(path: Path) -> str:
    return path.read_text()


def _path_exists(path: Path) -> bool:
    return path.exists()


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


async def persist_tool_output_report(
    *,
    content: str,
    tool_name: str,
    report_id: str | None = None,
    one_line: str = "",
    store: Any | None = None,
    runtime: Any | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
) -> dict[str, Any]:
    resolved_user, resolved_thread = resolve_scope(runtime=runtime, user_id=user_id, thread_id=thread_id)
    final_report_id = report_id or f"{tool_name}_{uuid4().hex[:8]}"
    output_dir = _thread_output_dir(root, resolved_user, resolved_thread)
    output_path = output_dir / f"{final_report_id}.md"

    metadata = _build_report_metadata(
        report_id=final_report_id,
        tool_name=tool_name,
        user_id=resolved_user,
        thread_id=resolved_thread,
        size_chars=len(content),
        one_line=one_line,
        file_path=output_path,
    )
    await asyncio.to_thread(_write_report_markdown, output_path, metadata, content)

    await _store_call(
        store,
        "aput",
        "put",
        thread_tool_outputs_namespace(resolved_user, resolved_thread),
        final_report_id,
        metadata,
        index=False,
    )
    await _store_call(
        store,
        "aput",
        "put",
        user_reports_namespace(resolved_user),
        final_report_id,
        metadata,
        index=False,
    )
    await asyncio.to_thread(_upsert_user_index_record, root, resolved_user, metadata)
    return metadata


def _find_record_in_filesystem(root: Path, user_id: str, report_id: str) -> dict[str, Any] | None:
    for candidate in (root / "users" / user_id).glob(f"threads/*/tool_outputs/{report_id}.md"):
        metadata, _ = _parse_frontmatter(candidate.read_text())
        if not metadata:
            metadata = {
                "id": report_id,
                "ref_id": report_id,
                "filename": candidate.name,
                "path": str(candidate),
                "status": "complete",
                "created_at": datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        if "id" not in metadata:
            metadata["id"] = report_id
        if "path" not in metadata:
            metadata["path"] = str(candidate)
        return metadata
    return None


def _list_thread_records_from_filesystem(
    output_dir: Path,
    scoped_user: str,
    scoped_thread: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not output_dir.exists():
        return records

    for file_path in sorted(output_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
        metadata, _ = _parse_frontmatter(file_path.read_text())
        if not metadata:
            metadata = {}
        metadata.setdefault("id", file_path.stem)
        metadata.setdefault("ref_id", file_path.stem)
        metadata.setdefault("filename", file_path.name)
        metadata.setdefault("path", str(file_path))
        metadata.setdefault("thread_id", scoped_thread)
        metadata.setdefault("user_id", scoped_user)
        metadata.setdefault("created_at", datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat())
        metadata.setdefault("status", "complete")
        metadata.setdefault("size_chars", file_path.stat().st_size)
        records.append(metadata)

    return records


async def list_reports(
    *,
    user_id: str,
    thread_id: str | None = None,
    store: Any | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
    limit: int = 200,
) -> list[dict[str, Any]]:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)

    items = await _store_call(
        store,
        "asearch",
        "search",
        user_reports_namespace(scoped_user),
        None,
        limit=limit,
    )

    records: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            value = _item_value(item)
            if isinstance(value, dict):
                record = dict(value)
                if "id" not in record:
                    key = _item_key(item)
                    if key:
                        record["id"] = key
                records.append(record)

    indexed_records = await asyncio.to_thread(_load_user_index, root, scoped_user)
    if indexed_records:
        merged: dict[str, dict[str, Any]] = {}
        for record in indexed_records:
            report_id = str(record.get("id") or record.get("ref_id") or "")
            if not report_id:
                continue
            merged[report_id] = dict(record)
        for record in records:
            report_id = str(record.get("id") or record.get("ref_id") or "")
            if not report_id:
                continue
            merged[report_id] = {**merged.get(report_id, {}), **record}
        records = list(merged.values())

    if thread_id:
        scoped_thread = _safe_segment(thread_id, "default")
        records = [item for item in records if str(item.get("thread_id", "")) == scoped_thread]

    return sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)


async def get_report(
    *,
    user_id: str,
    report_id: str,
    store: Any | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
) -> dict[str, Any] | None:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)

    item = await _store_call(
        store,
        "aget",
        "get",
        user_reports_namespace(scoped_user),
        report_id,
    )
    value = _item_value(item)
    if isinstance(value, dict):
        record = dict(value)
        record.setdefault("id", report_id)
        return record

    for candidate in await asyncio.to_thread(_load_user_index, root, scoped_user):
        if str(candidate.get("id")) == report_id:
            return candidate

    return await asyncio.to_thread(_find_record_in_filesystem, root, scoped_user, report_id)


async def get_report_content(
    *,
    user_id: str,
    report_id: str,
    store: Any | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
) -> str | None:
    record = await get_report(user_id=user_id, report_id=report_id, store=store, root=root)
    if not record:
        return None

    path_value = record.get("path")
    if not path_value:
        filesystem_record = await asyncio.to_thread(
            _find_record_in_filesystem,
            root,
            _safe_segment(user_id, DEFAULT_USER_ID),
            report_id,
        )
        if not filesystem_record:
            return None
        path_value = filesystem_record.get("path")
    if not path_value:
        return None

    file_path = Path(path_value)
    if not await asyncio.to_thread(_path_exists, file_path):
        return None
    file_content = await asyncio.to_thread(_read_file_text, file_path)
    _, content = _parse_frontmatter(file_content)
    return content


async def list_thread_tool_outputs(
    *,
    user_id: str,
    thread_id: str,
    store: Any | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
    limit: int = 200,
) -> list[dict[str, Any]]:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    scoped_thread = _safe_segment(thread_id, "default")

    items = await _store_call(
        store,
        "asearch",
        "search",
        thread_tool_outputs_namespace(scoped_user, scoped_thread),
        None,
        limit=limit,
    )
    records: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            value = _item_value(item)
            if isinstance(value, dict):
                record = dict(value)
                if "id" not in record:
                    key = _item_key(item)
                    if key:
                        record["id"] = key
                records.append(record)

    output_dir = _thread_output_dir(root, scoped_user, scoped_thread)
    fs_records = await asyncio.to_thread(_list_thread_records_from_filesystem, output_dir, scoped_user, scoped_thread)
    if fs_records:
        merged: dict[str, dict[str, Any]] = {}
        for record in fs_records:
            report_id = str(record.get("id") or record.get("ref_id") or "")
            if not report_id:
                continue
            merged[report_id] = dict(record)
        for record in records:
            report_id = str(record.get("id") or record.get("ref_id") or "")
            if not report_id:
                continue
            merged[report_id] = {**merged.get(report_id, {}), **record}
        records = list(merged.values())

    return sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)


async def delete_report(
    *,
    user_id: str,
    report_id: str,
    store: Any | None = None,
    root: Path = DEFAULT_REPORTS_ROOT,
) -> bool:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    record = await get_report(user_id=scoped_user, report_id=report_id, store=store, root=root)
    if not record:
        return False

    await _store_call(
        store,
        "adelete",
        "delete",
        user_reports_namespace(scoped_user),
        report_id,
    )

    thread_id = record.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        await _store_call(
            store,
            "adelete",
            "delete",
            thread_tool_outputs_namespace(scoped_user, _safe_segment(thread_id, "default")),
            report_id,
        )

    path_value = record.get("path")
    if isinstance(path_value, str) and path_value:
        file_path = Path(path_value)
        await asyncio.to_thread(_unlink_if_exists, file_path)

    await asyncio.to_thread(_remove_user_index_record, root, scoped_user, report_id)
    return True
