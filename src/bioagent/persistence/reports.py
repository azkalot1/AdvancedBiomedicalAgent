from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import asyncpg

DEFAULT_USER_ID = os.getenv("BIOAGENT_DEFAULT_USER_ID", "anonymous")
REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")

_POOL: asyncpg.Pool | None = None
_POOL_LOCK = asyncio.Lock()
_SCHEMA_INITIALIZED = False

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bioagent_reports (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tool_name TEXT,
    filename TEXT,
    display_name TEXT,
    one_line TEXT,
    status TEXT DEFAULT 'complete',
    size_chars INTEGER,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (user_id, id)
);

CREATE INDEX IF NOT EXISTS idx_bioagent_reports_user
    ON bioagent_reports (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_bioagent_reports_thread
    ON bioagent_reports (user_id, thread_id, created_at DESC);

CREATE TABLE IF NOT EXISTS bioagent_prompt_clicks (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT,
    category_id TEXT,
    category_title TEXT,
    prompt_text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'web_starter_prompt',
    clicked_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bioagent_prompt_clicks_user
    ON bioagent_prompt_clicks (user_id, clicked_at DESC);

CREATE INDEX IF NOT EXISTS idx_bioagent_prompt_clicks_clicked
    ON bioagent_prompt_clicks (clicked_at DESC);
"""


def _get_dsn() -> str:
    return (
        os.getenv("BIOAGENT_RESEARCH_OUTPUT_POSTGRES_URI")
        or os.getenv("AEGRA_POSTGRES_URI")
        or os.getenv("BIOAGENT_CHECKPOINT_POSTGRES_URI")
        or os.getenv("APP_DATABASE_URL")
        or os.getenv("POSTGRES_URI")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()


def _safe_segment(value: str | None, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    return raw.replace("/", "_").replace("\\", "_")


def normalize_report_id(report_id: str | None) -> str | None:
    if report_id is None:
        return None
    normalized = str(report_id).strip()
    if not normalized:
        return None
    if REPORT_ID_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


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


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)


async def _get_pool() -> asyncpg.Pool:
    global _POOL, _SCHEMA_INITIALIZED

    async with _POOL_LOCK:
        if _POOL is None:
            dsn = _get_dsn()
            if not dsn:
                raise RuntimeError(
                    "Missing reports database DSN. Set BIOAGENT_RESEARCH_OUTPUT_POSTGRES_URI "
                    "(or AEGRA_POSTGRES_URI/APP_DATABASE_URL/POSTGRES_URI/DATABASE_URL)."
                )
            _POOL = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=10,
            )

        if not _SCHEMA_INITIALIZED:
            await _ensure_schema(_POOL)
            _SCHEMA_INITIALIZED = True

        return _POOL


def _row_to_metadata(row: asyncpg.Record) -> dict[str, Any]:
    record = dict(row)

    created_at = record.get("created_at")
    if isinstance(created_at, datetime):
        record["created_at"] = created_at.isoformat()
    elif created_at is not None:
        record["created_at"] = str(created_at)

    record.pop("content", None)
    record["ref_id"] = str(record.get("id", "")).strip()

    display_name = str(record.get("display_name") or "").strip()
    if not display_name:
        display_name = (
            str(record.get("one_line") or "").strip()
            or str(record.get("filename") or "").strip()
            or str(record.get("id") or "").strip()
            or "report.md"
        )
    record["display_name"] = display_name

    return record


def _normalize_limit(limit: int, *, default: int, max_value: int) -> int:
    try:
        normalized = int(limit)
    except (TypeError, ValueError):
        normalized = default
    if normalized < 1:
        normalized = 1
    if normalized > max_value:
        normalized = max_value
    return normalized


def _clean_optional_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned[:max_length]


async def persist_tool_output_report(
    *,
    content: str,
    tool_name: str,
    report_id: str | None = None,
    one_line: str = "",
    display_name: str = "",
    runtime: Any | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    resolved_user, resolved_thread = resolve_scope(runtime=runtime, user_id=user_id, thread_id=thread_id)
    safe_tool_name = _safe_segment(tool_name, "tool")
    final_report_id = normalize_report_id(report_id) or f"{safe_tool_name}_{uuid4().hex[:8]}"
    filename = f"{final_report_id}.md"

    clean_one_line = one_line.strip()
    clean_display_name = display_name.strip() or clean_one_line or filename
    created_at = datetime.now(timezone.utc)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bioagent_reports (
                id,
                user_id,
                thread_id,
                tool_name,
                filename,
                display_name,
                one_line,
                status,
                size_chars,
                content,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'complete', $8, $9, $10)
            ON CONFLICT (user_id, id) DO UPDATE SET
                thread_id = EXCLUDED.thread_id,
                tool_name = EXCLUDED.tool_name,
                filename = EXCLUDED.filename,
                display_name = EXCLUDED.display_name,
                one_line = EXCLUDED.one_line,
                status = EXCLUDED.status,
                size_chars = EXCLUDED.size_chars,
                content = EXCLUDED.content
            """,
            final_report_id,
            resolved_user,
            resolved_thread,
            tool_name,
            filename,
            clean_display_name,
            clean_one_line,
            len(content),
            content,
            created_at,
        )

    return {
        "id": final_report_id,
        "ref_id": final_report_id,
        "tool_name": tool_name,
        "user_id": resolved_user,
        "thread_id": resolved_thread,
        "filename": filename,
        "display_name": clean_display_name,
        "status": "complete",
        "size_chars": len(content),
        "one_line": clean_one_line,
        "created_at": created_at.isoformat(),
    }


async def list_reports(
    *,
    user_id: str,
    thread_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    scoped_limit = _normalize_limit(limit, default=200, max_value=1000)
    pool = await _get_pool()

    async with pool.acquire() as conn:
        if thread_id:
            scoped_thread = _safe_segment(thread_id, "default")
            rows = await conn.fetch(
                """
                SELECT id, user_id, thread_id, tool_name, filename, display_name, one_line, status, size_chars, created_at
                FROM bioagent_reports
                WHERE user_id = $1 AND thread_id = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                scoped_user,
                scoped_thread,
                scoped_limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, user_id, thread_id, tool_name, filename, display_name, one_line, status, size_chars, created_at
                FROM bioagent_reports
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                scoped_user,
                scoped_limit,
            )

    return [_row_to_metadata(row) for row in rows]


async def get_report(
    *,
    user_id: str,
    report_id: str,
) -> dict[str, Any] | None:
    safe_report_id = normalize_report_id(report_id)
    if not safe_report_id:
        return None

    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, thread_id, tool_name, filename, display_name, one_line, status, size_chars, created_at
            FROM bioagent_reports
            WHERE user_id = $1 AND id = $2
            """,
            scoped_user,
            safe_report_id,
        )

    if row is None:
        return None
    return _row_to_metadata(row)


async def get_report_content(
    *,
    user_id: str,
    report_id: str,
) -> str | None:
    safe_report_id = normalize_report_id(report_id)
    if not safe_report_id:
        return None

    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT content
            FROM bioagent_reports
            WHERE user_id = $1 AND id = $2
            """,
            scoped_user,
            safe_report_id,
        )

    if row is None:
        return None
    value = row.get("content")
    return str(value) if value is not None else None


async def list_thread_tool_outputs(
    *,
    user_id: str,
    thread_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    scoped_thread = _safe_segment(thread_id, "default")
    scoped_limit = _normalize_limit(limit, default=200, max_value=1000)
    pool = await _get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, thread_id, tool_name, filename, display_name, one_line, status, size_chars, created_at
            FROM bioagent_reports
            WHERE user_id = $1 AND thread_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            scoped_user,
            scoped_thread,
            scoped_limit,
        )

    return [_row_to_metadata(row) for row in rows]


async def delete_report(
    *,
    user_id: str,
    report_id: str,
) -> bool:
    safe_report_id = normalize_report_id(report_id)
    if not safe_report_id:
        return False

    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM bioagent_reports
            WHERE user_id = $1 AND id = $2
            """,
            scoped_user,
            safe_report_id,
        )

    try:
        deleted_count = int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        return False
    return deleted_count > 0


async def persist_prompt_click(
    *,
    user_id: str,
    prompt_text: str,
    category_id: str | None = None,
    category_title: str | None = None,
    thread_id: str | None = None,
    source: str | None = "web_starter_prompt",
) -> dict[str, Any]:
    cleaned_prompt = str(prompt_text).strip()
    if not cleaned_prompt:
        raise ValueError("prompt_text cannot be empty.")

    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID)
    scoped_thread = _safe_segment(thread_id, "default") if thread_id and str(thread_id).strip() else None
    cleaned_category_id = _safe_segment(category_id, "category") if category_id and str(category_id).strip() else None
    cleaned_category_title = _clean_optional_text(category_title, max_length=200)
    cleaned_source = _safe_segment(source, "web_starter_prompt")
    cleaned_prompt = cleaned_prompt[:4000]

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO bioagent_prompt_clicks (
                user_id,
                thread_id,
                category_id,
                category_title,
                prompt_text,
                source
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, clicked_at
            """,
            scoped_user,
            scoped_thread,
            cleaned_category_id,
            cleaned_category_title,
            cleaned_prompt,
            cleaned_source,
        )

    clicked_at = row["clicked_at"] if row else datetime.now(timezone.utc)
    return {
        "id": int(row["id"]) if row else 0,
        "user_id": scoped_user,
        "thread_id": scoped_thread,
        "category_id": cleaned_category_id,
        "category_title": cleaned_category_title,
        "prompt_text": cleaned_prompt,
        "source": cleaned_source,
        "clicked_at": clicked_at.isoformat() if isinstance(clicked_at, datetime) else str(clicked_at),
    }


async def list_prompt_click_popularity(
    *,
    limit: int = 20,
    days: int = 90,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    scoped_limit = _normalize_limit(limit, default=20, max_value=200)
    try:
        scoped_days = int(days)
    except (TypeError, ValueError):
        scoped_days = 90
    if scoped_days < 1:
        scoped_days = 1
    if scoped_days > 3650:
        scoped_days = 3650

    scoped_user = _safe_segment(user_id, DEFAULT_USER_ID) if user_id and str(user_id).strip() else None

    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                category_id,
                category_title,
                prompt_text,
                COUNT(*)::BIGINT AS click_count,
                MAX(clicked_at) AS last_clicked_at
            FROM bioagent_prompt_clicks
            WHERE ($1::TEXT IS NULL OR user_id = $1)
              AND clicked_at >= now() - ($2::INT * INTERVAL '1 day')
            GROUP BY category_id, category_title, prompt_text
            ORDER BY click_count DESC, last_clicked_at DESC
            LIMIT $3
            """,
            scoped_user,
            scoped_days,
            scoped_limit,
        )

    output: list[dict[str, Any]] = []
    for row in rows:
        last_clicked_at = row.get("last_clicked_at")
        output.append(
            {
                "category_id": row.get("category_id"),
                "category_title": row.get("category_title"),
                "prompt_text": str(row.get("prompt_text") or ""),
                "click_count": int(row.get("click_count") or 0),
                "last_clicked_at": last_clicked_at.isoformat() if isinstance(last_clicked_at, datetime) else str(last_clicked_at),
            }
        )
    return output
