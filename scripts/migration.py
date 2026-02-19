#!/usr/bin/env python3
"""
Railway Cloud Migration — Resilient Table-by-Table Copy

Features:
- Persistent progress tracking (survives crashes/disconnects)
- Per-table row count verification
- Automatic retry with exponential backoff
- Chunked copy for large tables with checkpoint resume
- Idempotent: safe to re-run at any point

Usage:
    python scripts/migration.py --skip-rdkit --source-dsn ... --dest-dsn ...
    python scripts/migration.py --skip-rdkit --resume
    python scripts/migration.py --skip-rdkit --verify-only
    python scripts/migration.py --skip-rdkit --retry-failed
    python scripts/migration.py --skip-rdkit --table dm_molecule
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DBConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def psql_env(self) -> dict:
        return {**os.environ, "PGPASSWORD": self.password}

    @property
    def psql_args(self) -> list[str]:
        return ["-h", self.host, "-p", str(self.port), "-U", self.user, "-d", self.database]

    @classmethod
    def from_dsn(cls, dsn: str) -> "DBConfig":
        parsed = urlparse(dsn)
        if parsed.scheme not in {"postgresql", "postgres"}:
            raise ValueError("DSN must use postgresql:// or postgres://")
        if not parsed.hostname:
            raise ValueError("DSN missing host")
        if not parsed.path or parsed.path == "/":
            raise ValueError("DSN missing database name")
        if not parsed.username:
            raise ValueError("DSN missing username")
        if parsed.password is None:
            raise ValueError("DSN missing password")
        return cls(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/"),
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
        )


DEFAULT_SOURCE_DSN = "postgresql://postgres:postgres@localhost:5432/bioagent"
DEFAULT_DEST_DSN = "postgresql://postgres:postgres@localhost:5432/railway"

RDKIT_TYPES = {"mol", "bfp", "sfp"}

# Progress file — this is the key to resilience
PROGRESS_FILE = Path("migration_progress.json")

# Retry configuration
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5        # seconds
RETRY_MAX_DELAY = 120       # seconds
CONNECTION_TIMEOUT = 30      # seconds
COPY_STATEMENT_TIMEOUT = 600 # seconds (10 min per batch)


def _load_repo_env() -> None:
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    load_dotenv(override=False)


def _first_env(*keys: str) -> str:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def parse_db_config(source_dsn: str, dest_dsn: str) -> tuple[DBConfig, DBConfig]:
    return DBConfig.from_dsn(source_dsn), DBConfig.from_dsn(dest_dsn)


def _is_safe_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _require_safe_identifier(name: str, kind: str = "identifier") -> str:
    if not _is_safe_identifier(name):
        raise ValueError(f"Unsafe {kind}: {name!r}")
    return name


def _sql_literal(value: str) -> str:
    return psycopg2.extensions.adapt(value).getquoted().decode()


# =============================================================================
# PROGRESS TRACKING — survives crashes, disconnects, restarts
# =============================================================================

class TableStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COPIED = "copied"           # Data copied, not yet verified
    VERIFIED = "verified"       # Row counts match
    INDEXED = "indexed"         # Indexes rebuilt
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TableProgress:
    name: str
    status: TableStatus = TableStatus.PENDING
    source_rows: int = 0
    dest_rows: int = 0
    rows_copied_so_far: int = 0     # For chunked resume
    last_pk_copied: Optional[str] = None  # Checkpoint for chunked copy
    pk_column: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    columns_copied: Optional[list[str]] = None
    columns_skipped: Optional[list[str]] = None
    size_bytes: int = 0
    is_materialized_view: bool = False


@dataclass
class MigrationProgress:
    """Full migration state — serialized to JSON after every table."""
    version: str = "2.0"
    started_at: str = ""
    last_updated: str = ""
    schema_migrated: bool = False
    extensions_created: bool = False
    sequences_reset: bool = False
    functions_restored: bool = False
    views_restored: bool = False
    skip_rdkit: bool = False
    tables: dict[str, TableProgress] = field(default_factory=dict)

    def save(self, path: Path = PROGRESS_FILE):
        """Persist progress to disk — called after every significant operation."""
        self.last_updated = datetime.now(timezone.utc).isoformat()
        data = {
            "version": self.version,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "schema_migrated": self.schema_migrated,
            "extensions_created": self.extensions_created,
            "sequences_reset": self.sequences_reset,
            "functions_restored": self.functions_restored,
            "views_restored": self.views_restored,
            "skip_rdkit": self.skip_rdkit,
            "tables": {
                name: {
                    "name": tp.name,
                    "status": tp.status.value,
                    "source_rows": tp.source_rows,
                    "dest_rows": tp.dest_rows,
                    "rows_copied_so_far": tp.rows_copied_so_far,
                    "last_pk_copied": tp.last_pk_copied,
                    "pk_column": tp.pk_column,
                    "attempts": tp.attempts,
                    "last_error": tp.last_error,
                    "started_at": tp.started_at,
                    "completed_at": tp.completed_at,
                    "elapsed_seconds": tp.elapsed_seconds,
                    "columns_copied": tp.columns_copied,
                    "columns_skipped": tp.columns_skipped,
                    "size_bytes": tp.size_bytes,
                    "is_materialized_view": tp.is_materialized_view,
                }
                for name, tp in self.tables.items()
            },
        }
        # Write atomically (write to temp, then rename)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.rename(path)

    @classmethod
    def load(cls, path: Path = PROGRESS_FILE) -> "MigrationProgress":
        """Load progress from disk."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        progress = cls(
            version=data.get("version", "1.0"),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            schema_migrated=data.get("schema_migrated", False),
            extensions_created=data.get("extensions_created", False),
            sequences_reset=data.get("sequences_reset", False),
            functions_restored=data.get("functions_restored", False),
            views_restored=data.get("views_restored", False),
            skip_rdkit=data.get("skip_rdkit", False),
        )
        for name, td in data.get("tables", {}).items():
            progress.tables[name] = TableProgress(
                name=td["name"],
                status=TableStatus(td["status"]),
                source_rows=td.get("source_rows", 0),
                dest_rows=td.get("dest_rows", 0),
                rows_copied_so_far=td.get("rows_copied_so_far", 0),
                last_pk_copied=td.get("last_pk_copied"),
                pk_column=td.get("pk_column"),
                attempts=td.get("attempts", 0),
                last_error=td.get("last_error"),
                started_at=td.get("started_at"),
                completed_at=td.get("completed_at"),
                elapsed_seconds=td.get("elapsed_seconds", 0),
                columns_copied=td.get("columns_copied"),
                columns_skipped=td.get("columns_skipped"),
                size_bytes=td.get("size_bytes", 0),
                is_materialized_view=td.get("is_materialized_view", False),
            )
        return progress

    def summary(self) -> dict:
        """Get summary counts by status."""
        counts = {}
        for tp in self.tables.values():
            status = tp.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts


# =============================================================================
# CONNECTION MANAGEMENT WITH RETRY
# =============================================================================

class ConnectionManager:
    """Manages database connections with automatic reconnection."""

    def __init__(self, config: DBConfig, name: str = "db"):
        self.config = config
        self.name = name
        self._conn: Optional[psycopg2.extensions.connection] = None

    def get(self, autocommit: bool = True, readonly: bool = False) -> psycopg2.extensions.connection:
        """Get a connection, reconnecting if necessary."""
        if self._conn is not None:
            try:
                # Test if connection is alive
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
            except Exception:
                self._close_quiet()

        return self._connect(autocommit=autocommit, readonly=readonly)

    def _connect(self, autocommit: bool = True, readonly: bool = False) -> psycopg2.extensions.connection:
        """Establish connection with retry."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._conn = psycopg2.connect(
                    self.config.dsn,
                    connect_timeout=CONNECTION_TIMEOUT,
                    options=f"-c statement_timeout={COPY_STATEMENT_TIMEOUT * 1000}",
                )
                self._conn.set_session(autocommit=autocommit, readonly=readonly)
                return self._conn
            except Exception as e:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                print(f"     ⚠️  {self.name} connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"     ⏳ Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to {self.name} after {MAX_RETRIES} attempts"
                    ) from e

    def close(self):
        self._close_quiet()

    def _close_quiet(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def retry_operation(func, *args, max_retries: int = MAX_RETRIES, operation_name: str = "operation", **kwargs):
    """
    Retry a function with exponential backoff.
    Catches connection errors, timeouts, and transient failures.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (
            psycopg2.OperationalError,
            psycopg2.InterfaceError,
            ConnectionError,
            BrokenPipeError,
            ConnectionResetError,
            OSError,
            subprocess.TimeoutExpired,
            TimeoutError,
            RuntimeError,
        ) as e:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            print(f"     ⚠️  {operation_name} attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print(f"     ⏳ Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            # Non-retryable error
            raise


# =============================================================================
# TABLE DISCOVERY
# =============================================================================

def discover_tables(source_mgr: ConnectionManager) -> list[dict]:
    """Discover all tables, their sizes, PKs, and dependency order."""
    conn = source_mgr.get(readonly=True)

    with conn.cursor() as cur:
        # All tables in public schema
        cur.execute("""
            SELECT 
                t.tablename,
                pg_total_relation_size(quote_ident(t.tablename))::bigint AS size_bytes,
                c.reltuples::bigint AS approx_rows
            FROM pg_tables t
            JOIN pg_class c ON c.relname = t.tablename
            JOIN pg_namespace n ON c.relnamespace = n.oid AND n.nspname = 'public'
            WHERE t.schemaname = 'public'
            ORDER BY size_bytes DESC
        """)
        tables = []
        for tablename, size_bytes, approx_rows in cur.fetchall():
            tables.append({
                "name": tablename,
                "size_bytes": size_bytes,
                "approx_rows": max(approx_rows, 0),
            })

        # Find primary key columns for each table (needed for chunked resume)
        for t in tables:
            cur.execute("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass
                  AND i.indisprimary
                ORDER BY array_position(i.indkey, a.attnum)
                LIMIT 1
            """, (t["name"],))
            row = cur.fetchone()
            t["pk_column"] = row[0] if row else None

        # Materialized views
        cur.execute("""
            SELECT 
                matviewname,
                pg_total_relation_size(quote_ident(matviewname))::bigint AS size_bytes
            FROM pg_matviews
            WHERE schemaname = 'public'
        """)
        for mvname, size_bytes in cur.fetchall():
            tables.append({
                "name": mvname,
                "size_bytes": size_bytes,
                "approx_rows": 0,  # Will count later
                "pk_column": None,
                "is_materialized_view": True,
            })

    return tables


def get_exact_row_count(conn_mgr: ConnectionManager, table_name: str) -> int:
    """Get exact row count (slower but accurate)."""
    table_name = _require_safe_identifier(table_name, "table name")
    conn = conn_mgr.get(readonly=True)
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]


def get_non_rdkit_columns(conn_mgr: ConnectionManager, table_name: str) -> tuple[list[str], list[str]]:
    """Returns (columns_to_copy, rdkit_columns_skipped)."""
    conn = conn_mgr.get(readonly=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        to_copy, skipped = [], []
        for col_name, udt_name in cur.fetchall():
            if udt_name in RDKIT_TYPES:
                skipped.append(col_name)
            else:
                to_copy.append(col_name)
        return to_copy, skipped


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_table(
    source_mgr: ConnectionManager,
    dest_mgr: ConnectionManager,
    table_name: str,
    columns: Optional[list[str]] = None,
) -> tuple[bool, int, int, Optional[str]]:
    """
    Verify a table was copied correctly.
    
    Returns: (is_ok, source_count, dest_count, error_message)
    
    Checks:
    1. Row counts match
    2. (Optional) Checksum of first/last rows match
    """
    try:
        table_name = _require_safe_identifier(table_name, "table name")
        source_conn = source_mgr.get(readonly=True)
        dest_conn = dest_mgr.get()

        with source_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            source_count = cur.fetchone()[0]

        with dest_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            dest_count = cur.fetchone()[0]

        if source_count != dest_count:
            return (
                False,
                source_count,
                dest_count,
                f"Row count mismatch: source={source_count:,} dest={dest_count:,} (delta={source_count - dest_count:,})",
            )

        # Optional: spot-check a few rows via hash
        if columns and source_count > 0:
            col_concat = " || ".join(f"COALESCE({c}::text, '')" for c in columns[:5])
            check_sql = f"SELECT md5(string_agg(row_hash, '')) FROM (SELECT md5({col_concat}) AS row_hash FROM {table_name} ORDER BY 1 LIMIT 100) sub"

            try:
                with source_conn.cursor() as cur:
                    cur.execute(check_sql)
                    source_hash = cur.fetchone()[0]

                with dest_conn.cursor() as cur:
                    cur.execute(check_sql)
                    dest_hash = cur.fetchone()[0]

                if source_hash != dest_hash:
                    return (False, source_count, dest_count, "Data hash mismatch on spot check")
            except Exception:
                pass  # Hash check is best-effort

        return (True, source_count, dest_count, None)

    except Exception as e:
        return (False, 0, 0, f"Verification error: {e}")


# =============================================================================
# COPY WITH RESUME SUPPORT
# =============================================================================

def copy_table_streaming(
    source_config: DBConfig,
    dest_config: DBConfig,
    table_name: str,
    columns: Optional[list[str]] = None,
    timeout: int = COPY_STATEMENT_TIMEOUT,
) -> int:
    """Stream entire table via pipe. For small-medium tables."""
    table_name = _require_safe_identifier(table_name, "table name")
    if columns:
        for col in columns:
            _require_safe_identifier(col, "column name")
    col_list = f"({', '.join(columns)})" if columns else ""

    copy_out_cmd = [
        "psql",
        *source_config.psql_args,
        "-c",
        f"\\copy {table_name} {col_list} TO STDOUT WITH (FORMAT csv, HEADER true, NULL '\\N')",
    ]

    copy_in_cmd = [
        "psql",
        *dest_config.psql_args,
        "-c",
        f"\\copy {table_name} {col_list} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N')",
    ]

    source_proc = subprocess.Popen(
        copy_out_cmd,
        env=source_config.psql_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    dest_proc = subprocess.Popen(
        copy_in_cmd,
        env=dest_config.psql_env,
        stdin=source_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    source_proc.stdout.close()

    try:
        dest_stdout, dest_stderr = dest_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        dest_proc.kill()
        source_proc.kill()
        raise TimeoutError(f"COPY timed out after {timeout}s for {table_name}")

    source_proc.wait()

    if dest_proc.returncode != 0:
        error_msg = dest_stderr.decode() if dest_stderr else "Unknown error"
        raise RuntimeError(f"COPY INTO {table_name} failed: {error_msg[:500]}")

    output = dest_stdout.decode() if dest_stdout else ""
    try:
        return int(output.strip().split()[-1])
    except (ValueError, IndexError):
        return 0


def copy_table_chunked_resumable(
    source_config: DBConfig,
    dest_config: DBConfig,
    table_name: str,
    pk_column: str,
    columns: Optional[list[str]] = None,
    chunk_size: int = 50_000,
    last_pk: Optional[str] = None,
    progress: Optional[MigrationProgress] = None,
    table_progress: Optional[TableProgress] = None,
) -> int:
    """
    Copy a large table in chunks with checkpoint resume.
    
    After each chunk:
    1. Records last PK copied to progress file
    2. If we crash, resume from that PK
    
    This is the key to surviving network disconnects on large tables.
    """
    table_name = _require_safe_identifier(table_name, "table name")
    pk_column = _require_safe_identifier(pk_column, "pk column")
    if columns:
        for col in columns:
            _require_safe_identifier(col, "column name")

    col_list = ', '.join(columns) if columns else '*'
    col_insert_list = f"({', '.join(columns)})" if columns else ""

    # Ensure PK is in the column list for ordering
    if columns and pk_column not in columns:
        columns = [pk_column] + columns
        col_list = ', '.join(columns)
        col_insert_list = f"({', '.join(columns)})"

    # Get total rows remaining
    source_conn = psycopg2.connect(source_config.dsn, connect_timeout=CONNECTION_TIMEOUT)
    source_conn.set_session(autocommit=True, readonly=True)

    with source_conn.cursor() as cur:
        if last_pk is not None:
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {pk_column} > %s", (last_pk,))
        else:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_remaining = cur.fetchone()[0]

    source_conn.close()

    if total_remaining == 0:
        return 0

    copied_this_run = 0
    current_last_pk = last_pk

    print(f"     📦 Chunked copy: {total_remaining:,} rows remaining"
          f"{f' (resuming after pk={last_pk})' if last_pk else ''}")

    while True:
        # Build chunk query and compute checkpoint PK for this chunk.
        if current_last_pk is not None:
            where_clause = f"WHERE {pk_column} > {_sql_literal(current_last_pk)}"
        else:
            where_clause = ""

        chunk_query = (
            f"SELECT {col_list} FROM {table_name} "
            f"{where_clause} "
            f"ORDER BY {pk_column} LIMIT {chunk_size}"
        )

        checkpoint_sql = f"SELECT MAX(t.{pk_column})::text FROM ({chunk_query}) t"
        source_conn = psycopg2.connect(source_config.dsn, connect_timeout=CONNECTION_TIMEOUT)
        source_conn.set_session(autocommit=True, readonly=True)
        with source_conn.cursor() as cur:
            cur.execute(checkpoint_sql)
            row = cur.fetchone()
            next_last_pk = row[0] if row and row[0] is not None else None
        source_conn.close()

        if next_last_pk is None:
            break

        # Stream this chunk
        copy_out_cmd = [
            "psql",
            *source_config.psql_args,
            "-c",
            f"\\copy ({chunk_query}) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '\\N')",
        ]

        copy_in_cmd = [
            "psql",
            *dest_config.psql_args,
            "-c",
            f"\\copy {table_name} {col_insert_list} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N')",
        ]

        source_proc = subprocess.Popen(
            copy_out_cmd,
            env=source_config.psql_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        dest_proc = subprocess.Popen(
            copy_in_cmd,
            env=dest_config.psql_env,
            stdin=source_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        source_proc.stdout.close()

        try:
            dest_stdout, dest_stderr = dest_proc.communicate(timeout=COPY_STATEMENT_TIMEOUT)
        except subprocess.TimeoutExpired:
            dest_proc.kill()
            source_proc.kill()
            # Save checkpoint before raising
            if table_progress and progress:
                table_progress.last_pk_copied = current_last_pk
                progress.save()
            raise TimeoutError(f"Chunk timed out for {table_name}")

        source_proc.wait()

        if dest_proc.returncode != 0:
            error_msg = dest_stderr.decode() if dest_stderr else ""
            # Save checkpoint before raising
            if table_progress and progress:
                table_progress.last_pk_copied = current_last_pk
                progress.save()
            raise RuntimeError(f"Chunk COPY failed: {error_msg[:300]}")

        # Parse rows copied in this chunk
        output = dest_stdout.decode() if dest_stdout else ""
        try:
            chunk_rows = int(output.strip().split()[-1])
        except (ValueError, IndexError):
            chunk_rows = 0

        if chunk_rows == 0:
            break  # No more data

        copied_this_run += chunk_rows
        current_last_pk = next_last_pk

        # === CHECKPOINT: Save progress after every chunk ===
        if table_progress and progress:
            table_progress.last_pk_copied = current_last_pk
            table_progress.rows_copied_so_far += chunk_rows
            progress.save()

        pct = copied_this_run / total_remaining * 100 if total_remaining > 0 else 100
        print(f"\r     Progress: {copied_this_run:,}/{total_remaining:,} ({pct:.1f}%) "
              f"[checkpoint: pk={current_last_pk}]", end="", flush=True)

        if chunk_rows < chunk_size:
            break  # Last chunk was partial — we're done

    print()  # newline after progress
    return copied_this_run


# =============================================================================
# SINGLE TABLE COPY ORCHESTRATOR
# =============================================================================

LARGE_TABLE_THRESHOLD = 1_000_000  # 1M rows → use chunked copy


def copy_single_table(
    source_config: DBConfig,
    dest_config: DBConfig,
    source_mgr: ConnectionManager,
    dest_mgr: ConnectionManager,
    table_name: str,
    tp: TableProgress,
    progress: MigrationProgress,
    skip_rdkit: bool = False,
) -> bool:
    """
    Copy a single table with full retry and checkpoint logic.
    Returns True if successful.
    """
    table_name = _require_safe_identifier(table_name, "table name")
    tp.status = TableStatus.IN_PROGRESS
    tp.attempts += 1
    tp.started_at = datetime.now(timezone.utc).isoformat()
    progress.save()

    start = time.time()

    try:
        # Determine columns
        if skip_rdkit:
            columns, skipped = get_non_rdkit_columns(source_mgr, table_name)
            tp.columns_copied = columns
            tp.columns_skipped = skipped
            if skipped:
                print(f"     ⚠️  Skipping RDKit columns: {', '.join(skipped)}")
        else:
            columns = None

        # Get source row count
        source_count = get_exact_row_count(source_mgr, table_name)
        tp.source_rows = source_count

        if source_count == 0:
            tp.status = TableStatus.VERIFIED
            tp.dest_rows = 0
            tp.completed_at = datetime.now(timezone.utc).isoformat()
            tp.elapsed_seconds = time.time() - start
            progress.save()
            print(f"     ⏭️  Empty table")
            return True

        # Check if we're resuming a partial chunked copy
        is_resume = (
            tp.last_pk_copied is not None
            and tp.rows_copied_so_far > 0
            and tp.pk_column is not None
        )

        if is_resume:
            print(f"     ⏩ Resuming from pk={tp.last_pk_copied} "
                  f"({tp.rows_copied_so_far:,} rows already copied)")

        # Decide: streaming vs chunked
        use_chunked = (
            tp.pk_column is not None
            and (source_count > LARGE_TABLE_THRESHOLD or is_resume)
        )

        if not is_resume:
            # Clean slate: truncate destination table
            try:
                dest_conn = dest_mgr.get()
                with dest_conn.cursor() as cur:
                    cur.execute(f"TRUNCATE TABLE {table_name}")
            except Exception as e:
                print(f"     ⚠️  Truncate warning: {e}")

        if use_chunked:
            count = retry_operation(
                copy_table_chunked_resumable,
                source_config,
                dest_config,
                table_name,
                tp.pk_column,
                columns=columns,
                chunk_size=50_000,
                last_pk=tp.last_pk_copied if is_resume else None,
                progress=progress,
                table_progress=tp,
                operation_name=f"chunked copy {table_name}",
            )
        else:
            count = retry_operation(
                copy_table_streaming,
                source_config,
                dest_config,
                table_name,
                columns=columns,
                operation_name=f"stream copy {table_name}",
            )

        tp.status = TableStatus.COPIED
        tp.elapsed_seconds = time.time() - start
        progress.save()

        # Verify
        ok, src_count, dst_count, err = verify_table(
            source_mgr, dest_mgr, table_name, columns=columns
        )
        tp.source_rows = src_count
        tp.dest_rows = dst_count

        if ok:
            tp.status = TableStatus.VERIFIED
            tp.completed_at = datetime.now(timezone.utc).isoformat()
            rate = src_count / tp.elapsed_seconds if tp.elapsed_seconds > 0 else 0
            print(f"     ✅ Verified: {src_count:,} rows in {tp.elapsed_seconds:.1f}s "
                  f"({rate:,.0f} rows/s)")
        else:
            tp.status = TableStatus.FAILED
            tp.last_error = err
            print(f"     ❌ Verification failed: {err}")

        progress.save()
        return ok

    except Exception as e:
        tp.status = TableStatus.FAILED
        tp.last_error = str(e)[:500]
        tp.elapsed_seconds = time.time() - start
        progress.save()
        print(f"     ❌ Failed after {tp.elapsed_seconds:.1f}s: {e}")
        return False


# =============================================================================
# SCHEMA, EXTENSIONS, FUNCTIONS, VIEWS
# =============================================================================

def ensure_extensions(dest_mgr: ConnectionManager, skip_rdkit: bool) -> None:
    """Create required extensions on Railway."""
    extensions = ["pg_trgm", "unaccent", "pgvector"]
    if not skip_rdkit:
        extensions.append("rdkit")

    conn = dest_mgr.get()
    with conn.cursor() as cur:
        for ext in extensions:
            try:
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                print(f"  ✅ {ext}")
            except Exception as e:
                print(f"  ⚠️  {ext}: {e}")


def migrate_schema(source_config: DBConfig, dest_config: DBConfig, skip_rdkit: bool) -> None:
    """Dump schema from source and restore on destination."""
    print("📋 Dumping schema...")

    cmd = [
        "pg_dump",
        *source_config.psql_args,
        "--schema-only",
        "--no-owner",
        "--no-privileges",
        "--format=plain",
    ]

    result = subprocess.run(cmd, env=source_config.psql_env, capture_output=True, text=True)
    schema_sql = result.stdout

    if skip_rdkit:
        # Remove RDKit-dependent lines
        filtered_lines = []
        for line in schema_sql.split("\n"):
            lower = line.lower()
            if "create extension" in lower and "rdkit" in lower:
                continue
            if any(f" {t}" in lower for t in RDKIT_TYPES):
                # Check if it's a column definition inside CREATE TABLE
                stripped = line.strip().rstrip(",")
                if any(stripped.lower().endswith(f" {t}") or f" {t} " in stripped.lower()
                       for t in RDKIT_TYPES):
                    print(f"  ⚠️  Skipping: {stripped[:80]}")
                    continue
            if any(kw in lower for kw in ["mol_from_smiles", "morganbv_fp", "featmorganbv_fp",
                                           "gist_bfp_ops", "gist(mol"]):
                print(f"  ⚠️  Skipping: {line.strip()[:80]}")
                continue
            filtered_lines.append(line)
        schema_sql = "\n".join(filtered_lines)

    schema_path = Path("/tmp/migration_schema.sql")
    schema_path.write_text(schema_sql)

    print("  📥 Restoring schema on Railway...")
    result = subprocess.run(
        ["psql", *dest_config.psql_args, "-f", str(schema_path)],
        env=dest_config.psql_env,
        capture_output=True,
        text=True,
    )

    errors = [l for l in (result.stderr or "").split("\n") if "ERROR" in l and "already exists" not in l]
    if errors:
        print(f"  ⚠️  {len(errors)} schema errors (may be OK on resume):")
        for e in errors[:5]:
            print(f"     {e}")
    else:
        print("  ✅ Schema restored")


def restore_functions_and_views(
    source_mgr: ConnectionManager,
    dest_mgr: ConnectionManager,
    skip_rdkit: bool,
) -> None:
    """Restore functions and views after all data is copied."""
    source_conn = source_mgr.get(readonly=True)
    dest_conn = dest_mgr.get()

    # Functions
    print("\n🔧 Restoring functions...")
    with source_conn.cursor() as cur:
        cur.execute("""
            SELECT p.proname, pg_get_functiondef(p.oid)
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public' AND p.prokind IN ('f', 'p')
            ORDER BY p.proname
        """)
        functions = cur.fetchall()

    with dest_conn.cursor() as cur:
        for name, definition in functions:
            if skip_rdkit and any(kw in definition.lower() for kw in
                                  ["mol_from_smiles", "morganbv_fp", "featmorganbv_fp",
                                   " mol ", " bfp ", " sfp "]):
                print(f"  ⏭️  Skipping RDKit function: {name}")
                continue
            try:
                cur.execute(definition)
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ⚠️  {name}: {str(e)[:100]}")

    # Views
    print("\n👁️  Restoring views...")
    with source_conn.cursor() as cur:
        cur.execute("""
            SELECT viewname, definition
            FROM pg_views
            WHERE schemaname = 'public'
            ORDER BY viewname
        """)
        views = cur.fetchall()

    with dest_conn.cursor() as cur:
        for name, definition in views:
            try:
                cur.execute(f"CREATE OR REPLACE VIEW {name} AS {definition}")
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ⚠️  {name}: {str(e)[:100]}")


def reset_sequences(dest_mgr: ConnectionManager) -> None:
    """Reset all sequences to match data."""
    print("\n🔢 Resetting sequences...")
    conn = dest_mgr.get()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT seq.relname, tab.relname, a.attname
            FROM pg_class seq
            JOIN pg_namespace ns ON seq.relnamespace = ns.oid
            JOIN pg_depend d ON d.objid = seq.oid
            JOIN pg_class tab ON d.refobjid = tab.oid
            JOIN pg_attribute a ON a.attrelid = tab.oid AND a.attnum = d.refobjsubid
            WHERE seq.relkind = 'S' AND ns.nspname = 'public'
        """)
        for seq_name, table_name, col_name in cur.fetchall():
            try:
                cur.execute(f"SELECT setval('{seq_name}', COALESCE(MAX({col_name}), 1)) FROM {table_name}")
                print(f"  ✅ {seq_name}")
            except Exception as e:
                print(f"  ⚠️  {seq_name}: {e}")


# =============================================================================
# MAIN MIGRATION ORCHESTRATOR
# =============================================================================

def migrate(
    source_config: DBConfig,
    dest_config: DBConfig,
    skip_rdkit: bool = False,
    resume: bool = False,
    verify_only: bool = False,
    retry_failed: bool = False,
    single_table: Optional[str] = None,
    progress_file: Optional[Path] = None,
):
    global PROGRESS_FILE
    if progress_file is not None:
        PROGRESS_FILE = progress_file

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Railway Migration — Resilient Copy with Checkpoint Resume    ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Load or create progress
    if resume or retry_failed or verify_only:
        if not PROGRESS_FILE.exists():
            print(f"❌ No progress file found at {PROGRESS_FILE}")
            print("   Run without --resume first to start a new migration.")
            return
        progress = MigrationProgress.load()
        print(f"📂 Loaded progress from {PROGRESS_FILE}")
        print(f"   Last updated: {progress.last_updated}")
        summary = progress.summary()
        print(f"   Status: {summary}\n")
    else:
        progress = MigrationProgress()
        progress.started_at = datetime.now(timezone.utc).isoformat()
        progress.skip_rdkit = skip_rdkit

    source_mgr = ConnectionManager(source_config, "source")
    dest_mgr = ConnectionManager(dest_config, "railway")

    # Test connections
    print("🔌 Testing connections...")
    try:
        source_mgr.get(readonly=True)
        print("  ✅ Source connected")
    except Exception as e:
        print(f"  ❌ Source connection failed: {e}")
        return

    try:
        dest_mgr.get()
        print("  ✅ Railway connected")
    except Exception as e:
        print(f"  ❌ Railway connection failed: {e}")
        return

    # === Extensions ===
    if not progress.extensions_created:
        print("\n🔧 Creating extensions...")
        ensure_extensions(dest_mgr, skip_rdkit)
        progress.extensions_created = True
        progress.save()

    # === Schema ===
    if not progress.schema_migrated and not verify_only:
        print("\n📋 Migrating schema...")
        migrate_schema(source_config, dest_config, skip_rdkit)
        progress.schema_migrated = True
        progress.save()

    # === Discover tables ===
    print("\n📊 Discovering tables...")
    tables = discover_tables(source_mgr)
    print(f"   Found {len(tables)} tables/materialized views")

    # Initialize progress entries for new tables
    for t in tables:
        name = t["name"]
        if name not in progress.tables:
            progress.tables[name] = TableProgress(
                name=name,
                source_rows=t["approx_rows"],
                size_bytes=t["size_bytes"],
                pk_column=t.get("pk_column"),
                is_materialized_view=t.get("is_materialized_view", False),
            )
        else:
            # Update PK info in case it wasn't set
            if t.get("pk_column"):
                progress.tables[name].pk_column = t["pk_column"]
    progress.save()

    # === Determine which tables to process ===
    if single_table:
        _require_safe_identifier(single_table, "table name")
        to_process = [single_table]
    elif retry_failed:
        to_process = [
            name for name, tp in progress.tables.items()
            if tp.status == TableStatus.FAILED
        ]
        print(f"\n🔄 Retrying {len(to_process)} failed tables")
    elif verify_only:
        to_process = [
            name for name, tp in progress.tables.items()
            if tp.status in (TableStatus.COPIED, TableStatus.VERIFIED, TableStatus.INDEXED)
        ]
        print(f"\n🔍 Verifying {len(to_process)} tables")
    else:
        # Process all non-completed tables
        to_process = [
            name for name, tp in progress.tables.items()
            if tp.status not in (TableStatus.VERIFIED, TableStatus.INDEXED, TableStatus.SKIPPED)
        ]

    # Sort by size (smallest first for quick wins, or largest first for throughput)
    to_process.sort(key=lambda n: progress.tables[n].size_bytes)

    # === Print plan ===
    total_size = sum(progress.tables[n].size_bytes for n in to_process)
    print(f"\n📋 Migration plan: {len(to_process)} tables, ~{total_size / (1024**3):.1f} GB\n")

    for i, name in enumerate(to_process, 1):
        tp = progress.tables[name]
        size_mb = tp.size_bytes / (1024**2)
        status_icon = {
            TableStatus.PENDING: "⬜",
            TableStatus.IN_PROGRESS: "🔄",
            TableStatus.COPIED: "📦",
            TableStatus.VERIFIED: "✅",
            TableStatus.INDEXED: "📇",
            TableStatus.FAILED: "❌",
            TableStatus.SKIPPED: "⏭️",
        }.get(tp.status, "❓")
        resume_note = f" [resume from pk={tp.last_pk_copied}]" if tp.last_pk_copied else ""
        print(f"   {status_icon} {i:3}. {name:<45} {size_mb:>8.1f} MB  "
              f"{tp.source_rows:>12,} rows{resume_note}")

    if verify_only:
        print("\n🔍 Running verification...")
        for name in to_process:
            tp = progress.tables[name]
            ok, src, dst, err = verify_table(source_mgr, dest_mgr, name)
            if ok:
                print(f"  ✅ {name}: {src:,} rows match")
            else:
                print(f"  ❌ {name}: {err}")
        return

    # === Copy tables ===
    print(f"\n{'='*70}")
    print("📦 Starting data copy...\n")

    succeeded = 0
    failed = 0

    for i, name in enumerate(to_process, 1):
        tp = progress.tables[name]
        size_mb = tp.size_bytes / (1024**2)

        print(f"\n[{i}/{len(to_process)}] {name} ({size_mb:.1f} MB, ~{tp.source_rows:,} rows)"
              f" [attempt {tp.attempts + 1}]")

        if tp.is_materialized_view:
            print(f"     📊 Materialized view — copying as table")

        ok = copy_single_table(
            source_config, dest_config,
            source_mgr, dest_mgr,
            name, tp, progress,
            skip_rdkit=skip_rdkit,
        )

        if ok:
            succeeded += 1
        else:
            failed += 1

    # === Post-copy: functions, views, sequences ===
    if not progress.functions_restored:
        print("\n🔧 Restoring functions and views...")
        restore_functions_and_views(source_mgr, dest_mgr, skip_rdkit)
        progress.functions_restored = True
        progress.views_restored = True
        progress.save()

    if not progress.sequences_reset:
        reset_sequences(dest_mgr)
        progress.sequences_reset = True
        progress.save()

    # === Final summary ===
    source_mgr.close()
    dest_mgr.close()

    summary = progress.summary()
    print(f"\n{'='*70}")
    print("🏁 MIGRATION SUMMARY")
    print(f"   {summary}")
    print(f"   Succeeded this run: {succeeded}")
    print(f"   Failed this run:    {failed}")
    print(f"   Progress file:      {PROGRESS_FILE}")

    if failed > 0:
        print(f"\n   To retry failed tables:")
        print(f"   python scripts/migration.py {'--skip-rdkit ' if skip_rdkit else ''}--retry-failed")

    print(f"\n   To verify all tables:")
    print(f"   python scripts/migration.py --verify-only")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    _load_repo_env()
    parser = argparse.ArgumentParser(description="Resilient Railway migration")
    parser.add_argument("--skip-rdkit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't copy")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed tables")
    parser.add_argument("--table", type=str, help="Migrate a single table")
    parser.add_argument(
        "--source-dsn",
        default=_first_env("MIGRATE_SOURCE_DSN", "DATA_POSTGRES_URI", "DATA_POSTGRES_URL", "DATA_DATABASE_URL") or DEFAULT_SOURCE_DSN,
        help="Source PostgreSQL DSN (env: MIGRATE_SOURCE_DSN, DATA_POSTGRES_URI, DATA_POSTGRES_URL, DATA_DATABASE_URL).",
    )
    parser.add_argument(
        "--dest-dsn",
        default=os.getenv("MIGRATE_DEST_DSN")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DEST_DSN,
        help="Destination PostgreSQL DSN (env: MIGRATE_DEST_DSN, DATABASE_URL).",
    )
    parser.add_argument(
        "--progress-file",
        default=os.getenv("MIGRATE_PROGRESS_FILE", str(PROGRESS_FILE)),
        help=f"Progress JSON path (default: {PROGRESS_FILE}).",
    )
    args = parser.parse_args()

    try:
        source_cfg, dest_cfg = parse_db_config(args.source_dsn, args.dest_dsn)
    except Exception as exc:
        print(f"❌ Invalid DSN: {exc}", file=sys.stderr)
        raise SystemExit(2)

    migrate(
        source_config=source_cfg,
        dest_config=dest_cfg,
        skip_rdkit=args.skip_rdkit,
        resume=args.resume,
        verify_only=args.verify_only,
        retry_failed=args.retry_failed,
        single_table=args.table,
        progress_file=Path(args.progress_file),
    )
