#!/usr/bin/env python3
"""
Sync selected PostgreSQL tables from source -> destination using pg_dump/pg_restore.

The script can expand your seed tables by FK dependencies before dumping/restoring.
This helps avoid FK constraint failures when syncing parent/child table groups.

Examples:
  python scripts/sync_tables_via_pgrestore.py \
    --source-uri "postgresql://user:pass@src-host:5432/srcdb" \
    --dest-uri "postgresql://user:pass@dst-host:5432/dstdb" \
    --tables dm_molecule dm_biotherapeutic \
    --dependency-mode both \
    --yes

  # Dry-run only (no changes)
  python scripts/sync_tables_via_pgrestore.py \
    --source-uri "$SOURCE_URI" \
    --dest-uri "$DEST_URI" \
    --tables public.dm_molecule public.dm_biotherapeutic \
    --dependency-mode children
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql


def _load_repo_env() -> None:
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    load_dotenv(override=False)


def _normalize_table_name(raw: str, default_schema: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty table name is not allowed")
    if "." in value:
        schema, table = value.split(".", 1)
        schema = schema.strip()
        table = table.strip()
    else:
        schema = default_schema
        table = value
    if not schema or not table:
        raise ValueError(f"Invalid table name: '{raw}'")
    return f"{schema}.{table}"


def _split_table_name(qualified: str) -> tuple[str, str]:
    schema, table = qualified.split(".", 1)
    return schema, table


def _connect(uri: str):
    return psycopg2.connect(uri)


def _table_exists(conn, qualified_table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (qualified_table,))
        row = cur.fetchone()
        return bool(row and row[0])


def _get_table_columns(conn, qualified_table: str) -> list[tuple[str, str]]:
    schema, table = _split_table_name(qualified_table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                a.attname,
                format_type(a.atttypid, a.atttypmod) AS type_name
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (schema, table),
        )
        rows = cur.fetchall()
    return [(str(name), str(type_name)) for name, type_name in rows]


def _sync_missing_columns(
    source_uri: str,
    dest_uri: str,
    tables: list[str],
) -> list[str]:
    """
    Add columns that exist in source table schemas but are missing in destination.
    Returns a list of added columns in "schema.table.column type" format.
    """
    src_conn = _connect(source_uri)
    dst_conn = _connect(dest_uri)
    added: list[str] = []
    try:
        dst_conn.autocommit = False
        with dst_conn.cursor() as cur:
            for qualified_table in tables:
                src_cols = _get_table_columns(src_conn, qualified_table)
                dst_cols = _get_table_columns(dst_conn, qualified_table)
                dst_col_names = {name for name, _ in dst_cols}
                schema, table = _split_table_name(qualified_table)

                for col_name, col_type in src_cols:
                    if col_name in dst_col_names:
                        continue
                    cur.execute(
                        sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} {}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.Identifier(col_name),
                            sql.SQL(col_type),
                        )
                    )
                    added.append(f"{qualified_table}.{col_name} {col_type}")
        dst_conn.commit()
    except Exception:
        dst_conn.rollback()
        raise
    finally:
        src_conn.close()
        dst_conn.close()
    return added


def _load_fk_edges(conn) -> list[tuple[str, str, str]]:
    """
    Return FK edges as tuples:
      (child_table_qualified, parent_table_qualified, constraint_name)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ns_child.nspname AS child_schema,
                child.relname AS child_table,
                ns_parent.nspname AS parent_schema,
                parent.relname AS parent_table,
                con.conname
            FROM pg_constraint con
            JOIN pg_class child ON child.oid = con.conrelid
            JOIN pg_namespace ns_child ON ns_child.oid = child.relnamespace
            JOIN pg_class parent ON parent.oid = con.confrelid
            JOIN pg_namespace ns_parent ON ns_parent.oid = parent.relnamespace
            WHERE con.contype = 'f'
            ORDER BY 1, 2, 3, 4, 5
            """
        )
        rows = cur.fetchall()

    edges: list[tuple[str, str, str]] = []
    for child_schema, child_table, parent_schema, parent_table, conname in rows:
        child_q = f"{child_schema}.{child_table}"
        parent_q = f"{parent_schema}.{parent_table}"
        edges.append((child_q, parent_q, conname))
    return edges


def _expand_tables_by_dependencies(
    seed_tables: set[str],
    fk_edges: list[tuple[str, str, str]],
    mode: str,
    recursive: bool,
) -> set[str]:
    selected = set(seed_tables)

    if mode == "none":
        return selected

    def step(current: set[str]) -> set[str]:
        additions: set[str] = set()
        for child, parent, _ in fk_edges:
            if mode in {"children", "both"} and parent in current:
                additions.add(child)
            if mode in {"parents", "both"} and child in current:
                additions.add(parent)
        return additions

    if not recursive:
        selected |= step(selected)
        return selected

    while True:
        additions = step(selected) - selected
        if not additions:
            break
        selected |= additions
    return selected


def _constraints_touching_tables(
    selected_tables: set[str],
    fk_edges: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    return [
        (child, parent, conname)
        for child, parent, conname in fk_edges
        if child in selected_tables or parent in selected_tables
    ]


def _run_cmd(cmd: list[str]) -> None:
    print(f"▶ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _dump_selected_tables(source_uri: str, tables: list[str], dump_file: Path) -> None:
    cmd = [
        "pg_dump",
        source_uri,
        "-Fc",
        "--data-only",
        "--no-owner",
        "--no-privileges",
        "-f",
        str(dump_file),
    ]
    for table in tables:
        cmd.extend(["-t", table])
    _run_cmd(cmd)


def _truncate_destination_tables(dest_uri: str, tables: list[str], cascade: bool) -> None:
    conn = _connect(dest_uri)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            identifiers = [
                sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
                for schema, table in (_split_table_name(t) for t in tables)
            ]
            truncate_sql = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY{}").format(
                sql.SQL(", ").join(identifiers),
                sql.SQL(" CASCADE") if cascade else sql.SQL(""),
            )
            cur.execute(truncate_sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _restore_dump(dest_uri: str, dump_file: Path) -> None:
    list_result = subprocess.run(
        ["pg_restore", "-l", str(dump_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    toc_lines = list_result.stdout.splitlines()
    filtered_toc_lines = [line for line in toc_lines if " SEQUENCE SET " not in line]

    fd, list_path = tempfile.mkstemp(prefix="pg_restore_list_", suffix=".txt")
    os.close(fd)
    use_list = Path(list_path)
    try:
        use_list.write_text("\n".join(filtered_toc_lines) + "\n", encoding="utf-8")
        cmd = [
            "pg_restore",
            f"--dbname={dest_uri}",
            "--data-only",
            "--no-owner",
            "--no-privileges",
            "--single-transaction",
            "--disable-triggers",
            f"--use-list={use_list}",
            str(dump_file),
        ]
        _run_cmd(cmd)
    finally:
        use_list.unlink(missing_ok=True)


def _sync_sequences_for_tables(dest_uri: str, tables: list[str]) -> list[str]:
    """
    Re-sync serial/identity sequences for selected tables after restore.
    Returns "schema.table.column -> sequence" entries that were synchronized.
    """
    conn = _connect(dest_uri)
    conn.autocommit = False
    synced: list[str] = []
    try:
        with conn.cursor() as cur:
            for qualified_table in tables:
                schema, table = _split_table_name(qualified_table)
                cur.execute(
                    """
                    SELECT a.attname
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s
                      AND c.relname = %s
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    ORDER BY a.attnum
                    """,
                    (schema, table),
                )
                columns = [row[0] for row in cur.fetchall()]

                for column in columns:
                    cur.execute("SELECT pg_get_serial_sequence(%s, %s)", (qualified_table, column))
                    seq_row = cur.fetchone()
                    sequence_name = seq_row[0] if seq_row else None
                    if not sequence_name:
                        continue

                    setval_sql = sql.SQL(
                        """
                        SELECT pg_catalog.setval(
                            %s,
                            COALESCE((SELECT MAX({col})::bigint FROM {schema}.{table}), 1),
                            EXISTS (SELECT 1 FROM {schema}.{table})
                        )
                        """
                    ).format(
                        col=sql.Identifier(column),
                        schema=sql.Identifier(schema),
                        table=sql.Identifier(table),
                    )
                    cur.execute(setval_sql, (sequence_name,))
                    synced.append(f"{qualified_table}.{column} -> {sequence_name}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return synced


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync selected tables source->destination using pg_dump/pg_restore."
    )
    parser.add_argument(
        "--source-uri",
        default=(os.getenv("SYNC_SOURCE_URI") or "").strip(),
        help="Source PostgreSQL URI (env: SYNC_SOURCE_URI)",
    )
    parser.add_argument(
        "--dest-uri",
        default=(os.getenv("SYNC_DEST_URI") or "").strip(),
        help="Destination PostgreSQL URI (env: SYNC_DEST_URI)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        required=True,
        help="Seed table names to sync (e.g. dm_molecule public.dm_biotherapeutic)",
    )
    parser.add_argument(
        "--default-schema",
        default="public",
        help="Schema to use when table is unqualified (default: public)",
    )
    parser.add_argument(
        "--dependency-mode",
        choices=["none", "children", "parents", "both"],
        default="both",
        help="How to expand seed tables using FK graph (default: both)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only expand one dependency level instead of recursive closure",
    )
    parser.add_argument(
        "--dump-file",
        default="",
        help="Path to dump file (.dump). Defaults to a temp file.",
    )
    parser.add_argument(
        "--keep-dump",
        action="store_true",
        help="Keep dump file when using temp location.",
    )
    parser.add_argument(
        "--truncate-cascade",
        action="store_true",
        help="Use TRUNCATE ... CASCADE on destination.",
    )
    parser.add_argument(
        "--no-sync-missing-columns",
        action="store_true",
        help="Do not auto-add missing destination columns from source schema.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to perform truncate/restore. Without this, script runs dry-run only.",
    )
    return parser.parse_args()


def main() -> int:
    _load_repo_env()
    args = _parse_args()

    if not args.source_uri:
        print("❌ Missing --source-uri (or SYNC_SOURCE_URI).", file=sys.stderr)
        return 2
    if not args.dest_uri:
        print("❌ Missing --dest-uri (or SYNC_DEST_URI).", file=sys.stderr)
        return 2

    try:
        seed_tables = {
            _normalize_table_name(t, args.default_schema) for t in args.tables
        }
    except ValueError as exc:
        print(f"❌ Invalid table list: {exc}", file=sys.stderr)
        return 2

    src_conn = _connect(args.source_uri)
    dst_conn = _connect(args.dest_uri)
    try:
        missing_src = [t for t in sorted(seed_tables) if not _table_exists(src_conn, t)]
        if missing_src:
            print("❌ These seed tables do not exist in source DB:", file=sys.stderr)
            for t in missing_src:
                print(f"   - {t}", file=sys.stderr)
            return 2

        fk_edges = _load_fk_edges(src_conn)
        selected_tables = _expand_tables_by_dependencies(
            seed_tables=seed_tables,
            fk_edges=fk_edges,
            mode=args.dependency_mode,
            recursive=not args.no_recursive,
        )

        missing_dst = [t for t in sorted(selected_tables) if not _table_exists(dst_conn, t)]
        if missing_dst:
            print("❌ These selected tables do not exist in destination DB:", file=sys.stderr)
            for t in missing_dst:
                print(f"   - {t}", file=sys.stderr)
            return 2

        touched_constraints = _constraints_touching_tables(selected_tables, fk_edges)
    finally:
        src_conn.close()
        dst_conn.close()

    ordered_tables = sorted(selected_tables)

    print("\n=== Sync Plan ===")
    print(f"Seed tables ({len(seed_tables)}):")
    for t in sorted(seed_tables):
        print(f"  - {t}")
    print(
        f"\nDependency mode: {args.dependency_mode}"
        f"{' (non-recursive)' if args.no_recursive else ' (recursive)'}"
    )
    print(f"Selected tables to sync ({len(ordered_tables)}):")
    for t in ordered_tables:
        print(f"  - {t}")

    print(f"\nFK constraints touching selected tables ({len(touched_constraints)}):")
    for child, parent, conname in touched_constraints:
        print(f"  - {conname}: {child} -> {parent}")

    if not args.yes:
        print("\nDry-run complete. Re-run with --yes to execute dump + truncate + restore.")
        return 0

    dump_file: Path
    using_temp = False
    if args.dump_file:
        dump_file = Path(args.dump_file).resolve()
        dump_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_path = tempfile.mkstemp(prefix="pg_sync_", suffix=".dump")
        os.close(fd)
        dump_file = Path(tmp_path)
        using_temp = True

    try:
        if args.no_sync_missing_columns:
            print("\nℹ️  Skipping destination schema alignment (--no-sync-missing-columns).")
        else:
            print("\n🧩 Aligning destination schema: adding missing columns...")
            added_columns = _sync_missing_columns(args.source_uri, args.dest_uri, ordered_tables)
            if added_columns:
                print(f"  ✅ Added {len(added_columns)} missing columns:")
                for item in added_columns:
                    print(f"     - {item}")
            else:
                print("  ✅ No missing columns detected.")

        print(f"\n📦 Creating dump: {dump_file}")
        _dump_selected_tables(args.source_uri, ordered_tables, dump_file)

        print("\n🧹 Truncating destination tables...")
        _truncate_destination_tables(args.dest_uri, ordered_tables, cascade=args.truncate_cascade)

        print("\n♻️ Restoring dump into destination...")
        _restore_dump(args.dest_uri, dump_file)

        print("\n🔢 Re-syncing destination sequences to max IDs...")
        synced_sequences = _sync_sequences_for_tables(args.dest_uri, ordered_tables)
        if synced_sequences:
            print(f"  ✅ Synced {len(synced_sequences)} sequences:")
            for item in synced_sequences:
                print(f"     - {item}")
        else:
            print("  ✅ No serial/identity sequences found for selected tables.")

        print("\n✅ Sync complete.")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n❌ Sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if using_temp and not args.keep_dump and dump_file.exists():
            dump_file.unlink(missing_ok=True)
        elif dump_file.exists():
            print(f"ℹ️  Dump file kept at: {dump_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

