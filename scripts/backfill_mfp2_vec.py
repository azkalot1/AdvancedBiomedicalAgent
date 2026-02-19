#!/usr/bin/env python3
"""
Backfill dm_molecule.mfp2_vec from canonical_smiles using Python RDKit.

Usage:
  python scripts/backfill_mfp2_vec.py --dsn postgresql://user:pass@host:5432/db
  python scripts/backfill_mfp2_vec.py --dsn postgresql://user:pass@host:5432/db --create-index
  python scripts/backfill_mfp2_vec.py --dsn postgresql://user:pass@host:5432/db --reset-column
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_batch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


FP_BITS = 1024
FP_RADIUS = 2  # Morgan radius 2 (ECFP4)
DEFAULT_BATCH_SIZE = 1000
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def _load_repo_env() -> None:
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    load_dotenv(override=False)


def _legacy_dsn_from_parts() -> str:
    host = (os.getenv("DB_HOST") or "").strip()
    port = (os.getenv("DB_PORT") or "").strip()
    database = (os.getenv("DB_NAME") or "").strip()
    user = (os.getenv("DB_USER") or "").strip()
    password = os.getenv("DB_PASSWORD")
    if not all([host, port, database, user]) or password is None:
        return ""
    return (
        "postgresql://"
        f"{quote(user, safe='')}:{quote(password, safe='')}@"
        f"{host}:{port}/{database}"
    )


def _default_dsn() -> str:
    for key in ("DATA_POSTGRES_URI", "DATA_POSTGRES_URL", "DATA_DATABASE_URL"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return _legacy_dsn_from_parts()


def smiles_to_vector_literal(smiles: str) -> str | None:
    """Convert SMILES to pgvector literal '[0,1,...]'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = MORGAN_GENERATOR.GetFingerprint(mol)
    vec = [0] * FP_BITS
    for idx in fp.GetOnBits():
        vec[idx] = 1
    return "[" + ",".join(str(v) for v in vec) + "]"


def ensure_column(cur, reset_column: bool = False) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'dm_molecule'
          AND a.attname = 'mfp2_vec'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
    )
    row = cur.fetchone()
    expected = f"vector({FP_BITS})"
    if row is None:
        cur.execute(f"ALTER TABLE dm_molecule ADD COLUMN mfp2_vec {expected}")
        return

    actual = row[0]
    if actual == expected:
        return

    if not reset_column:
        raise RuntimeError(
            f"Existing dm_molecule.mfp2_vec type is {actual}, expected {expected}. "
            "Rerun with --reset-column to recreate it."
        )

    cur.execute("DROP INDEX IF EXISTS idx_dm_molecule_mfp2_vec")
    cur.execute("ALTER TABLE dm_molecule DROP COLUMN mfp2_vec")
    cur.execute(f"ALTER TABLE dm_molecule ADD COLUMN mfp2_vec {expected}")


def create_hnsw_index(cur, m: int, ef_construction: int) -> None:
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_dm_molecule_mfp2_vec
        ON dm_molecule
        USING hnsw (mfp2_vec vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
        """
    )


def backfill(
    dsn: str,
    batch_size: int,
    create_index: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
    reset_column: bool,
) -> int:
    started = time.time()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    with conn.cursor() as cur:
        ensure_column(cur, reset_column=reset_column)
        conn.commit()

        cur.execute(
            """
            SELECT COUNT(*)
            FROM dm_molecule
            WHERE canonical_smiles IS NOT NULL
              AND mfp2_vec IS NULL
            """
        )
        total = cur.fetchone()[0]
        print(f"Total molecules to backfill: {total:,}")

    if total == 0:
        if create_index:
            print("No rows need backfill; creating index...")
            with conn.cursor() as cur:
                create_hnsw_index(cur, hnsw_m, hnsw_ef_construction)
            conn.commit()
        conn.close()
        return 0

    processed = 0
    updated = 0

    last_mol_id = 0
    while True:
        with conn.cursor() as read_cur:
            read_cur.execute(
                """
                SELECT mol_id, canonical_smiles
                FROM dm_molecule
                WHERE canonical_smiles IS NOT NULL
                  AND mfp2_vec IS NULL
                  AND mol_id > %s
                ORDER BY mol_id
                LIMIT %s
                """,
                (last_mol_id, batch_size),
            )
            rows = read_cur.fetchall()

        if not rows:
            break

        last_mol_id = rows[-1][0]
        updates: list[tuple[str, int]] = []
        for mol_id, smiles in rows:
            vec_literal = smiles_to_vector_literal(smiles)
            if vec_literal is not None:
                updates.append((vec_literal, mol_id))

        if updates:
            with conn.cursor() as write_cur:
                execute_batch(
                    write_cur,
                    f"UPDATE dm_molecule SET mfp2_vec = %s::vector({FP_BITS}) WHERE mol_id = %s",
                    updates,
                    page_size=200,
                )
            conn.commit()
            updated += len(updates)

        processed += len(rows)
        pct = (processed / total) * 100 if total else 100
        print(f"\rProcessed {processed:,}/{total:,} ({pct:.1f}%), updated {updated:,}", end="", flush=True)

    if create_index:
        print("\nCreating HNSW index idx_dm_molecule_mfp2_vec ...")
        with conn.cursor() as cur:
            create_hnsw_index(cur, hnsw_m, hnsw_ef_construction)
        conn.commit()
        print("Index created.")

    conn.close()
    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.1f}s. Updated {updated:,} molecules.")
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill dm_molecule.mfp2_vec using RDKit Morgan fingerprints.")
    parser.add_argument(
        "--dsn",
        default=_default_dsn(),
        help="PostgreSQL DSN. Defaults to DATA_POSTGRES_URI (or DATA_POSTGRES_URL/DATA_DATABASE_URL), then legacy DB_* parts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per fetch batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create idx_dm_molecule_mfp2_vec HNSW index after backfill.",
    )
    parser.add_argument(
        "--reset-column",
        action="store_true",
        help="Drop and recreate dm_molecule.mfp2_vec if it exists with a different dimension.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="HNSW parameter m (default: 16).",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=128,
        help="HNSW parameter ef_construction (default: 128).",
    )
    return parser.parse_args()


def main() -> int:
    _load_repo_env()
    args = parse_args()
    if not args.dsn:
        print("Missing DSN. Pass --dsn or set DATA_POSTGRES_URI / DATA_POSTGRES_URL / DATA_DATABASE_URL / DB_*.", file=sys.stderr)
        return 1
    if args.batch_size <= 0:
        print("--batch-size must be > 0", file=sys.stderr)
        return 1
    if args.hnsw_m <= 0:
        print("--hnsw-m must be > 0", file=sys.stderr)
        return 1
    if args.hnsw_ef_construction <= 0:
        print("--hnsw-ef-construction must be > 0", file=sys.stderr)
        return 1

    try:
        backfill(
            args.dsn,
            args.batch_size,
            args.create_index,
            args.hnsw_m,
            args.hnsw_ef_construction,
            args.reset_column,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nBackfill failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
