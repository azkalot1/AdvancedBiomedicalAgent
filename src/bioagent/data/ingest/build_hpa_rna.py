#!/usr/bin/env python3
"""
Human Protein Atlas (HPA) single-cell RNA expression by cell type.

Downloads rna_single_cell_type.tsv.zip, loads into hpa_rna_cell_type_expression,
builds hpa_cell_type_summary, and optionally backfills dm_target.ensembl_gene_id.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import psycopg2.extras
import requests
from tqdm import tqdm

try:
    from .config import DatabaseConfig, get_connection
    from .constants import DEFAULT_HEADERS, HPA_RNA_CELL_TYPE_URL
except ImportError:
    from config import DatabaseConfig, get_connection
    from constants import DEFAULT_HEADERS, HPA_RNA_CELL_TYPE_URL

HPA_ZIP_NAME = "rna_single_cell_type.tsv.zip"
HPA_TSV_NAME = "rna_single_cell_type.tsv"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"🔄 Reusing existing {dest.name}")
        return dest
    with requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            if total_size > 0:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest.name}",
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return dest


def _find_tsv_member(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1].lower()
        if base.endswith(".tsv") and "rna" in base and "cell" in base:
            return name
    for name in zf.namelist():
        if name.lower().endswith(".tsv"):
            return name
    raise FileNotFoundError("No .tsv file found in HPA zip archive")


def _normalize_header(h: str) -> str:
    return (h or "").strip().lower().replace(" ", "_")


def _init_schema(con) -> None:
    with con.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hpa_rna_cell_type_expression (
                id SERIAL PRIMARY KEY,
                ensembl_gene_id TEXT NOT NULL,
                gene_symbol TEXT NOT NULL,
                cell_type TEXT NOT NULL,
                ncpm REAL NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ensembl_gene_id, cell_type)
            );

            CREATE INDEX IF NOT EXISTS idx_hpa_rna_gene_symbol
                ON hpa_rna_cell_type_expression(gene_symbol);
            CREATE INDEX IF NOT EXISTS idx_hpa_rna_ensembl
                ON hpa_rna_cell_type_expression(ensembl_gene_id);
            CREATE INDEX IF NOT EXISTS idx_hpa_rna_cell_type
                ON hpa_rna_cell_type_expression(cell_type);
            CREATE INDEX IF NOT EXISTS idx_hpa_rna_cell_type_trgm
                ON hpa_rna_cell_type_expression USING GIN (cell_type gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_hpa_rna_ncpm
                ON hpa_rna_cell_type_expression(ncpm DESC);

            CREATE TABLE IF NOT EXISTS hpa_cell_type_summary (
                id SERIAL PRIMARY KEY,
                cell_type TEXT NOT NULL UNIQUE,
                gene_count INTEGER NOT NULL,
                avg_ncpm REAL,
                max_ncpm REAL,
                top_gene_symbol TEXT
            );
            """
        )
    con.commit()


def _truncate_tables(con) -> None:
    with con.cursor() as cur:
        cur.execute("TRUNCATE hpa_cell_type_summary RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE hpa_rna_cell_type_expression RESTART IDENTITY CASCADE;")
    con.commit()


def _populate_summary(con) -> None:
    with con.cursor() as cur:
        cur.execute("TRUNCATE hpa_cell_type_summary RESTART IDENTITY CASCADE;")
        cur.execute(
            """
            INSERT INTO hpa_cell_type_summary (
                cell_type, gene_count, avg_ncpm, max_ncpm, top_gene_symbol
            )
            SELECT
                agg.cell_type,
                agg.gene_count,
                agg.avg_ncpm,
                agg.max_ncpm,
                top.gene_symbol
            FROM (
                SELECT
                    cell_type,
                    COUNT(DISTINCT gene_symbol)::INTEGER AS gene_count,
                    AVG(ncpm)::REAL AS avg_ncpm,
                    MAX(ncpm)::REAL AS max_ncpm
                FROM hpa_rna_cell_type_expression
                GROUP BY cell_type
            ) agg
            LEFT JOIN LATERAL (
                SELECT gene_symbol
                FROM hpa_rna_cell_type_expression e
                WHERE e.cell_type = agg.cell_type
                ORDER BY e.ncpm DESC NULLS LAST, e.gene_symbol
                LIMIT 1
            ) top ON TRUE
            ORDER BY agg.cell_type;
            """
        )
    con.commit()


def _backfill_ensembl(con) -> int:
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'dm_target'
            )
            """
        )
        if not cur.fetchone()[0]:
            return 0
        cur.execute(
            """
            UPDATE dm_target t
            SET ensembl_gene_id = h.ensembl_gene_id
            FROM (
                SELECT DISTINCT ON (gene_symbol)
                    gene_symbol,
                    ensembl_gene_id
                FROM hpa_rna_cell_type_expression
                ORDER BY gene_symbol, ensembl_gene_id
            ) h
            WHERE t.gene_symbol = h.gene_symbol
              AND (t.ensembl_gene_id IS NULL OR t.ensembl_gene_id = '');
            """
        )
        return cur.rowcount


def ingest_hpa_rna(
    config: DatabaseConfig,
    raw_dir: Path,
    batch_size: int = 10000,
    backfill_ensembl: bool = True,
) -> None:
    """
    Download HPA RNA single-cell-type TSV, load PostgreSQL tables, optional Ensembl backfill.

    Args:
        config: Database configuration.
        raw_dir: Directory for downloaded zip (e.g. repo raw/).
        batch_size: Rows per INSERT batch.
        backfill_ensembl: If True, set dm_target.ensembl_gene_id where missing and gene_symbol matches.
    """
    dest = raw_dir / HPA_ZIP_NAME
    print("📥 Downloading HPA RNA single-cell-type data...")
    _download(HPA_RNA_CELL_TYPE_URL, dest)

    with get_connection(config) as con:
        print("📋 Initializing HPA schema...")
        _init_schema(con)
        print("🗑️  Clearing existing HPA rows...")
        _truncate_tables(con)

        batch: list[tuple[str, str, str, float]] = []
        total = 0
        skipped = 0

        insert_sql = """
            INSERT INTO hpa_rna_cell_type_expression (ensembl_gene_id, gene_symbol, cell_type, ncpm)
            VALUES %s
            ON CONFLICT (ensembl_gene_id, cell_type)
            DO UPDATE SET
                gene_symbol = EXCLUDED.gene_symbol,
                ncpm = EXCLUDED.ncpm
        """

        with zipfile.ZipFile(dest, "r") as zf:
            member = _find_tsv_member(zf)
            print(f"📄 Reading {member}...")
            with zf.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                reader = csv.DictReader(text, delimiter="\t")

                if not reader.fieldnames:
                    raise ValueError("HPA TSV has no header row")

                col_map: dict[str, str] = {}
                for fn in reader.fieldnames:
                    key = _normalize_header(fn)
                    col_map[key] = fn

                def pick(*candidates: str) -> str | None:
                    for c in candidates:
                        if c in col_map:
                            return col_map[c]
                    return None

                col_gene = pick("gene", "ensembl_gene_id")
                col_name = pick("gene_name", "genename")
                col_cell = pick("cell_type", "celltype")
                col_ncpm = pick("ncpm", "n_cpm")

                if not all([col_gene, col_name, col_cell, col_ncpm]):
                    raise ValueError(
                        f"Unexpected HPA columns: {reader.fieldnames}. "
                        "Expected Gene, Gene name, Cell type, nCPM."
                    )

                with con.cursor() as cur:
                    for row in tqdm(reader, desc="HPA rows", unit="row"):
                        gid = (row.get(col_gene) or "").strip()
                        sym = (row.get(col_name) or "").strip()
                        ctype = (row.get(col_cell) or "").strip()
                        raw_ncpm = (row.get(col_ncpm) or "").strip()
                        if not gid or not sym or not ctype:
                            skipped += 1
                            continue
                        try:
                            ncpm = float(raw_ncpm)
                        except ValueError:
                            skipped += 1
                            continue
                        batch.append((gid, sym, ctype, ncpm))
                        if len(batch) >= batch_size:
                            psycopg2.extras.execute_values(
                                cur,
                                insert_sql,
                                batch,
                                page_size=batch_size,
                            )
                            con.commit()
                            total += len(batch)
                            batch.clear()

                    if batch:
                        psycopg2.extras.execute_values(
                            cur,
                            insert_sql,
                            batch,
                            page_size=len(batch),
                        )
                        con.commit()
                        total += len(batch)

        print(f"✅ Loaded {total:,} HPA expression rows ({skipped:,} skipped).")
        print("📊 Building cell-type summary...")
        _populate_summary(con)

        if backfill_ensembl:
            n = _backfill_ensembl(con)
            con.commit()
            print(f"✅ Backfilled ensembl_gene_id on {n:,} dm_target row(s).")
        else:
            print("⏭️  Skipping dm_target ensembl_gene_id backfill.")

        with con.cursor() as cur:
            cur.execute("ANALYZE hpa_rna_cell_type_expression;")
            cur.execute("ANALYZE hpa_cell_type_summary;")
        con.commit()
