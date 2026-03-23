#!/usr/bin/env python3
"""
STRING DB human protein-protein interactions (9606).

Downloads protein.info (ENSP -> gene symbol) and protein.links.detailed (scored edges),
loads string_protein_info and string_ppi tables.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import psycopg2.extras
import requests
from tqdm import tqdm

try:
    from .config import DatabaseConfig, get_connection
    from .constants import DEFAULT_HEADERS, STRING_PPI_LINKS_URL, STRING_PROTEIN_INFO_URL
except ImportError:
    from config import DatabaseConfig, get_connection
    from constants import DEFAULT_HEADERS, STRING_PPI_LINKS_URL, STRING_PROTEIN_INFO_URL

INFO_GZ_NAME = "9606.protein.info.v12.0.txt.gz"
LINKS_GZ_NAME = "9606.protein.links.detailed.v12.0.txt.gz"


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


def _normalize_string_id(raw: str) -> str | None:
    """Strip taxon prefix (9606.) from STRING protein id; return ENSP... or None."""
    s = (raw or "").strip()
    if not s:
        return None
    if "." in s:
        _, rest = s.split(".", 1)
        return rest.strip() or None
    return s


def _init_schema(con) -> None:
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS string_protein_info (
                id SERIAL PRIMARY KEY,
                ensp_id TEXT NOT NULL UNIQUE,
                gene_symbol TEXT NOT NULL,
                protein_size INTEGER,
                annotation TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_string_pinfo_gene
                ON string_protein_info(gene_symbol);

            CREATE TABLE IF NOT EXISTS string_ppi (
                id SERIAL PRIMARY KEY,
                gene_symbol_1 TEXT NOT NULL,
                gene_symbol_2 TEXT NOT NULL,
                ensp_id_1 TEXT NOT NULL,
                ensp_id_2 TEXT NOT NULL,
                neighborhood SMALLINT NOT NULL DEFAULT 0,
                fusion SMALLINT NOT NULL DEFAULT 0,
                cooccurence SMALLINT NOT NULL DEFAULT 0,
                coexpression SMALLINT NOT NULL DEFAULT 0,
                experimental SMALLINT NOT NULL DEFAULT 0,
                database SMALLINT NOT NULL DEFAULT 0,
                textmining SMALLINT NOT NULL DEFAULT 0,
                combined_score SMALLINT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ensp_id_1, ensp_id_2)
            );
            CREATE INDEX IF NOT EXISTS idx_string_ppi_gene1 ON string_ppi(gene_symbol_1);
            CREATE INDEX IF NOT EXISTS idx_string_ppi_gene2 ON string_ppi(gene_symbol_2);
            CREATE INDEX IF NOT EXISTS idx_string_ppi_combined ON string_ppi(combined_score DESC);
            CREATE INDEX IF NOT EXISTS idx_string_ppi_g1_score
                ON string_ppi(gene_symbol_1, combined_score DESC);
            CREATE INDEX IF NOT EXISTS idx_string_ppi_g2_score
                ON string_ppi(gene_symbol_2, combined_score DESC);
            """
        )
    con.commit()


def _truncate_tables(con) -> None:
    with con.cursor() as cur:
        cur.execute("TRUNCATE string_ppi RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE string_protein_info RESTART IDENTITY CASCADE;")
    con.commit()


def _load_protein_info(con, info_path: Path) -> dict[str, str]:
    """Parse protein.info.gz into ensp_id -> gene_symbol. Returns mapping dict."""
    mapping: dict[str, str] = {}
    batch: list[tuple[str, str, int | None, str | None]] = []

    insert_sql = """
        INSERT INTO string_protein_info (ensp_id, gene_symbol, protein_size, annotation)
        VALUES %s
        ON CONFLICT (ensp_id) DO UPDATE SET
            gene_symbol = EXCLUDED.gene_symbol,
            protein_size = EXCLUDED.protein_size,
            annotation = EXCLUDED.annotation
    """

    with gzip.open(info_path, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first:
            raise ValueError("Empty protein.info file")

        with con.cursor() as cur:
            for line in tqdm(f, desc="string_protein_info", unit="row"):
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 2:
                    continue
                raw_id = cols[0].lstrip("#").strip()
                ensp = _normalize_string_id(raw_id)
                if not ensp:
                    continue
                preferred = (cols[1] or "").strip()
                if not preferred or preferred.lower() in ("preferred_name", "protein_size"):
                    continue
                if "protein" in ensp.lower() and "id" in ensp.lower():
                    continue
                mapping[ensp] = preferred
                psize: int | None = None
                if len(cols) > 2 and cols[2].strip().isdigit():
                    psize = int(cols[2])
                annot = cols[3].strip() if len(cols) > 3 else None
                batch.append((ensp, preferred, psize, annot))
                if len(batch) >= 5000:
                    psycopg2.extras.execute_values(cur, insert_sql, batch, page_size=5000)
                    con.commit()
                    batch.clear()
            if batch:
                psycopg2.extras.execute_values(cur, insert_sql, batch, page_size=len(batch))
                con.commit()

    print(f"✅ Loaded {len(mapping):,} STRING protein -> gene_symbol mappings.")
    return mapping


def ingest_string_ppi(
    config: DatabaseConfig,
    raw_dir: Path,
    min_combined_score: int = 400,
    batch_size: int = 10000,
) -> None:
    """
    Download STRING human PPI files and load PostgreSQL.

    Args:
        config: Database configuration.
        raw_dir: Directory for downloaded .gz files.
        min_combined_score: Keep edges with combined_score >= this (0-1000 scale in file).
        batch_size: Rows per INSERT batch for string_ppi.
    """
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    info_path = raw_dir / INFO_GZ_NAME
    links_path = raw_dir / LINKS_GZ_NAME

    print("📥 Downloading STRING protein.info...")
    _download(STRING_PROTEIN_INFO_URL, info_path)
    print("📥 Downloading STRING protein.links.detailed...")
    _download(STRING_PPI_LINKS_URL, links_path)

    with get_connection(config) as con:
        print("📋 Initializing STRING schema...")
        _init_schema(con)
        print("🗑️  Clearing existing STRING PPI rows...")
        _truncate_tables(con)

        ensp_to_gene = _load_protein_info(con, info_path)

        insert_ppi = """
            INSERT INTO string_ppi (
                gene_symbol_1, gene_symbol_2, ensp_id_1, ensp_id_2,
                neighborhood, fusion, cooccurence, coexpression,
                experimental, database, textmining, combined_score
            )
            VALUES %s
            ON CONFLICT (ensp_id_1, ensp_id_2) DO UPDATE SET
                gene_symbol_1 = EXCLUDED.gene_symbol_1,
                gene_symbol_2 = EXCLUDED.gene_symbol_2,
                neighborhood = EXCLUDED.neighborhood,
                fusion = EXCLUDED.fusion,
                cooccurence = EXCLUDED.cooccurence,
                coexpression = EXCLUDED.coexpression,
                experimental = EXCLUDED.experimental,
                database = EXCLUDED.database,
                textmining = EXCLUDED.textmining,
                combined_score = EXCLUDED.combined_score
        """

        batch: list[tuple] = []
        total_kept = 0
        total_skipped_score = 0
        total_skipped_map = 0

        with gzip.open(links_path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                raise ValueError("Empty links file")
            print(f"   Links header: {header.strip()[:120]}...")

            with con.cursor() as cur:
                for line in tqdm(f, desc="string_ppi edges", unit="row"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # STRING v12 detailed: protein1 protein2 + 7 evidence columns + combined_score (= 10 cols)
                    if len(parts) < 10:
                        continue
                    p1_raw, p2_raw = parts[0], parts[1]
                    try:
                        # STRING v12 detailed: neighborhood, fusion, cooccurence, coexpression,
                        # experimental, database, textmining, combined_score
                        nh = int(parts[2])
                        fu = int(parts[3])
                        cooc = int(parts[4])
                        coex = int(parts[5])
                        exp = int(parts[6])
                        db = int(parts[7])
                        tm = int(parts[8])
                        combined = int(parts[9])
                    except ValueError:
                        continue

                    if combined < min_combined_score:
                        total_skipped_score += 1
                        continue

                    e1 = _normalize_string_id(p1_raw)
                    e2 = _normalize_string_id(p2_raw)
                    if not e1 or not e2:
                        total_skipped_map += 1
                        continue
                    g1 = ensp_to_gene.get(e1)
                    g2 = ensp_to_gene.get(e2)
                    if not g1 or not g2:
                        total_skipped_map += 1
                        continue

                    # DB column cooccurence <- file cooccurrence; coexpression matches file
                    batch.append(
                        (g1, g2, e1, e2, nh, fu, cooc, coex, exp, db, tm, combined)
                    )
                    if len(batch) >= batch_size:
                        psycopg2.extras.execute_values(cur, insert_ppi, batch, page_size=batch_size)
                        con.commit()
                        total_kept += len(batch)
                        batch.clear()

                if batch:
                    psycopg2.extras.execute_values(cur, insert_ppi, batch, page_size=len(batch))
                    con.commit()
                    total_kept += len(batch)

        print(
            f"✅ Inserted {total_kept:,} STRING PPI rows "
            f"(min_combined_score={min_combined_score})."
        )
        print(f"   Skipped (below score): {total_skipped_score:,}")
        print(f"   Skipped (unmapped ENSP): {total_skipped_map:,}")

        with con.cursor() as cur:
            cur.execute("ANALYZE string_protein_info;")
            cur.execute("ANALYZE string_ppi;")
        con.commit()
