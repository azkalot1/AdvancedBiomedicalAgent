#!/usr/bin/env python3
"""
DrugCentral ingestion for PostgreSQL (SDF, 2023+):
- Streams structures.molV3.sdf.gz (fallback: V2) from the 2023 release
- Extracts ID, PREFERRED_NAME, SYNONYMS, CAS_RN, URL
- Optionally computes SMILES, InChIKey, formula, exact mass via RDKit
- Creates PostgreSQL tables with full-text search capabilities
"""
from __future__ import annotations

import gzip
import re
import warnings
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests
from tqdm import tqdm

# Import our database config
from .config import DatabaseConfig, get_connection

# ---- RDKit (optional) --------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.inchi import MolToInchiKey  # type: ignore

    RDKit_OK = True
except Exception:
    warnings.warn("RDKit is not installed", stacklevel=2)
    RDKit_OK = False

# Import constants
from .constants import DRUGCENTRAL_BASE_URL

BASE_URL = DRUGCENTRAL_BASE_URL
CANDIDATE_FILES = ["structures.molV3.sdf.gz", "structures.molV2.sdf.gz"]


@dataclass
class SDFRecord:
    dc_id: str | None
    preferred_name: str | None
    synonyms: str | None  # semicolon-joined
    cas_rn: str | None
    url: str | None
    smiles: str | None
    inchi_key: str | None
    formula: str | None
    exact_mass: float | None


# ------------------------------ Database Schema -------------------------------


def _init_schema(con: psycopg2.extensions.connection) -> None:
    """Initialize DrugCentral schema in PostgreSQL."""
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS drugcentral_drugs (
              id SERIAL PRIMARY KEY,
              dc_id TEXT UNIQUE NOT NULL,
              name TEXT,
              name_norm TEXT,
              inchi_key TEXT,
              smiles TEXT,
              formula TEXT,
              exact_mass REAL,
              cas_rn TEXT,
              cas_rn_digits TEXT,
              url TEXT,
              synonyms TEXT,
              name_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(name, ''))) STORED,
              synonyms_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(synonyms, ''))) STORED,
              cas_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(cas_rn, ''))) STORED,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW()
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_drugcentral_dc_id ON drugcentral_drugs(dc_id);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_cas_digits ON drugcentral_drugs(cas_rn_digits);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_name_norm ON drugcentral_drugs(name_norm);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_inchi_key ON drugcentral_drugs(inchi_key);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_exact_mass ON drugcentral_drugs(exact_mass);

            -- Full-text search indexes
            CREATE INDEX IF NOT EXISTS idx_drugcentral_name_vector ON drugcentral_drugs USING GIN(name_vector);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_synonyms_vector ON drugcentral_drugs USING GIN(synonyms_vector);
            CREATE INDEX IF NOT EXISTS idx_drugcentral_cas_vector ON drugcentral_drugs USING GIN(cas_vector);

            -- Combined full-text search index
            CREATE INDEX IF NOT EXISTS idx_drugcentral_combined_text ON drugcentral_drugs
            USING GIN((name_vector || synonyms_vector || cas_vector));

            -- Trigger function for updated_at
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';

            -- Trigger for drugcentral_drugs
            DROP TRIGGER IF EXISTS update_drugcentral_drugs_updated_at ON drugcentral_drugs;
            CREATE TRIGGER update_drugcentral_drugs_updated_at
                BEFORE UPDATE ON drugcentral_drugs
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """
        )
        con.commit()


# ------------------------------ Download & Parsing ---------------------------


def _download(url: str, dest: Path) -> None:
    """Download file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(dest, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)


def _choose_and_fetch_sdf(raw_dir: Path) -> Path:
    """Download DrugCentral SDF file, trying multiple candidates."""
    out_dir = raw_dir / "drugcentral"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None

    for fname in CANDIDATE_FILES:
        url = BASE_URL + fname
        dest = out_dir / fname
        if dest.exists():
            print(f"🔄 Reusing existing {dest.name}")
            return dest
        try:
            print(f"📥 Downloading {fname} from DrugCentral...")
            _download(url, dest)
            return dest
        except Exception as e:
            last_err = e
            print(f"⚠️  Failed to download {fname}: {e}")

    raise RuntimeError(f"Could not fetch DrugCentral SDF from {BASE_URL}") from last_err


# ------------------------------ SDF parsing -----------------------------------

_PROP_HEAD_RE = re.compile(r">+\s*<([^>]+)>\s*$")


def _iter_sdf_records_stream(path: Path) -> Generator[tuple[str, dict[str, str]], None, None]:
    """Yield (molblock, props) from gzipped SDF in a streaming, low-mem way."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        mol_lines: list[str] = []
        props: dict[str, str] = {}
        cur_key: str | None = None
        cur_val_lines: list[str] = []
        in_props = False

        for line in fh:
            if line.startswith("$$$$"):
                if cur_key is not None:
                    props[cur_key] = "\n".join(cur_val_lines).strip()
                yield ("\n".join(mol_lines).rstrip(), props)
                mol_lines, props, cur_key, cur_val_lines, in_props = [], {}, None, [], False
                continue

            if not in_props:
                mol_lines.append(line.rstrip("\n"))
                if line.startswith("M  END"):
                    in_props = True
                continue

            m = _PROP_HEAD_RE.match(line)
            if m:
                if cur_key is not None:
                    props[cur_key] = "\n".join(cur_val_lines).strip()
                cur_key = m.group(1).strip()
                cur_val_lines = []
            else:
                if line.strip() == "":
                    if cur_key is not None:
                        props[cur_key] = "\n".join(cur_val_lines).strip()
                        cur_key = None
                        cur_val_lines = []
                else:
                    cur_val_lines.append(line.rstrip("\n"))


def _compute_rdkit_fields(molblock: str) -> tuple[str | None, str | None, str | None, float | None]:
    """Compute molecular properties using RDKit if available."""
    if not RDKit_OK:
        return None, None, None, None
    try:
        mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
        if mol is None:
            return None, None, None, None

        smiles = Chem.MolToSmiles(mol) or None

        try:
            inchikey = MolToInchiKey(mol) or None
        except Exception:
            inchikey = None

        try:
            formula = rdMolDescriptors.CalcMolFormula(mol) or None
        except Exception:
            formula = None

        try:
            exact_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
        except Exception:
            exact_mass = None

        return smiles, inchikey, formula, exact_mass
    except Exception:
        return None, None, None, None


def _norm_synonyms(text: str | None) -> str | None:
    """Normalize synonyms by deduplicating and joining with semicolons."""
    if not text:
        return None
    parts = re.split(r"[;\n\r]+", text)
    uniq, seen = [], set()
    for p in (x.strip() for x in parts):
        if p and p not in seen:
            uniq.append(p)
            seen.add(p)
    return ";".join(uniq) if uniq else None


def _digits_only(s: str | None) -> str | None:
    """Extract only digits from a string (useful for CAS number search)."""
    if not s:
        return None
    return "".join(ch for ch in s if ch.isdigit()) or None


def _norm_name(s: str | None) -> str | None:
    """Normalize name for searching."""
    if not s:
        return None
    return s.strip().lower()


def _record_from_sdf(molblock: str, props: dict[str, str]) -> SDFRecord:
    """Extract structured data from SDF molblock and properties."""
    dc_id = (props.get("ID") or "").strip() or None
    preferred_name = (props.get("PREFERRED_NAME") or "").strip() or None
    synonyms = _norm_synonyms(props.get("SYNONYMS"))
    cas_rn = (props.get("CAS_RN") or "").strip() or None
    url = (props.get("URL") or "").strip() or None

    smiles, inchikey, formula, exact_mass = _compute_rdkit_fields(molblock)

    return SDFRecord(dc_id, preferred_name, synonyms, cas_rn, url, smiles, inchikey, formula, exact_mass)


# ------------------------------ PostgreSQL Ingestion -------------------------


def _bulk_upsert(config: DatabaseConfig, recs: Iterable[SDFRecord]) -> int:
    """Insert/merge all records with progress tracking."""
    with get_connection(config) as con:
        with con.cursor() as cur:
            n = 0
            batch = []
            batch_size = 1000

            for r in recs:
                if not r.dc_id:
                    continue

                # First, try to insert or get existing synonyms
                batch.append(
                    (
                        r.dc_id,
                        r.preferred_name,
                        _norm_name(r.preferred_name),
                        r.inchi_key,
                        r.smiles,
                        r.formula,
                        r.exact_mass,
                        r.cas_rn,
                        _digits_only(r.cas_rn),
                        r.url,
                        r.synonyms,
                    )
                )
                n += 1

                if len(batch) >= batch_size:
                    # Execute batch upsert
                    cur.executemany(
                        """
                        INSERT INTO drugcentral_drugs
                        (dc_id, name, name_norm, inchi_key, smiles, formula, exact_mass,
                         cas_rn, cas_rn_digits, url, synonyms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (dc_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            name_norm = EXCLUDED.name_norm,
                            inchi_key = EXCLUDED.inchi_key,
                            smiles = EXCLUDED.smiles,
                            formula = EXCLUDED.formula,
                            exact_mass = EXCLUDED.exact_mass,
                            cas_rn = EXCLUDED.cas_rn,
                            cas_rn_digits = EXCLUDED.cas_rn_digits,
                            url = EXCLUDED.url,
                            synonyms = CASE
                                WHEN EXCLUDED.synonyms IS NOT NULL THEN
                                    CASE
                                        WHEN drugcentral_drugs.synonyms IS NULL OR drugcentral_drugs.synonyms = ''
                                        THEN EXCLUDED.synonyms
                                        ELSE drugcentral_drugs.synonyms || ';' || EXCLUDED.synonyms
                                    END
                                ELSE drugcentral_drugs.synonyms
                            END,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                        batch,
                    )
                    con.commit()
                    batch = []
                    print(f"💾 Committed {n:,} records...")

            # Insert remaining records
            if batch:
                cur.executemany(
                    """
                    INSERT INTO drugcentral_drugs
                    (dc_id, name, name_norm, inchi_key, smiles, formula, exact_mass,
                     cas_rn, cas_rn_digits, url, synonyms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dc_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        name_norm = EXCLUDED.name_norm,
                        inchi_key = EXCLUDED.inchi_key,
                        smiles = EXCLUDED.smiles,
                        formula = EXCLUDED.formula,
                        exact_mass = EXCLUDED.exact_mass,
                        cas_rn = EXCLUDED.cas_rn,
                        cas_rn_digits = EXCLUDED.cas_rn_digits,
                        url = EXCLUDED.url,
                        synonyms = CASE
                            WHEN EXCLUDED.synonyms IS NOT NULL THEN
                                CASE
                                    WHEN drugcentral_drugs.synonyms IS NULL OR drugcentral_drugs.synonyms = ''
                                    THEN EXCLUDED.synonyms
                                    ELSE drugcentral_drugs.synonyms || ';' || EXCLUDED.synonyms
                                END
                            ELSE drugcentral_drugs.synonyms
                        END,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    batch,
                )
                con.commit()

    return n


# ------------------------------ Search Functions ------------------------------

# Search functions have been moved to src/curebench/data/app/search_drugcentral.py

# ------------------------------ Main Entry Point -----------------------------


def build_drugcentral(config: DatabaseConfig, raw_dir: Path) -> None:
    """Main ingestion function for DrugCentral into PostgreSQL."""
    print("🚀 Starting DrugCentral PostgreSQL ingestion...")

    try:
        # Download SDF file
        sdf_path = _choose_and_fetch_sdf(raw_dir)

        # Initialize schema
        with get_connection(config) as con:
            _init_schema(con)

        print(f"📋 Processing SDF file: {sdf_path.name}")
        print(f"🧪 RDKit: {'Available' if RDKit_OK else 'Not available (limited molecular properties)'}")

        # Create generator for records
        def gen() -> Generator[SDFRecord, None, None]:
            for molblock, props in tqdm(_iter_sdf_records_stream(sdf_path), desc="Processing DrugCentral SDF", unit="molecule"):
                yield _record_from_sdf(molblock, props)

        # Bulk insert with progress tracking
        count = _bulk_upsert(config, gen())

        print(f"✅ [DrugCentral] Successfully loaded {count:,} molecules from {sdf_path.name}")
        print(f"🧪 RDKit molecular properties: {'computed' if RDKit_OK else 'skipped (install rdkit-pypi for SMILES/InChI)'}")
        print("🔍 Data available in drugcentral_drugs table")

    except Exception as e:
        print(f"❌ [DrugCentral] Error during ingestion: {e}")
        raise
