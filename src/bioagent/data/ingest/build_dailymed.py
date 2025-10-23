#!/usr/bin/env python3
"""
DailyMed ingestion pipeline for PostgreSQL with Semantic Search (v3 - Corrected):

Phase 1: Ingestion
- Downloads and processes all SPL XML files.
- Populates `dailymed_products` and `dailymed_sections`.
- **Enforces that only sections with a valid, non-empty canonical title are ingested.**

Phase 2: Embedding & Linking
- Gathers all unique canonical titles from the ingested data.
- Generates vector embeddings for each unique title.
- Populates the `dailymed_section_names` table.
- Backfills the `section_name_id` foreign key in the `dailymed_sections` table.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
    from .constants import DAILYMED_BASE_URL, DAILYMED_PAGE, DAILYMED_SITE_BASE
except ImportError:
    from config import DatabaseConfig, get_connection
    from constants import DAILYMED_BASE_URL, DAILYMED_PAGE, DAILYMED_SITE_BASE


# ----------------------------- fetch utils (Unchanged) ------------------------------------
def _full_release_human_rx_urls() -> list[str]:
    base = DAILYMED_BASE_URL
    return [
        base + "dm_spl_release_human_rx_part1.zip",
        base + "dm_spl_release_human_rx_part2.zip",
        base + "dm_spl_release_human_rx_part3.zip",
        base + "dm_spl_release_human_rx_part4.zip",
    ]


def _latest_weekly_zip_url() -> str | None:
    try:
        r = requests.get(DAILYMED_PAGE, timeout=60)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]
        weekly = [u for u in links if u and "dm_spl_weekly_update_" in u and u.endswith(".zip")]
        if not weekly:
            return None

        def key(u: str) -> str:
            m = re.search(r"(\d{6,8})_(\d{6,8})", u)
            return m.group(0) if m else ""

        weekly.sort(key=key, reverse=True)
        url = weekly[0]
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("/"):
            url = DAILYMED_SITE_BASE + url
        return url
    except Exception:
        return None


def _download(url: str, dest: Path) -> None:
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


# ----------------------------- schema -----------------------------------------


def _init_schema(con: psycopg2.extensions.connection) -> None:
    """Initializes the complete schema required for the ingestion and semantic search."""
    with con.cursor() as cur:
        print("Initializing database schema...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dailymed_products (
                id SERIAL PRIMARY KEY, set_id TEXT UNIQUE NOT NULL, product_name TEXT, brand_names TEXT,
                generic_names TEXT, substances TEXT,
                name_vector tsvector GENERATED ALWAYS AS (to_tsvector('english',
                    COALESCE(product_name, '') || ' ' || COALESCE(brand_names, '') || ' ' ||
                    COALESCE(generic_names, '') || ' ' || COALESCE(substances, ''))) STORED,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS dailymed_sections(
                id SERIAL PRIMARY KEY, set_id TEXT NOT NULL, code TEXT, title_original TEXT,
                title_canonical TEXT, text TEXT, section_name_id INTEGER,
                text_vector tsvector GENERATED ALWAYS AS (to_tsvector('english',
                    COALESCE(title_canonical, '') || ' ' || COALESCE(title_original, '') || ' ' ||
                    COALESCE(text, ''))) STORED,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS dailymed_section_names (
                id SERIAL PRIMARY KEY, section_name TEXT UNIQUE NOT NULL,
                embedding VECTOR(384) NOT NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_dailymed_products_set_id ON dailymed_products(set_id);
            CREATE INDEX IF NOT EXISTS idx_dailymed_sections_set_id ON dailymed_sections(set_id);
            CREATE INDEX IF NOT EXISTS idx_dailymed_sections_title_canonical ON dailymed_sections(title_canonical);
            CREATE INDEX IF NOT EXISTS idx_dailymed_products_name_vector ON dailymed_products USING GIN(name_vector);
            CREATE INDEX IF NOT EXISTS idx_dailymed_sections_text_vector ON dailymed_sections USING GIN(text_vector);
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ language 'plpgsql';
            DROP TRIGGER IF EXISTS update_dailymed_products_updated_at ON dailymed_products;
            CREATE TRIGGER update_dailymed_products_updated_at
                BEFORE UPDATE ON dailymed_products FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        )
        con.commit()
        print("Schema initialization complete.")


# ----------------------------- XML iteration (Unchanged) ----------------------------------
def _iter_xmls_in_zipfile(zf: zipfile.ZipFile) -> Iterator[bytes]:
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        low = name.lower()
        if low.endswith(".xml"):
            with zf.open(info) as fh:
                yield fh.read()
        elif low.endswith(".zip"):
            with zf.open(info) as fh:
                data = fh.read()
            try:
                with zipfile.ZipFile(io.BytesIO(data), "r") as inner:
                    yield from _iter_xmls_in_zipfile(inner)
            except zipfile.BadZipFile:
                continue


def _iter_xml_from_weekly_zip(path: Path) -> Iterator[bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        yield from _iter_xmls_in_zipfile(zf)


# ----------------------------- SPL parsing ------------------------------------
class SPLData(BaseModel):
    set_id: str
    product_name: str | None = None
    brand_names: list[str] = Field(default_factory=list)
    generic_names: list[str] = Field(default_factory=list)
    active_substances: list[str] = Field(default_factory=list)
    sections: list[tuple[str, str, str, str]] = Field(default_factory=list)


CANONICAL_TITLE_MAP: dict[str, list[str]] = {
    "indications and usage": ["indications and usage", "use", "uses"],
    "contraindications": ["contraindications", "contraindication"],
    "adverse reactions": ["adverse reactions", "adverse reaction", "adverse events"],
    "how supplied": ["how supplied", "storage and handling"],
    "active ingredient": ["active ingredient", "active ingredients"],
    "inactive ingredient": ["inactive ingredient", "inactive ingredients"],
    "purpose": ["purpose", "purposes"],
    "precautions": ["precautions", "precaution", "general"],
}
TITLE_LOOKUP_MAP: dict[str, str] = {v: k for k, vs in CANONICAL_TITLE_MAP.items() for v in vs}


def _normalize_title(title: str) -> str:
    if title.startswith('warning'):
        return 'warnings'
    if title.startswith('dosage'):
        return 'dosage and administration'
    if title.startswith('use in'):
        return 'use in specific populations'
    if title.startswith('use with'):
        return 'drug interactions'
    if title.startswith('treatment of'):
        return 'indications and usage'
    if title.startswith('stop use'):
        return 'stop use'
    if title.startswith('ask a doctor'):
        return 'ask a doctor'
    if 'risk' in title:
        return 'risks'
    if 'effect' in title:
        return 'drug effects'
    if 'special population' in title or 'special risk' in title or 'special precaution' in title:
        return 'use in specific populations'
    if 'principal display panel' in title:
        return 'principal display panel'
    return TITLE_LOOKUP_MAP.get(title, title)


def _clean(txt: str | None) -> str:
    if not txt:
        return ""
    return " ".join(txt.split()).strip()


def _process_and_clean_sections(sections: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    processed = []
    for code, title, text in sections:
        cleaned_text = text.strip()
        if len(cleaned_text.split()) < 5:
            continue
        cleaned_title = title.strip()
        if not cleaned_title:
            text_lower = cleaned_text.lower()
            if text_lower.startswith('do not use'):
                cleaned_title = 'Do Not Use'
            elif text_lower.startswith('ask a doctor'):
                cleaned_title = 'Ask a Doctor'
            elif text_lower.startswith('stop use'):
                cleaned_title = 'Stop Use'
            else:
                first_colon = cleaned_text.find(':')
                if 0 < first_colon < 100:
                    potential = cleaned_text[:first_colon]
                    if len(potential.split()) <= 15 and '.' not in potential:
                        cleaned_title = potential
                        cleaned_text = cleaned_text[first_colon + 1 :].strip()

        original_sanitized, canonical = "", ""
        if cleaned_title:
            sanitized = re.sub(r'[^a-z\s]', '', cleaned_title.lower())
            original_sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            canonical = _normalize_title(original_sanitized)

        # *** THE FIX IS HERE ***
        # Only keep sections that have a valid, non-empty canonical title.
        if canonical:
            processed.append((code, original_sanitized, canonical, cleaned_text))

    return processed


def _extract_spl_data(xml_bytes: bytes) -> SPLData | None:
    # ... (This function is unchanged, but now calls the fixed _process_and_clean_sections) ...
    soup = BeautifulSoup(xml_bytes, "xml")
    set_id_tag = soup.find("setId")
    if not set_id_tag or not set_id_tag.has_attr("root"):
        return None
    set_id = set_id_tag["root"].strip()
    brand_names, generic_names, substances = set(), set(), set()
    data_section = soup.find("code", code="48780-1")
    area = data_section.parent if data_section else soup
    for p in area.find_all("manufacturedProduct"):
        if n := p.find("name"):
            brand_names.add(_clean(n.text))
        if g := p.find("genericMedicine"):
            if n := g.find("name"):
                generic_names.add(_clean(n.text))
    for i in area.find_all("ingredient", classCode="ACTIB"):
        if s := i.find("name"):
            substances.add(_clean(s.text))
    prod_name = next(iter(brand_names), next(iter(generic_names), None))
    raw_sections = []
    for sec in soup.find_all("section"):
        code = c["code"].strip() if (c := sec.find("code")) and c.has_attr("code") else ""
        title = _clean(t.text if (t := sec.find("title")) else "")
        body = _clean(b.get_text(separator=" ") if (b := sec.find("text")) else "")
        raw_sections.append((code, title, body))
    cleaned_sections = _process_and_clean_sections(raw_sections)
    return SPLData(
        set_id=set_id,
        product_name=prod_name,
        brand_names=list(brand_names),
        generic_names=list(generic_names),
        active_substances=list(substances),
        sections=cleaned_sections,
    )


# ----------------------------- Ingestion Phases ------------------------------


def _ingest_xml_data(con: psycopg2.extensions.connection, dests: list[Path], n_max: int | None) -> None:
    print("\n--- Phase 1: Ingesting XML data ---")
    total_xml, products_processed = 0, set()
    with con.cursor() as cur:
        for top_zip in dests:
            print(f"Processing {top_zip.name}...")
            xml_files = list(_iter_xml_from_weekly_zip(top_zip))
            with tqdm(xml_files, desc=f"Ingesting {top_zip.name}", unit="xml") as pbar:
                for xml_bytes in pbar:
                    total_xml += 1
                    spl_data = _extract_spl_data(xml_bytes)
                    if not spl_data:
                        continue
                    cur.execute(
                        """
                        INSERT INTO dailymed_products (set_id, product_name, brand_names, generic_names, substances)
                        VALUES (%s, %s, %s, %s, %s) ON CONFLICT(set_id) DO UPDATE SET
                        product_name = EXCLUDED.product_name, brand_names = EXCLUDED.brand_names,
                        generic_names = EXCLUDED.generic_names, substances = EXCLUDED.substances,
                        updated_at = NOW();
                        """,
                        (
                            spl_data.set_id,
                            spl_data.product_name,
                            "; ".join(spl_data.brand_names),
                            "; ".join(spl_data.generic_names),
                            "; ".join(spl_data.active_substances),
                        ),
                    )
                    cur.execute("DELETE FROM dailymed_sections WHERE set_id = %s", (spl_data.set_id,))
                    if spl_data.sections:
                        sections_to_insert = [(spl_data.set_id, c, o, can, t) for c, o, can, t in spl_data.sections]
                        if sections_to_insert:
                            psycopg2.extras.execute_values(
                                cur,
                                "INSERT INTO dailymed_sections(set_id, code, title_original, title_canonical, text) VALUES %s",
                                sections_to_insert,
                                page_size=200,
                            )
                    products_processed.add(spl_data.set_id)
                    if len(products_processed) % 500 == 0:
                        con.commit()
                    if n_max and total_xml >= n_max:
                        break
            con.commit()
    print(f"✅ Phase 1 complete. Processed {total_xml:,} XML files for {len(products_processed):,} unique products.")


def _update_section_names_and_embeddings(con: psycopg2.extensions.connection) -> None:
    print("\n--- Phase 2: Generating Embeddings for Section Names ---")
    with con.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT title_canonical FROM dailymed_sections WHERE title_canonical IS NOT NULL AND title_canonical != '';"
        )
        unique_titles = [row[0] for row in cur.fetchall()]
        print(f"Found {len(unique_titles)} unique canonical titles to process.")
        if not unique_titles:
            return
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Generating embeddings...")
        embeddings = model.encode(unique_titles, show_progress_bar=True)
        print("Populating dailymed_section_names table...")
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO dailymed_section_names (section_name, embedding) VALUES %s ON CONFLICT (section_name) DO NOTHING;",
            list(zip(unique_titles, [e.tolist() for e in embeddings], strict=False)),
            page_size=200,
        )
        con.commit()
        print("Fetching name-to-ID map...")
        cur.execute("SELECT section_name, id FROM dailymed_section_names;")
        name_to_id_map = {row[0]: row[1] for row in cur.fetchall()}
        print("Backfilling section_name_id in dailymed_sections table...")
        update_data = [(id, name) for name, id in name_to_id_map.items()]
        psycopg2.extras.execute_values(
            cur,
            "UPDATE dailymed_sections SET section_name_id = data.id FROM (VALUES %s) AS data(id, name) WHERE title_canonical = data.name;",
            update_data,
            page_size=200,
        )
        con.commit()
    print("✅ Phase 2 complete. Semantic search is now enabled.")


def _apply_final_constraints(con: psycopg2.extensions.connection) -> None:
    """Phase 3: Apply foreign key constraints now that data is consistent."""
    print("\n--- Phase 3: Applying Final Database Constraints ---")
    with con.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE dailymed_sections
            DROP CONSTRAINT IF EXISTS fk_dailymed_section_name;

            ALTER TABLE dailymed_sections
            ADD CONSTRAINT fk_dailymed_section_name
            FOREIGN KEY (section_name_id) REFERENCES dailymed_section_names(id);
        """
        )
        con.commit()
    print("✅ Foreign key constraints applied successfully.")


# ----------------------------- public entrypoint ------------------------------


def run_dailymed_ingestion_pipeline(config: DatabaseConfig, raw_dir: Path, mode: str, n_max: int | None = None) -> None:
    """Runs the full ingestion and embedding pipeline for DailyMed data."""
    print(f"🚀 Starting DailyMed ingestion pipeline (mode: {mode})...")
    if mode == "full":
        urls, dests = _full_release_human_rx_urls(), []
        for url in urls:
            dest = raw_dir / "dailymed" / "full" / Path(url).name
            if not dest.exists():
                _download(url, dest)
            dests.append(dest)
    else:
        url = _latest_weekly_zip_url()
        if not url:
            print("❌ Could not locate latest weekly ZIP. Skipping.")
            return
        dest = raw_dir / "dailymed" / Path(url).name
        if not dest.exists():
            _download(url, dest)
        dests = [dest]
    try:
        with get_connection(config) as con:
            _init_schema(con)
            _ingest_xml_data(con, dests, n_max)
            _update_section_names_and_embeddings(con)
            _apply_final_constraints(con)
        print("\n🎉 DailyMed pipeline completed successfully!")
    except Exception as e:
        print(f"❌ An error occurred during the pipeline: {e}")
        raise
