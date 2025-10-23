#!/usr/bin/env python3
"""
openFDA ingestion with normalized data structure - PostgreSQL Version

This version uses PostgreSQL for better performance, full-text search, and scalability.
Key improvements over SQLite version:
- Native JSONB support for flexible data storage
- Advanced full-text search with tsvector
- Better concurrent access and performance
- ILIKE and advanced SQL features
- Proper ACID transactions

Normalized structure:
- mapping_package_ndc(set_id, package_ndc)
- mapping_substance_name(set_id, substance_name)
- mapping_rxcui(set_id, rxcui)
- And other mapping tables for each identifier type

Sections are normalized:
- sections(set_id, section_name, text) - multiple rows per section if it's a list

Database setup required:
```sql
CREATE DATABASE curebench;
CREATE USER curebench_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE curebench TO curebench_user;
```
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import ijson
import numpy as np
import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
    from .constants import MAPPING_TABLES, OPENFDA_DOWNLOAD_INDEX
except ImportError:
    from config import DatabaseConfig, get_connection
    from constants import MAPPING_TABLES, OPENFDA_DOWNLOAD_INDEX

DOWNLOAD_INDEX = OPENFDA_DOWNLOAD_INDEX

# --- Model and Vector Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_DIMENSION = 384  # This model outputs 384-dimensional vectors


# ----------------------------- HTTP / index ----------------------------------
def clean_html_table(html_string):
    """
    Clean up HTML table for better LLM parsing

    Args:
        html_string (str): Raw HTML table string

    Returns:
        dict: Cleaned table data with structure
    """

    # Parse HTML
    soup = BeautifulSoup(html_string, 'html.parser')

    # Extract table caption/title
    caption = soup.find('caption')
    table_title = caption.get_text(strip=True) if caption else "Table"

    # Extract footnotes
    footnotes = []
    tfoot = soup.find('tfoot')
    if tfoot:
        footnote_text = tfoot.get_text(strip=True)
        # Split numbered footnotes
        footnotes = re.findall(r'(\d+)\.\s*([^0-9]+)(?=\s*\d+\.|\s*$)', footnote_text)
        footnotes = [(num, text.strip()) for num, text in footnotes]

    # Extract headers and data
    headers = []
    data_rows = []

    # Find all rows
    rows = soup.find_all('tr')

    # Process header rows
    header_rows = []
    data_start_idx = 0

    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue

        # Check if this looks like a header row
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if any('bold' in str(cell) or cell.get('align') == 'center' for cell in cells) or i < 2:
            header_rows.append(cell_texts)
        else:
            data_start_idx = i
            break

    # Process data rows
    for row in rows[data_start_idx:]:
        cells = row.find_all(['th', 'td'])
        if cells:
            cell_texts = []
            for cell in cells:
                # Clean up cell text
                text = cell.get_text(strip=True)
                # Remove extra whitespace and superscript references temporarily
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'<sup>(\d+)</sup>', r'[\1]', text)
                cell_texts.append(text)
            data_rows.append(cell_texts)

    # Build cleaned structure
    result = {'title': table_title, 'headers': header_rows, 'data': data_rows, 'footnotes': dict(footnotes) if footnotes else {}}

    return result


def format_table_for_llm(cleaned_table):
    """
    Format cleaned table data for optimal LLM parsing

    Args:
        cleaned_table (dict): Output from clean_html_table

    Returns:
        str: Formatted table string
    """

    output = []

    # Title
    output.append(f"## {cleaned_table['title']}")
    output.append("")

    # Headers (if complex, show structure)
    if cleaned_table['headers']:
        output.append("### Table Structure:")
        for i, header_row in enumerate(cleaned_table['headers']):
            if any(header_row):  # Only show non-empty rows
                output.append(f"Header Row {i+1}: {' | '.join(filter(None, header_row))}")
        output.append("")

    # Data in markdown table format
    if cleaned_table['data']:
        output.append("### Data:")

        # Find the maximum number of columns
        max_cols = max(len(row) for row in cleaned_table['data'])

        # Create markdown table
        for i, row in enumerate(cleaned_table['data']):
            # Pad row to max columns
            padded_row = row + [''] * (max_cols - len(row))

            # Format as markdown table
            if i == 0:
                # Header separator
                output.append('| ' + ' | '.join(padded_row) + ' |')
                output.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            else:
                output.append('| ' + ' | '.join(padded_row) + ' |')

        output.append("")

    # Footnotes
    if cleaned_table['footnotes']:
        output.append("### Footnotes:")
        for num, text in cleaned_table['footnotes'].items():
            output.append(f"{num}. {text}")
        output.append("")

    # Summary statistics (if applicable)
    output.append("### Key Data Points:")

    # Extract percentages and numbers
    for row in cleaned_table['data']:
        for cell in row:
            # Look for patterns like "101 (61%)"
            matches = re.findall(r'(\d+)\s*\((\d+)%\)', cell)
            if matches:
                for num, pct in matches:
                    output.append(f"- {num} patients ({pct}%)")

    return '\n'.join(output)


def _fetch_index() -> dict:
    r = requests.get(DOWNLOAD_INDEX, timeout=60)
    r.raise_for_status()
    return r.json()


def _select_files(index: dict, endpoint: str, max_files: int | None) -> list[str]:
    """
    Resolve partition URLs for an endpoint like 'drug/label'.
    Works with both modern 'partitions' and legacy 'downloaded_files'.
    """
    node: dict = index.get("results", {})
    for key in endpoint.split("/"):
        if key:
            node = node.get(key, {}) if isinstance(node, dict) else {}
    urls: list[str] = []
    parts = node.get("partitions") or []
    if parts:
        urls = [p.get("file") for p in sorted(parts, key=lambda p: p.get("file", "")) if p.get("file")]
    if not urls:
        df = node.get("downloaded_files") or []
        urls = sorted([f.get("url") for f in df if f.get("url")])
    if max_files is not None and max_files >= 0:
        urls = urls[:max_files]
    return urls


def _download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name
    dest = dest_dir / name
    if dest.exists():
        return dest

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(dest, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {name}", leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Fallback if no content-length header
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return dest


def _log_selection(index: dict, endpoint: str, urls: list[str]) -> None:
    node = index.get("results", {})
    for key in endpoint.split("/"):
        if key:
            node = node.get(key, {}) if isinstance(node, dict) else {}
    export_date = node.get("export_date")
    total = node.get("total_records") or sum(p.get("records", 0) for p in node.get("partitions", []) or [])
    print(f"[openFDA_postgres] {endpoint}: export_date={export_date} parts={len(urls)} total_records≈{total}")
    for u in urls:
        print("  -", Path(u).name)


# ----------------------------- Normalization ---------------------------------


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-\+]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _dedup_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        items = [str(x).strip() for x in val if str(x).strip()]
    else:
        items = [str(val).strip()]
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_title_from_openfda(of: dict) -> str | None:
    """Get title from OpenFDA data, preferring brand_name over generic_name."""
    brand_names = _dedup_list(of.get("brand_name", []))
    if brand_names:
        return brand_names[0]

    generic_names = _dedup_list(of.get("generic_name", []))
    if generic_names:
        return generic_names[0]

    return None


# ----------------------------- Section handling ------------------------------

# Canonical section names
CANON_SECTIONS = [
    'abuse',
    'abuse_table',
    'accessories',
    'active_ingredient',
    'active_ingredient_table',
    'adverse_reactions',
    'adverse_reactions_table',
    'alarms',
    'animal_pharmacology_and_or_toxicology',
    'animal_pharmacology_and_or_toxicology_table',
    'ask_doctor',
    'ask_doctor_or_pharmacist',
    'ask_doctor_or_pharmacist_table',
    'ask_doctor_table',
    'boxed_warning',
    'boxed_warning_table',
    'calibration_instructions',
    'carcinogenesis_and_mutagenesis_and_impairment_of_fertility',
    'carcinogenesis_and_mutagenesis_and_impairment_of_fertility_table',
    'cleaning',
    'clinical_pharmacology',
    'clinical_pharmacology_table',
    'clinical_studies',
    'clinical_studies_table',
    'components',
    'contraindications',
    'contraindications_table',
    'controlled_substance',
    'dependence',
    'description',
    'description_table',
    'disposal_and_waste_handling',
    'do_not_use',
    'do_not_use_table',
    'dosage_and_administration',
    'dosage_and_administration_table',
    'dosage_forms_and_strengths',
    'dosage_forms_and_strengths_table',
    'drug_abuse_and_dependence',
    'drug_abuse_and_dependence_table',
    'drug_and_or_laboratory_test_interactions',
    'drug_and_or_laboratory_test_interactions_table',
    'drug_interactions',
    'drug_interactions_table',
    'general_precautions',
    'general_precautions_table',
    'geriatric_use',
    'geriatric_use_table',
    'health_care_provider_letter',
    'health_care_provider_letter_table',
    'health_claim',
    'how_supplied',
    'how_supplied_table',
    'inactive_ingredient',
    'inactive_ingredient_table',
    'indications_and_usage',
    'indications_and_usage_table',
    'information_for_owners_or_caregivers',
    'information_for_owners_or_caregivers_table',
    'information_for_patients',
    'information_for_patients_table',
    'instructions_for_use',
    'instructions_for_use_table',
    'intended_use_of_the_device',
    'keep_out_of_reach_of_children',
    'keep_out_of_reach_of_children_table',
    'labor_and_delivery',
    'laboratory_tests',
    'laboratory_tests_table',
    'mechanism_of_action',
    'mechanism_of_action_table',
    'microbiology',
    'microbiology_table',
    'nonclinical_toxicology',
    'nonclinical_toxicology_table',
    'nonteratogenic_effects',
    'nursing_mothers',
    'other_safety_information',
    'other_safety_information_table',
    'overdosage',
    'overdosage_table',
    'package_label_principal_display_panel',
    'package_label_principal_display_panel_table',
    'patient_medication_information',
    'patient_medication_information_table',
    'pediatric_use',
    'pediatric_use_table',
    'pharmacodynamics',
    'pharmacodynamics_table',
    'pharmacogenomics',
    'pharmacokinetics',
    'pharmacokinetics_table',
    'precautions',
    'precautions_table',
    'pregnancy',
    'pregnancy_or_breast_feeding',
    'pregnancy_or_breast_feeding_table',
    'pregnancy_table',
    'purpose',
    'purpose_table',
    'questions',
    'questions_table',
    'recent_major_changes',
    'recent_major_changes_table',
    'references',
    'references_table',
    'risks',
    'risks_table',
    'route',
    'safe_handling_warning',
    'spl_indexing_data_elements',
    'spl_medguide',
    'spl_medguide_table',
    'spl_patient_package_insert',
    'spl_patient_package_insert_table',
    'spl_product_data_elements',
    'spl_unclassified_section',
    'spl_unclassified_section_table',
    'statement_of_identity',
    'stop_use',
    'stop_use_table',
    'storage_and_handling',
    'storage_and_handling_table',
    'summary_of_safety_and_effectiveness',
    'teratogenic_effects',
    'troubleshooting',
    'use_in_specific_populations',
    'use_in_specific_populations_table',
    'user_safety_warnings',
    'veterinary_indications',
    'warnings',
    'warnings_and_cautions',
    'warnings_and_cautions_table',
    'warnings_table',
    'when_using',
    'when_using_table',
]


def _coerce_html(val) -> str | None:
    """
    Convert a value (str | list[str] | None) into a clean HTML string.
    """
    if isinstance(val, str):
        return format_table_for_llm(clean_html_table(val))
    elif isinstance(val, list):
        return "\n\n".join([_coerce_html(x) for x in val if x])
    else:
        return None


def _coerce_text(val) -> str | None:
    """
    Convert a value (str | list[str] | None) into a clean text block.
    If HTML-ish (e.g., tables), strip tags to readable text.
    """
    if val is None:
        return None
    if isinstance(val, list):
        pieces = [x for x in val if isinstance(x, str) and x.strip()]
        if not pieces:
            return None
        text = "\n\n".join(pieces)
    elif isinstance(val, str):
        text = val
    else:
        return None

    t = text.strip()
    if not t:
        return None
    return t


def _iter_section_texts(rec: dict) -> list[tuple[str, list[str]]]:
    """
    Yield (canonical_section_name, [text1, text2, ...]) pairs present in this label record,
    applying alias mapping and coercing to plain text. Returns list of texts to support
    normalized storage (one row per text).
    """
    out: list[tuple[str, list[str]]] = []
    for sec in CANON_SECTIONS:
        if sec.endswith("_table"):
            if sec in rec and rec[sec]:
                if isinstance(rec[sec], list):
                    texts = []
                    for item in rec[sec]:
                        txt = _coerce_html(item)
                        if txt:
                            texts.append(txt.lower())  # Convert to lowercase
                    if texts:
                        out.append((sec, texts))
                else:
                    txt = _coerce_html(rec[sec])
                    if txt:
                        out.append((sec, [txt.lower()]))  # Convert to lowercase
        else:
            if sec in rec and rec[sec]:
                if isinstance(rec[sec], list):
                    texts = []
                    for item in rec[sec]:
                        txt = _coerce_text(item)
                        if txt:
                            texts.append(txt.lower())  # Convert to lowercase
                    if texts:
                        out.append((sec, texts))
                else:
                    txt = _coerce_text(rec[sec])
                    if txt:
                        out.append((sec, [txt.lower()]))  # Convert to lowercase
    return out


# Mapping configuration imported above with other constants

# ----------------------------- Schema ----------------------------------------


def _init_normalized_schema(con: psycopg2.extensions.connection) -> None:
    """Initialize normalized schema with separate mapping tables and proper foreign keys."""
    with con.cursor() as cur:
        # Create reference tables first
        cur.execute(
            """
            -- Reference table for all set_ids
            CREATE TABLE IF NOT EXISTS set_ids(
              id SERIAL PRIMARY KEY,
              set_id TEXT UNIQUE NOT NULL,
              created_at TIMESTAMP DEFAULT NOW()
            );

            -- Reference table for all canonical section names
            CREATE TABLE IF NOT EXISTS section_names(
              id SERIAL PRIMARY KEY,
              section_name TEXT UNIQUE NOT NULL,
              created_at TIMESTAMP DEFAULT NOW()
            );

            -- Populate section_names with canonical sections
            INSERT INTO section_names(section_name)
            VALUES """
            + ",".join([f"('{section}')" for section in CANON_SECTIONS])
            + """
            ON CONFLICT (section_name) DO NOTHING;
        """
        )

        # Create core tables with foreign keys
        cur.execute(
            """
            -- Main labels metadata table
            CREATE TABLE IF NOT EXISTS labels_meta(
              set_id_id INTEGER PRIMARY KEY REFERENCES set_ids(id) ON DELETE CASCADE,
              doc_id TEXT,
              title TEXT,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW()
            );

            -- Simple mapping table for set_id to section_name relationships
            CREATE TABLE IF NOT EXISTS sections_per_set_id(
              id SERIAL PRIMARY KEY,
              set_id_id INTEGER NOT NULL REFERENCES set_ids(id) ON DELETE CASCADE,
              section_name_id INTEGER NOT NULL REFERENCES section_names(id) ON DELETE CASCADE,
              created_at TIMESTAMP DEFAULT NOW(),
              UNIQUE(set_id_id, section_name_id)
            );

            -- Normalized sections table - one row per section text
            CREATE TABLE IF NOT EXISTS sections(
              id SERIAL PRIMARY KEY,
              set_id_id INTEGER NOT NULL REFERENCES set_ids(id) ON DELETE CASCADE,
              section_name_id INTEGER NOT NULL REFERENCES section_names(id) ON DELETE CASCADE,
              text TEXT NOT NULL,
              text_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
              created_at TIMESTAMP DEFAULT NOW()
            );

            -- Indexes for core tables
            CREATE INDEX IF NOT EXISTS idx_sections_set_id_id ON sections(set_id_id);
            CREATE INDEX IF NOT EXISTS idx_sections_section_name_id ON sections(section_name_id);
            CREATE INDEX IF NOT EXISTS idx_sections_text_vector ON sections USING GIN(text_vector);
            CREATE INDEX IF NOT EXISTS idx_sections_per_set_id_set_id_id ON sections_per_set_id(set_id_id);
            CREATE INDEX IF NOT EXISTS idx_sections_per_set_id_section_name_id ON sections_per_set_id(section_name_id);

            -- Trigger to update updated_at on labels_meta
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';

            DROP TRIGGER IF EXISTS update_labels_meta_updated_at ON labels_meta;
            CREATE TRIGGER update_labels_meta_updated_at
                BEFORE UPDATE ON labels_meta
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """
        )

        # Create mapping tables dynamically from MAPPING_TABLES configuration
        for table_name, column_name, _ in MAPPING_TABLES:
            # Create table with foreign key to set_ids
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                  id SERIAL PRIMARY KEY,
                  set_id_id INTEGER NOT NULL REFERENCES set_ids(id) ON DELETE CASCADE,
                  {column_name} TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT NOW(),
                  UNIQUE(set_id_id, {column_name})
                )
            """
            )

            # Create indexes
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name} ON {table_name}({column_name})")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_set_id_id ON {table_name}(set_id_id)")

            # Create text search index for name fields
            if 'name' in column_name.lower():
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_gin
                    ON {table_name} USING GIN(to_tsvector('english', {column_name}))
                """
                )

        con.commit()


def _generate_and_populate_embeddings(con: psycopg2.extensions.connection) -> None:
    """
    Generate embeddings for section names and populate the name_embedding column.
    This function adds semantic search capabilities to the section_names table.
    """
    try:
        # Register the vector adapter right after connection
        print("Registering vector type with psycopg2...")
        register_vector(con)

        with con.cursor() as cur:
            # Alter the table to add the vector column
            print("Altering table 'section_names' to add 'name_embedding' column...")
            alter_table_query = f"""
            ALTER TABLE public.section_names
            ADD COLUMN IF NOT EXISTS name_embedding vector({VECTOR_DIMENSION});
            """
            cur.execute(alter_table_query)
            print("Table altered successfully.")

            # Fetch all section names from the database
            print("Fetching all section names from the database...")
            cur.execute("SELECT id, section_name FROM public.section_names ORDER BY id;")
            section_data = cur.fetchall()
            if not section_data:
                print("No section names found. Exiting.")
                return

            print(f"Found {len(section_data)} section names to process.")

            # Generate embeddings using Sentence Transformers
            print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
            model = SentenceTransformer(MODEL_NAME)
            names_to_encode = [row[1].replace('_', ' ') for row in section_data]

            print("Generating embeddings for all section names...")
            embeddings = model.encode(names_to_encode, show_progress_bar=True)
            print("Embeddings generated successfully.")

            # Use the robust TEMP TABLE method for updates
            print("Creating a temporary table for the update...")
            temp_table_query = f"""
            CREATE TEMP TABLE temp_embedding_update (
                id INTEGER PRIMARY KEY,
                embedding vector({VECTOR_DIMENSION})
            );
            """
            cur.execute(temp_table_query)

            # Prepare data for insertion into the temp table
            update_data = []
            for i, row in enumerate(section_data):
                update_data.append((row[0], embeddings[i]))

            print(f"Populating temporary table with {len(update_data)} records...")
            # Use execute_values for a highly efficient bulk insert
            insert_query = "INSERT INTO temp_embedding_update (id, embedding) VALUES %s"
            psycopg2.extras.execute_values(cur, insert_query, update_data)
            print("Temporary table populated successfully.")

            # Perform the final, single bulk update from the temporary table
            print("Performing final bulk update on 'section_names' table...")
            update_from_temp_query = """
            UPDATE public.section_names sn
            SET name_embedding = teu.embedding
            FROM temp_embedding_update teu
            WHERE sn.id = teu.id;
            """
            cur.execute(update_from_temp_query)
            print(f"{cur.rowcount} rows updated successfully.")

            # Create an index for faster similarity searches
            print("Creating an IVFFlat index on the 'name_embedding' column for performance...")
            num_rows = len(section_data)
            num_lists = int(np.sqrt(num_rows))

            cur.execute("DROP INDEX IF EXISTS idx_section_names_embedding;")

            index_query = f"""
            CREATE INDEX idx_section_names_embedding
            ON public.section_names
            USING ivfflat (name_embedding vector_cosine_ops)
            WITH (lists = {num_lists});
            """
            cur.execute(index_query)
            print("Index created successfully.")

        # Commit the transaction to make the changes permanent
        con.commit()
        print("\nProcess complete! Your 'section_names' table is now ready for semantic search.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if con:
            con.rollback()  # Roll back changes on error
    except Exception as e:
        # Catch other potential errors, e.g., from the model loading
        print(f"An unexpected error occurred: {e}")
        if con:
            con.rollback()


# ----------------------------- Ingestion helpers -----------------------------


def _get_or_insert_set_id(cur: psycopg2.extensions.cursor, set_id: str) -> int:
    """Get or insert set_id and return its ID."""
    cur.execute("SELECT id FROM set_ids WHERE set_id = %s", (set_id,))
    result = cur.fetchone()
    if result:
        return result[0]

    cur.execute("INSERT INTO set_ids(set_id) VALUES (%s) RETURNING id", (set_id,))
    return cur.fetchone()[0]


def _get_section_name_id(cur: psycopg2.extensions.cursor, section_name: str) -> int:
    """Get section_name ID (should exist from initialization)."""
    cur.execute("SELECT id FROM section_names WHERE section_name = %s", (section_name,))
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        raise ValueError(f"Section name '{section_name}' not found in section_names table")


def _insert_mapping_values(
    cur: psycopg2.extensions.cursor, set_id_id: int, table_name: str, column_name: str, values: list[str]
) -> None:
    """Insert values into a mapping table using efficient batch insert."""
    if not values:
        return

    # Use VALUES clause for efficient batch insert
    query = f"""
        INSERT INTO {table_name}(set_id_id, {column_name})
        VALUES {",".join("(%s, %s)" for _ in values)}
        ON CONFLICT (set_id_id, {column_name}) DO NOTHING
    """

    # Flatten the parameters: (set_id_id, value1, set_id_id, value2, ...)
    params = []
    for value in _dedup_list(values):
        if value:
            params.extend([set_id_id, value])

    if params:
        cur.execute(query, params)


# ----------------------------- Ingestion (labels) ----------------------------


def ingest_drug_label_normalized(
    config: DatabaseConfig, index: dict, raw_dir: Path, max_files: int | None, n_max: int | None
) -> int:
    """Ingest openFDA drug labels into normalized PostgreSQL schema."""
    urls = _select_files(index, "drug/label", max_files)
    _log_selection(index, "drug/label", urls)

    total_sections = 0
    total_records = 0

    with get_connection(config) as con:
        with con.cursor() as cur:
            # Initialize overall progress tracking
            all_records = []

            # First pass: collect all records for accurate progress tracking
            print("🔍 Scanning files to count total records...")
            with tqdm(urls, desc="Scanning files", unit="file") as scan_pbar:
                for url in scan_pbar:
                    scan_pbar.set_postfix_str(f"Scanning {Path(url).name}")
                    path = _download(url, raw_dir / "openfda")

                    with zipfile.ZipFile(path, "r") as zf:
                        json_files = [info for info in zf.infolist() if info.filename.endswith(".json")]
                        for info in json_files:
                            with zf.open(info) as f:
                                parser = ijson.items(f, "results.item")
                                for rec in parser:
                                    of = rec.get("openfda") or {}
                                    set_id = rec.get("set_id") or (_dedup_list(of.get("spl_set_id"))[:1] or [None])[0]
                                    if set_id:
                                        all_records.append((url, info.filename, rec))

            print(f"📊 Found {len(all_records):,} total records to process")

            # Second pass: process records with accurate progress bar
            with tqdm(all_records, desc="Processing records", unit="record") as record_pbar:
                for _url, filename, rec in record_pbar:
                    total_records += 1

                    # Update progress bar description with current file info
                    record_pbar.set_postfix_str(f"File: {Path(filename).name}, Sections: {total_sections:,}")

                    of = rec.get("openfda") or {}
                    set_id = rec.get("set_id") or (_dedup_list(of.get("spl_set_id"))[:1] or [None])[0]
                    if not set_id:
                        continue

                    doc_id = rec.get("id")
                    title = _get_title_from_openfda(of)

                    # Get or insert set_id and get its ID
                    set_id_id = _get_or_insert_set_id(cur, set_id)

                    # Check if we already have this set_id (simple existence check)
                    cur.execute("SELECT 1 FROM labels_meta WHERE set_id_id = %s", (set_id_id,))
                    existing = cur.fetchone()

                    # Process if new or if we want to update existing records
                    if not existing:
                        # Delete old sections for this set_id
                        cur.execute("DELETE FROM sections WHERE set_id_id = %s", (set_id_id,))
                        cur.execute("DELETE FROM sections_per_set_id WHERE set_id_id = %s", (set_id_id,))

                        # Insert normalized sections and track section names
                        section_names_added = set()
                        for sec_name, texts in _iter_section_texts(rec):
                            section_name_id = _get_section_name_id(cur, sec_name)

                            # Add to sections_per_set_id if not already added
                            if sec_name not in section_names_added:
                                cur.execute(
                                    "INSERT INTO sections_per_set_id(set_id_id, section_name_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                                    (set_id_id, section_name_id),
                                )
                                section_names_added.add(sec_name)

                            # Insert section texts
                            for text in texts:
                                cur.execute(
                                    "INSERT INTO sections(set_id_id, section_name_id, text) VALUES (%s, %s, %s)",
                                    (set_id_id, section_name_id, text),
                                )
                                total_sections += 1

                        # Update metadata
                        cur.execute(
                            """INSERT INTO labels_meta(set_id_id, doc_id, title)
                               VALUES (%s, %s, %s)
                               ON CONFLICT(set_id_id) DO UPDATE SET
                                 doc_id = EXCLUDED.doc_id,
                                 title = COALESCE(EXCLUDED.title, labels_meta.title)""",
                            (set_id_id, doc_id, title),
                        )

                        # Delete old mapping entries for this set_id
                        for table_name, _, _ in MAPPING_TABLES:
                            cur.execute(f"DELETE FROM {table_name} WHERE set_id_id = %s", (set_id_id,))

                        # Insert normalized mapping data using the MAPPING_TABLES configuration
                        for table_name, column_name, openfda_field in MAPPING_TABLES:
                            _insert_mapping_values(cur, set_id_id, table_name, column_name, of.get(openfda_field, []))

                        # Commit every 100 records for better performance
                        if total_records % 100 == 0:
                            con.commit()

                        if n_max and total_records >= n_max:
                            break

            # Final commit
            con.commit()

            print(f"\n[openFDA_postgres] Processed {total_records:,} records, extracted {total_sections:,} sections")
            print("🔍 PostgreSQL automatically maintains indexes and full-text search vectors")

    return total_sections


# ----------------------------- Orchestrator ----------------------------------


def build_fda_normalized(
    config: DatabaseConfig, raw_dir: Path, max_label_files: int | None = None, n_max: int | None = None
) -> None:
    """
    Ingest openFDA labels into normalized PostgreSQL schema. Pass max_label_files=None for all partitions.
    """
    try:
        con = get_connection(config)
        _init_normalized_schema(con)

        # Generate embeddings for section names to enable semantic search
        print("\n🧠 Generating embeddings for semantic search...")
        _generate_and_populate_embeddings(con)

        con.close()

        print(f"✅ Connected to PostgreSQL at {config.host}:{config.port}/{config.database}")
    except psycopg2.Error as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print(f"Make sure PostgreSQL is running and database '{config.database}' exists")
        return

    index = _fetch_index()

    label_rows = ingest_drug_label_normalized(config, index, raw_dir=raw_dir, max_files=max_label_files, n_max=n_max)

    print("\n🎉 [openFDA_postgres] Normalized ingestion complete!")
    print(f"📊 Total sections processed: {label_rows:,}")
    print(f"💾 Database: {config.database} on {config.host}:{config.port}")
    print("🔍 Ready for advanced searches with PostgreSQL features")
    print("🧠 Semantic search enabled with vector embeddings")
