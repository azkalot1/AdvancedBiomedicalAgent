#!/usr/bin/env python3
"""
Orange Book ingestion for PostgreSQL: download the current monthly ZIP and load product.txt.
Integrates with the same database as OpenFDA normalized structure.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import psycopg2
import requests
from tqdm import tqdm

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
    from .constants import ORANGE_BOOK_URL, DEFAULT_HEADERS
except ImportError:
    from config import DatabaseConfig, get_connection
    from constants import ORANGE_BOOK_URL, DEFAULT_HEADERS

OB_URL = ORANGE_BOOK_URL


def _download(url: str, dest: Path) -> Path:
    """Download file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"🔄 Reusing existing {dest.name}")
        return dest

    with requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(dest, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dest.name}", leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return dest


def _init_schema(con: psycopg2.extensions.connection) -> None:
    """Initialize Orange Book schema in PostgreSQL."""
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orange_book_products(
              id SERIAL PRIMARY KEY,
              appl_no TEXT,
              product_no TEXT,
              trade_name TEXT,
              ingredient TEXT,
              dosage_form TEXT,
              route TEXT,
              strength TEXT,
              te_code TEXT,
              applicant TEXT,
              approval_date TEXT,
              drug_type TEXT,  -- RX/OTC
              appl_type TEXT,  -- N/A
              rld TEXT,        -- Reference Listed Drug
              rs TEXT,         -- Reference Standard
              applicant_full_name TEXT,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW(),
              UNIQUE(appl_no, product_no)
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_orange_book_trade_name ON orange_book_products(trade_name);
            CREATE INDEX IF NOT EXISTS idx_orange_book_ingredient ON orange_book_products(ingredient);
            CREATE INDEX IF NOT EXISTS idx_orange_book_applicant ON orange_book_products(applicant);
            CREATE INDEX IF NOT EXISTS idx_orange_book_te_code ON orange_book_products(te_code);
            CREATE INDEX IF NOT EXISTS idx_orange_book_appl_no ON orange_book_products(appl_no);
            CREATE INDEX IF NOT EXISTS idx_orange_book_drug_type ON orange_book_products(drug_type);

            -- Full-text search indexes
            CREATE INDEX IF NOT EXISTS idx_orange_book_trade_name_gin
            ON orange_book_products USING GIN(to_tsvector('english', trade_name));

            CREATE INDEX IF NOT EXISTS idx_orange_book_ingredient_gin
            ON orange_book_products USING GIN(to_tsvector('english', ingredient));

            -- Trigger function for updated_at
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';

            -- Trigger for orange_book_products
            DROP TRIGGER IF EXISTS update_orange_book_updated_at ON orange_book_products;
            CREATE TRIGGER update_orange_book_updated_at
                BEFORE UPDATE ON orange_book_products
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();

            -- Orange Book patents
            CREATE TABLE IF NOT EXISTS orange_book_patents(
              id SERIAL PRIMARY KEY,
              appl_no TEXT,
              product_no TEXT,
              patent_no TEXT,
              patent_expiration_date TEXT,
              drug_substance_flag TEXT,
              drug_product_flag TEXT,
              patent_use_code TEXT,
              patent_use_code_description TEXT,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW(),
              UNIQUE(appl_no, product_no, patent_no, patent_use_code)
            );

            CREATE INDEX IF NOT EXISTS idx_orange_book_patents_appl ON orange_book_patents(appl_no);
            CREATE INDEX IF NOT EXISTS idx_orange_book_patents_product ON orange_book_patents(product_no);
            CREATE INDEX IF NOT EXISTS idx_orange_book_patents_patent_no ON orange_book_patents(patent_no);

            -- Orange Book exclusivity
            CREATE TABLE IF NOT EXISTS orange_book_exclusivity(
              id SERIAL PRIMARY KEY,
              appl_no TEXT,
              product_no TEXT,
              exclusivity_code TEXT,
              exclusivity_date TEXT,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW(),
              UNIQUE(appl_no, product_no, exclusivity_code)
            );

            CREATE INDEX IF NOT EXISTS idx_orange_book_excl_appl ON orange_book_exclusivity(appl_no);
            CREATE INDEX IF NOT EXISTS idx_orange_book_excl_product ON orange_book_exclusivity(product_no);
            CREATE INDEX IF NOT EXISTS idx_orange_book_excl_code ON orange_book_exclusivity(exclusivity_code);
        """
        )
        con.commit()


def ingest_orange_book(config: DatabaseConfig, raw_dir: Path) -> None:
    """Ingest Orange Book data into PostgreSQL."""
    dest = raw_dir / "orange_book_latest.zip"
    print("📥 Downloading Orange Book data...")
    _download(OB_URL, dest)

    try:
        with get_connection(config) as con:
            _init_schema(con)

            n_rows = 0
            records_to_insert = []
            patent_records = []
            exclusivity_records = []
            use_code_map: dict[str, str] = {}

            print("📋 Processing Orange Book ZIP file...")
            with zipfile.ZipFile(dest, "r") as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith("products.txt"):
                        print(f"📄 Found products file: {info.filename}")

                        with zf.open(info) as f:
                            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                            reader = csv.DictReader(text, delimiter="~")

                            # Collect all records first for progress tracking
                            print("🔍 Reading product records...")
                            for row in reader:
                                records_to_insert.append(
                                    (
                                        row.get("Appl_No"),
                                        row.get("Product_No"),
                                        row.get("Trade_Name"),
                                        row.get("Ingredient"),
                                        row.get("DF"),  # Dosage Form
                                        row.get("Route"),
                                        row.get("Strength"),
                                        row.get("TE_Code"),
                                        row.get("Applicant"),
                                        row.get("Approval_Date"),
                                        row.get("Type"),  # RX/OTC
                                        row.get("Appl_Type"),  # N/A
                                        row.get("RLD"),  # Reference Listed Drug
                                        row.get("RS"),  # Reference Standard
                                        row.get("Applicant_Full_Name"),
                                    )
                                )
                        break

                # Optional: usecode.txt for patent use code descriptions
                for info in zf.infolist():
                    if info.filename.lower().endswith("usecode.txt"):
                        with zf.open(info) as f:
                            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                            reader = csv.DictReader(text, delimiter="~")
                            for row in reader:
                                code = row.get("Patent_Use_Code") or row.get("Patent_Use_Code_Number")
                                desc = row.get("Patent_Use_Description") or row.get("Use_Code_Description")
                                if code and desc:
                                    use_code_map[code] = desc
                        break

                # Optional: patents.txt
                for info in zf.infolist():
                    if info.filename.lower().endswith("patent.txt"):
                        print(f"📄 Found patents file: {info.filename}")
                        with zf.open(info) as f:
                            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                            reader = csv.DictReader(text, delimiter="~")
                            for row in reader:
                                code = row.get("Patent_Use_Code")
                                patent_records.append(
                                    (
                                        row.get("Appl_No"),
                                        row.get("Product_No"),
                                        row.get("Patent_No"),
                                        row.get("Patent_Expiration_Date"),
                                        row.get("Drug_Substance_Flag"),
                                        row.get("Drug_Product_Flag"),
                                        code,
                                        use_code_map.get(code),
                                    )
                                )
                        break

                # Optional: exclusivity.txt
                for info in zf.infolist():
                    if info.filename.lower().endswith("exclusivity.txt"):
                        print(f"📄 Found exclusivity file: {info.filename}")
                        with zf.open(info) as f:
                            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                            reader = csv.DictReader(text, delimiter="~")
                            for row in reader:
                                exclusivity_records.append(
                                    (
                                        row.get("Appl_No"),
                                        row.get("Product_No"),
                                        row.get("Exclusivity_Code"),
                                        row.get("Exclusivity_Date"),
                                    )
                                )
                        break

            # Batch insert with progress bar
            if records_to_insert:
                print(f"💾 Inserting {len(records_to_insert):,} Orange Book products...")

                with con.cursor() as cur:
                    # Use batch insert for better performance
                    with tqdm(records_to_insert, desc="Inserting products", unit="record") as pbar:
                        batch = []
                        batch_size = 1000

                        for record in pbar:
                            batch.append(record)
                            n_rows += 1

                            if len(batch) >= batch_size:
                                # Execute batch insert
                                cur.executemany(
                                    """
                                    INSERT INTO orange_book_products
                                    (appl_no, product_no, trade_name, ingredient, dosage_form, route,
                                     strength, te_code, applicant, approval_date, drug_type, appl_type,
                                     rld, rs, applicant_full_name)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (appl_no, product_no) DO UPDATE SET
                                        trade_name = EXCLUDED.trade_name,
                                        ingredient = EXCLUDED.ingredient,
                                        dosage_form = EXCLUDED.dosage_form,
                                        route = EXCLUDED.route,
                                        strength = EXCLUDED.strength,
                                        te_code = EXCLUDED.te_code,
                                        applicant = EXCLUDED.applicant,
                                        approval_date = EXCLUDED.approval_date,
                                        drug_type = EXCLUDED.drug_type,
                                        appl_type = EXCLUDED.appl_type,
                                        rld = EXCLUDED.rld,
                                        rs = EXCLUDED.rs,
                                        applicant_full_name = EXCLUDED.applicant_full_name,
                                        updated_at = CURRENT_TIMESTAMP
                                """,
                                    batch,
                                )
                                con.commit()
                                batch = []
                                pbar.set_postfix_str(f"Committed {n_rows:,} records")

                        # Insert remaining records
                        if batch:
                            cur.executemany(
                                """
                                INSERT INTO orange_book_products
                                (appl_no, product_no, trade_name, ingredient, dosage_form, route,
                                 strength, te_code, applicant, approval_date, drug_type, appl_type,
                                 rld, rs, applicant_full_name)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (appl_no, product_no) DO UPDATE SET
                                    trade_name = EXCLUDED.trade_name,
                                    ingredient = EXCLUDED.ingredient,
                                    dosage_form = EXCLUDED.dosage_form,
                                    route = EXCLUDED.route,
                                    strength = EXCLUDED.strength,
                                    te_code = EXCLUDED.te_code,
                                    applicant = EXCLUDED.applicant,
                                    approval_date = EXCLUDED.approval_date,
                                    drug_type = EXCLUDED.drug_type,
                                    appl_type = EXCLUDED.appl_type,
                                    rld = EXCLUDED.rld,
                                    rs = EXCLUDED.rs,
                                    applicant_full_name = EXCLUDED.applicant_full_name,
                                    updated_at = CURRENT_TIMESTAMP
                            """,
                                batch,
                            )
                            con.commit()

        print(f"✅ [Orange Book] Successfully loaded {n_rows:,} product records.")
        if patent_records:
            with con.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO orange_book_patents
                    (appl_no, product_no, patent_no, patent_expiration_date, drug_substance_flag,
                     drug_product_flag, patent_use_code, patent_use_code_description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (appl_no, product_no, patent_no, patent_use_code) DO UPDATE SET
                        patent_expiration_date = EXCLUDED.patent_expiration_date,
                        drug_substance_flag = EXCLUDED.drug_substance_flag,
                        drug_product_flag = EXCLUDED.drug_product_flag,
                        patent_use_code_description = EXCLUDED.patent_use_code_description,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    patent_records,
                )
                con.commit()
            print(f"✅ [Orange Book] Loaded {len(patent_records):,} patent records.")
        if exclusivity_records:
            with con.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO orange_book_exclusivity
                    (appl_no, product_no, exclusivity_code, exclusivity_date)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (appl_no, product_no, exclusivity_code) DO UPDATE SET
                        exclusivity_date = EXCLUDED.exclusivity_date,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    exclusivity_records,
                )
                con.commit()
            print(f"✅ [Orange Book] Loaded {len(exclusivity_records):,} exclusivity records.")
        print("🔍 Data available in orange_book_products table")

    except Exception as e:
        print(f"❌ [Orange Book] Error during ingestion: {e}")
        raise


# Search functions have been moved to src/curebench/data/app/search_orange_book.py


if __name__ == "__main__":
    # Command line interface for Orange Book ingestion
    import argparse
    import sys

    # Handle imports for both direct execution and module import
    try:
        from .config import DEFAULT_CONFIG
    except ImportError:
        from config import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(
        description="Orange Book ingestion for PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_orange_book.py --raw-dir ./data/raw
  python build_orange_book.py -d /path/to/data/raw
        """,
    )
    parser.add_argument(
        "--raw-dir",
        "-d",
        type=Path,
        default=Path("./raw"),
        help="Directory to store downloaded Orange Book data (default: ./raw)",
    )

    args = parser.parse_args()

    print("🍊 Orange Book Ingestion")
    print("=" * 50)
    print(f"📁 Raw directory: {args.raw_dir.absolute()}")
    print()

    try:
        ingest_orange_book(DEFAULT_CONFIG, args.raw_dir)
        print()
        print("✅ Orange Book ingestion completed successfully!")
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        sys.exit(1)
