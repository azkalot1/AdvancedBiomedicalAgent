#!/usr/bin/env python3
"""
ClinicalTrials.gov ingestion for PostgreSQL.

Downloads and restores CTTI AACT PostgreSQL dump, then moves tables to public schema.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Any

import psycopg2
import requests
from tqdm import tqdm

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
except ImportError:
    from config import DatabaseConfig, get_connection

# CTTI AACT dump URL
CTGOV_DUMP_URL = "https://ctti-aact.nyc3.digitaloceanspaces.com/yr48nmlriax3euffw45a773wx2ha"

# Table vector configuration for full-text search
TABLE_VECTOR_CONFIG: list[dict[str, Any]] = [
    {"table_name": "ctgov_brief_summaries", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_browse_conditions", "vector_column": "term_vector", "source_columns": ["downcase_mesh_term"]},
    {"table_name": "ctgov_browse_interventions", "vector_column": "term_vector", "source_columns": ["downcase_mesh_term"]},
    {"table_name": "ctgov_conditions", "vector_column": "name_vector", "source_columns": ["downcase_name"]},
    {"table_name": "ctgov_design_groups", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_design_outcomes", "vector_column": "search_vector", "source_columns": ["description", "measure"]},
    {"table_name": "ctgov_detailed_descriptions", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_interventions", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_outcome_measurements", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_outcomes", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_result_groups", "vector_column": "search_vector", "source_columns": ["title", "description"]},
]


def _download_dump(dest: Path) -> Path:
    """Download CTTI AACT PostgreSQL dump."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading CTTI AACT dump from: {CTGOV_DUMP_URL}")
    try:
        with requests.get(CTGOV_DUMP_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            with open(dest, "wb") as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading CTTI AACT dump") as pbar:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        print(f"✅ Downloaded dump to {dest}")
        return dest
    except Exception as e:
        print(f"❌ Failed to download dump: {e}")
        raise


def _unzip_dump(zip_path: Path, extract_dir: Path) -> Path:
    """Unzip the downloaded dump file."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the .dmp file in the archive
        dmp_files = [f for f in zf.namelist() if f.endswith('.dmp')]
        if not dmp_files:
            raise RuntimeError("No .dmp file found in the archive")

        dmp_file = dmp_files[0]  # Take the first .dmp file
        print(f"📄 Found dump file: {dmp_file}")

        # Extract the .dmp file
        zf.extract(dmp_file, extract_dir)
        extracted_path = extract_dir / dmp_file

        print(f"✅ Extracted dump to {extracted_path}")
        return extracted_path


def _restore_dump(dump_path: Path, config: DatabaseConfig) -> None:
    """Restore the PostgreSQL dump to the database."""
    print("🔄 Restoring dump to database...")

    # Build pg_restore command
    cmd = [
        "pg_restore",
        "-h",
        config.host,
        "-U",
        config.user,
        "-d",
        config.database,
        "--verbose",
        "--no-owner",
        "-O",
        "-x",
        str(dump_path),
    ]

    # Set password via environment variable
    env = {"PGPASSWORD": config.password}

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print("✅ Successfully restored dump to database")
        if result.stdout:
            print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to restore dump: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        raise


def _get_ctgov_tables(con: psycopg2.extensions.connection) -> list[str]:
    """Get list of tables in the ctgov schema."""
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'ctgov'
            ORDER BY table_name
        """
        )
        return [row[0] for row in cur.fetchall()]


def _get_ctgov_views(con: psycopg2.extensions.connection) -> list[str]:
    """Get list of views in the ctgov schema."""
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'ctgov'
            ORDER BY table_name
        """
        )
        return [row[0] for row in cur.fetchall()]


def _move_tables_to_public(con: psycopg2.extensions.connection) -> None:
    """Move all tables from ctgov schema to public schema with ctgov_ prefix."""
    tables = _get_ctgov_tables(con)

    if not tables:
        print("⚠️  No tables found in ctgov schema")
        return

    print(f"🔄 Moving {len(tables)} tables from ctgov schema to public schema...")

    with con.cursor() as cur:
        for table in tables:
            new_name = f"ctgov_{table}"
            print(f"📋 Moving {table} -> {new_name}")

            # Move table from ctgov schema to public schema with new name
            cur.execute(
                f"""
                ALTER TABLE ctgov.{table}
                SET SCHEMA public
            """
            )

            cur.execute(
                f"""
                ALTER TABLE public.{table}
                RENAME TO {new_name}
            """
            )

        con.commit()
        print("✅ All tables moved successfully!")


def _move_views_to_public(con: psycopg2.extensions.connection) -> None:
    """Move all views from ctgov schema to public schema with ctgov_ prefix."""
    views = _get_ctgov_views(con)

    if not views:
        print("⚠️  No views found in ctgov schema")
        return

    print(f"🔄 Moving {len(views)} views from ctgov schema to public schema...")

    with con.cursor() as cur:
        for view in views:
            new_name = f"ctgov_{view}"
            print(f"👁️  Moving view {view} -> {new_name}")

            # Move view from ctgov schema to public schema with new name
            cur.execute(
                f"""
                ALTER VIEW ctgov.{view}
                SET SCHEMA public
            """
            )

            cur.execute(
                f"""
                ALTER VIEW public.{view}
                RENAME TO {new_name}
            """
            )

        con.commit()
        print("✅ All views moved successfully!")


def _verify_and_cleanup_ctgov_schema(con: psycopg2.extensions.connection) -> None:
    """Verify that ctgov schema is empty and clean it up."""
    print("🔍 Verifying ctgov schema is empty and cleaning up...")
    
    with con.cursor() as cur:
        # Check for remaining tables in ctgov schema
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'ctgov'
            """
        )
        remaining_tables = cur.fetchone()[0]
        
        # Check for remaining views in ctgov schema
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.views 
            WHERE table_schema = 'ctgov'
            """
        )
        remaining_views = cur.fetchone()[0]
        
        # Check for remaining functions/procedures in ctgov schema
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.routines 
            WHERE routine_schema = 'ctgov'
            """
        )
        remaining_routines = cur.fetchone()[0]
        
        # Check for remaining types in ctgov schema
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.user_defined_types 
            WHERE user_defined_type_schema = 'ctgov'
            """
        )
        remaining_types = cur.fetchone()[0]
        
        total_remaining = remaining_tables + remaining_views + remaining_routines + remaining_types
        
        if total_remaining > 0:
            print(f"⚠️  ctgov schema still contains objects:")
            print(f"   - Tables: {remaining_tables}")
            print(f"   - Views: {remaining_views}")
            print(f"   - Functions/Procedures: {remaining_routines}")
            print(f"   - Types: {remaining_types}")
            print("   Skipping schema cleanup - manual review needed")
            return
        
        # Schema is empty, safe to drop
        print("✅ ctgov schema is empty, dropping it...")
        cur.execute("DROP SCHEMA IF EXISTS ctgov CASCADE")
        con.commit()
        print("✅ ctgov schema dropped successfully!")
        
        # Verify schema is gone
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.schemata 
            WHERE schema_name = 'ctgov'
            """
        )
        schema_exists = cur.fetchone()[0] > 0
        
        if not schema_exists:
            print("✅ Verified: ctgov schema has been completely removed")
        else:
            print("⚠️  Warning: ctgov schema still exists after drop attempt")


def verify_indexes_preserved(con: psycopg2.extensions.connection) -> None:
    """Check that indexes were preserved after schema move."""
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT
                t.tablename,
                i.indexname,
                i.indexdef
            FROM pg_indexes i
            JOIN pg_tables t ON i.tablename = t.tablename
            WHERE t.schemaname = 'public'
            AND t.tablename LIKE 'ctgov_%'
            ORDER BY t.tablename, i.indexname
        """
        )

        indexes = cur.fetchall()
        print(f"📊 Found {len(indexes)} indexes on moved tables:")

        for table, index, _indexdef in indexes:
            print(f"  📋 {table}: {index}")

        return indexes


def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Checks if a column already exists in a table using a synchronous cursor."""
    query = """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s;
    """
    cursor.execute(query, (table_name, column_name))
    return cursor.fetchone() is not None


def create_full_text_search_indexes(con: psycopg2.extensions.connection) -> None:
    """
    Creates tsvector columns and GIN indexes for full-text search on CT.gov tables.
    """
    print("🔍 Starting full-text search indexing process...")

    with con.cursor() as cursor:
        for config in TABLE_VECTOR_CONFIG:
            table = config["table_name"]
            vector_col = config["vector_column"]
            source_cols = config["source_columns"]
            index_name = f"idx_gin_{table}_{vector_col}"

            print(f"\n--- Processing table: {table} ---")

            # 1. Add the tsvector column if it doesn't exist
            if not check_column_exists(cursor, table, vector_col):
                print(f"  Adding column '{vector_col}'...")
                alter_query = f"ALTER TABLE {table} ADD COLUMN {vector_col} tsvector;"
                cursor.execute(alter_query)
                print(f"  Column '{vector_col}' added.")
            else:
                print(f"  Column '{vector_col}' already exists, skipping.")

            # 2. Populate the tsvector column
            print(f"  Populating '{vector_col}' from column(s): {', '.join(source_cols)}...")
            concatenated_sources = " || ' ' || ".join([f"coalesce({col}, '')" for col in source_cols])
            update_query = f"UPDATE {table} SET {vector_col} = to_tsvector('english', {concatenated_sources});"
            cursor.execute(update_query)
            print(f"  Column '{vector_col}' populated.")

            # 3. Create the GIN index if it doesn't exist
            print(f"  Creating GIN index '{index_name}'...")
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING GIN({vector_col});"
            cursor.execute(index_query)
            print(f"  Index '{index_name}' created or already exists.")

        # Commit all changes
        con.commit()
        print("\n✅ Full-text search indexing process completed successfully!")


def execute_sql_file_via_psql(config: DatabaseConfig, sql_file_path: Path) -> None:
    """
    Execute a SQL file using psql subprocess (better for complex SQL with meta-commands).

    Args:
        config: Database configuration
        sql_file_path: Path to the SQL file to execute
    """
    if not sql_file_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

    print(f"📄 Executing SQL file via psql: {sql_file_path.name}")

    # Build psql command
    cmd = [
        "psql",
        "-h", config.host,
        "-U", config.user,
        "-d", config.database,
        "-f", str(sql_file_path),
        "--echo-errors",
        "--set", "ON_ERROR_STOP=1"
    ]

    # Set password via environment variable
    env = {"PGPASSWORD": config.password}

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ Successfully executed {sql_file_path.name}")
        
        # Show output if there's any
        if result.stdout.strip():
            print("📋 Output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
                    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing {sql_file_path.name}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        raise


def create_rag_functionality(config: DatabaseConfig, sql_dir: Path) -> None:
    """
    Create RAG functionality by executing rag_study_json.sql via psql.

    Args:
        config: Database configuration
        sql_dir: Directory containing SQL files
    """
    rag_sql_path = sql_dir / "rag_study_json.sql"

    if not rag_sql_path.exists():
        print(f"⚠️  rag_study_json.sql not found at {rag_sql_path}, skipping RAG functionality setup")
        return

    print("🧠 Setting up RAG functionality...")
    execute_sql_file_via_psql(config, rag_sql_path)
    print("✅ RAG functionality setup completed!")


def create_full_text_search_indexes_standalone(config: DatabaseConfig) -> None:
    """
    Standalone function to create full-text search indexes on existing CT.gov tables.
    This can be called independently after data ingestion is complete.
    """
    print("🔍 Starting standalone full-text search indexing process...")

    with get_connection(config) as con:
        create_full_text_search_indexes(con)

    print("✅ Standalone full-text search indexing completed!")


def create_rag_functionality_standalone(config: DatabaseConfig, sql_dir: Path) -> None:
    """
    Standalone function to create RAG functionality by executing rag_study_json.sql.
    This can be called independently after data ingestion is complete.
    """
    print("🧠 Starting standalone RAG functionality setup...")
    create_rag_functionality(config, sql_dir)
    print("✅ Standalone RAG functionality setup completed!")


def ingest_ctgov_full(config: DatabaseConfig, raw_dir: Path, n_max: int | None = None) -> None:
    """Ingest CTTI AACT PostgreSQL dump and move tables to public schema."""
    # Set up paths
    ctgov_dir = raw_dir / "ctgov"
    zip_path = ctgov_dir / "ctti_aact_dump.zip"
    extract_dir = ctgov_dir / "extracted"
    dump_path = None

    try:
        # Step 1: Download dump if not exists
        if not zip_path.exists():
            _download_dump(zip_path)
        else:
            print(f"🔄 Reusing existing {zip_path.name}")

        # Step 2: Unzip dump
        if not extract_dir.exists() or not any(extract_dir.glob("*.dmp")):
            dump_path = _unzip_dump(zip_path, extract_dir)
        else:
            # Find existing .dmp file
            dmp_files = list(extract_dir.glob("*.dmp"))
            if dmp_files:
                dump_path = dmp_files[0]
                print(f"🔄 Reusing existing dump: {dump_path}")
            else:
                dump_path = _unzip_dump(zip_path, extract_dir)

        # Step 3: Restore dump
        _restore_dump(dump_path, config)

        # Step 4: Move tables and views from ctgov schema to public schema
        with get_connection(config) as con:
            _move_tables_to_public(con)
            _move_views_to_public(con)
            
            # Step 4.1: Verify and cleanup empty ctgov schema
            _verify_and_cleanup_ctgov_schema(con)

        # Step 5: Create full-text search indexes
        print("\n🔍 Creating full-text search indexes...")
        with get_connection(config) as con:
            create_full_text_search_indexes(con)

        # Step 6: Create RAG functionality (rag_study_json.sql)
        print("\n🧠 Setting up RAG functionality...")
        sql_dir = Path(__file__).parent  # Directory containing this script and rag_study_json.sql
        create_rag_functionality(config, sql_dir)

        print("\n🔍 After move and indexing:")
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'ctgov_%'"
                )
                public_tables = cur.fetchone()[0]
                print(f"  Tables in public schema: {public_tables}")

                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public' AND table_name LIKE 'ctgov_%'"
                )
                public_views = cur.fetchone()[0]
                print(f"  Views in public schema: {public_views}")

                cur.execute("SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public' AND tablename LIKE 'ctgov_%'")
                public_indexes = cur.fetchone()[0]
                print(f"  Indexes in public schema: {public_indexes}")

                # Count tsvector columns
                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'ctgov_%'
                    AND data_type = 'USER-DEFINED'
                    AND udt_name = 'tsvector'
                """
                )
                tsvector_columns = cur.fetchone()[0]
                print(f"  Tsvector columns created: {tsvector_columns}")

                # Check for RAG functionality
                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_study_corpus'
                """
                )
                rag_corpus_exists = cur.fetchone()[0] > 0
                print(f"  RAG study corpus table: {'✅' if rag_corpus_exists else '❌'}")

                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.views
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_study_keys'
                """
                )
                rag_keys_exists = cur.fetchone()[0] > 0
                print(f"  RAG study keys view: {'✅' if rag_keys_exists else '❌'}")

                # Verify ctgov schema is gone
                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.schemata 
                    WHERE schema_name = 'ctgov'
                    """
                )
                ctgov_schema_exists = cur.fetchone()[0] > 0
                print(f"  ctgov schema removed: {'❌' if ctgov_schema_exists else '✅'}")

            verify_indexes_preserved(con)

        print("✅ [CT.gov] Successfully ingested CTTI AACT data")
        print("🔍 Data available in public schema with ctgov_ prefix (tables and views)")
        print("🔍 Full-text search indexes created for enhanced search capabilities")
        print("🧠 RAG functionality created for advanced trial search and analysis")
        print("🧹 Original ctgov schema cleaned up and removed")

    except Exception as e:
        print(f"❌ [CT.gov] Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    # Command line interface for CT.gov operations
    import sys

    # Handle imports for both direct execution and module import
    try:
        from .config import DEFAULT_CONFIG
    except ImportError:
        from config import DEFAULT_CONFIG

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_ctgov.py ingest <raw_dir>     - Full CT.gov ingestion with tsvector indexes and RAG functionality")
        print("  python build_ctgov.py tsvector             - Create tsvector indexes on existing tables")
        print("  python build_ctgov.py rag                  - Create RAG functionality (rag_study_json.sql)")
        print("  python build_ctgov.py verify               - Verify database state")
        print("  python build_ctgov.py cleanup              - Verify and cleanup empty ctgov schema")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("❌ Please provide raw_dir path")
            print("Usage: python build_ctgov.py ingest <raw_dir>")
            sys.exit(1)

        raw_dir = Path(sys.argv[2])
        print(f"🚀 Starting full CT.gov ingestion to {raw_dir}")
        ingest_ctgov_full(DEFAULT_CONFIG, raw_dir)

    elif command == "tsvector":
        print("🔍 Creating tsvector indexes on existing CT.gov tables...")
        create_full_text_search_indexes_standalone(DEFAULT_CONFIG)

    elif command == "rag":
        print("🧠 Creating RAG functionality...")
        sql_dir = Path(__file__).parent
        create_rag_functionality_standalone(DEFAULT_CONFIG, sql_dir)

    elif command == "verify":
        print("🔍 Verifying database state...")
        with get_connection(DEFAULT_CONFIG) as con:
            verify_indexes_preserved(con)

            # Show tsvector columns
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'ctgov_%'
                    AND data_type = 'USER-DEFINED'
                    AND udt_name = 'tsvector'
                    ORDER BY table_name, column_name
                """
                )
                tsvector_cols = cur.fetchall()

                if tsvector_cols:
                    print(f"\n📊 Found {len(tsvector_cols)} tsvector columns:")
                    for table, column in tsvector_cols:
                        print(f"  - {table}.{column}")
                else:
                    print("\n⚠️  No tsvector columns found")

    elif command == "cleanup":
        print("🧹 Verifying and cleaning up ctgov schema...")
        with get_connection(DEFAULT_CONFIG) as con:
            _verify_and_cleanup_ctgov_schema(con)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
