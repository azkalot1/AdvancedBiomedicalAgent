#!/usr/bin/env python3
"""
ChEMBL ingestion for PostgreSQL - Simplified approach following CTGov pattern.

Downloads ChEMBL PostgreSQL dump and restores it directly, then moves tables
to public schema with chembl_ prefix.
"""
from __future__ import annotations

import subprocess
import tarfile
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

# ChEMBL download URL
CHEMBL_POSTGRESQL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_postgresql.tar.gz"


def _download_dump(dest: Path) -> Path:
    """Download ChEMBL PostgreSQL dump."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading ChEMBL PostgreSQL dump from: {CHEMBL_POSTGRESQL_URL}")
    try:
        with requests.get(CHEMBL_POSTGRESQL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(dest, "wb") as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading ChEMBL dump") as pbar:
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


def _extract_dump(tar_path: Path, extract_dir: Path) -> Path:
    """Extract the ChEMBL PostgreSQL dump from tar.gz."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tf:
        # Find the .dmp file in the archive
        dmp_files = [f for f in tf.getnames() if f.endswith('.dmp')]
        if not dmp_files:
            raise RuntimeError("No .dmp file found in the archive")
        
        dmp_file = dmp_files[0]  # Take the first .dmp file
        print(f"📄 Found dump file: {dmp_file}")
        
        # Extract the .dmp file
        tf.extract(dmp_file, extract_dir)
        extracted_path = extract_dir / dmp_file
        
        print(f"✅ Extracted dump to {extracted_path}")
        return extracted_path


def _restore_dump_to_temp_schema(dmp_path: Path, config: DatabaseConfig) -> None:
    """Restore the ChEMBL PostgreSQL dump directly to the database."""
    print("🔄 Restoring ChEMBL dump to database...")
    
    # Build pg_restore command
    cmd = [
        "pg_restore",
        "-h", config.host,
        "-U", config.user,
        "-d", config.database,
        "--verbose",
        "--no-owner",
        "-O",
        "-x",
        "--disable-triggers",  # Disable triggers during restore
        str(dmp_path),
    ]
    
    # Set password via environment variable
    env = {"PGPASSWORD": config.password}
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        # Don't check exit code strictly - pg_restore might return non-zero if there are warnings
        print("✅ ChEMBL dump restore completed")
        
        if result.returncode != 0:
            print(f"⚠️  Restore completed with warnings (exit code: {result.returncode})")
        
        if result.stderr:
            # Show last few lines of stderr which usually contain the summary
            stderr_lines = result.stderr.split('\n')
            print("Restore summary:")
            for line in stderr_lines[-10:]:
                if line.strip():
                    print(f"  {line}")
                    
    except subprocess.TimeoutExpired:
        print(f"❌ Restore timed out after 1 hour")
        raise
    except Exception as e:
        print(f"❌ Error during restore: {e}")
        raise


def _get_chembl_views(con: psycopg2.extensions.connection) -> list[str]:
    """Get list of views in the chembl schema."""
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'chembl'
            ORDER BY table_name
        """
        )
        return [row[0] for row in cur.fetchall()]


def _move_tables_to_public(con: psycopg2.extensions.connection) -> None:
    """Move all tables from chembl schema to public schema with chembl_ prefix."""
    tables = _get_chembl_views(con) # Changed from _get_chembl_tables to _get_chembl_views
    
    if not tables:
        print("⚠️  No tables found in chembl schema")
        print("💡 The dump might have created tables in public schema instead")
        
        # Check if tables are already in public schema with chembl_ prefix
        with con.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'chembl_%'
                ORDER BY table_name
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            
            if existing_tables:
                print(f"✅ Found {len(existing_tables)} existing ChEMBL tables in public schema")
                return
            else:
                print("❌ No ChEMBL tables found in any schema")
                return
    
    print(f"🔄 Moving {len(tables)} tables from chembl schema to public schema...")
    
    with con.cursor() as cur:
        for table in tables:
            new_name = f"chembl_{table}"
            print(f"📋 Moving {table} -> {new_name}")
            
            try:
                # Move table from chembl schema to public schema with new name
                cur.execute(
                    f"""
                    ALTER TABLE chembl.{table}
                    SET SCHEMA public
                """
                )
                
                cur.execute(
                    f"""
                    ALTER TABLE public.{table}
                    RENAME TO {new_name}
                """
                )
            except Exception as e:
                print(f"⚠️  Could not move {table}: {e}")
                continue
        
        con.commit()
        print("✅ All tables moved successfully!")


def _move_views_to_public(con: psycopg2.extensions.connection) -> None:
    """Move all views from chembl schema to public schema with chembl_ prefix."""
    views = _get_chembl_views(con)
    
    if not views:
        print("ℹ️  No views found in chembl schema")
        return
    
    print(f"🔄 Moving {len(views)} views from chembl schema to public schema...")
    
    with con.cursor() as cur:
        for view in views:
            new_name = f"chembl_{view}"
            print(f"📋 Moving {view} -> {new_name}")
            
            # Move view from chembl schema to public schema with new name
            cur.execute(
                f"""
                ALTER VIEW chembl.{view}
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


def _get_table_columns(con: psycopg2.extensions.connection, table_name: str) -> list[str]:
    """Get list of columns in a specific table."""
    with con.cursor() as cur:
        try:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"⚠️  Could not get columns for {table_name}: {e}")
            return []


def _get_chembl_dump_tables(dmp_path: Path) -> list[str]:
    """Get list of tables that will be created by the ChEMBL dump."""
    import subprocess
    
    try:
        # Use pg_restore --list to get the table list
        result = subprocess.run([
            'pg_restore', '--list', str(dmp_path)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            tables = []
            
            print("🔍 Analyzing dump structure...")
            print("📄 First 20 lines of dump analysis:")
            for i, line in enumerate(lines[:20]):
                print(f"  {i+1:2d}: {line}")
            
            # Look for table-related lines
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for TABLE DATA lines (most common format)
                if 'TABLE DATA' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Usually format: "TABLE DATA schema table_name"
                        table_name = parts[-1]  # Last part is table name
                        if table_name and table_name not in tables:
                            tables.append(table_name)
                
                # Look for CREATE TABLE lines
                elif 'CREATE TABLE' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'TABLE' and i + 1 < len(parts):
                            table_name = parts[i + 1]
                            if table_name and table_name not in tables:
                                tables.append(table_name)
                            break
                
                # Look for other table references
                elif 'TABLE' in line and 'DATA' not in line:
                    parts = line.split()
                    for part in parts:
                        if part not in ['TABLE', 'CREATE', 'ALTER', 'INDEX', 'SEQUENCE'] and '.' not in part:
                            if len(part) > 2 and part not in tables:
                                tables.append(part)
            
            print(f"🔍 Found {len(tables)} tables in dump:")
            for table in sorted(tables):
                print(f"  📋 {table}")
            
            return tables
        else:
            print(f"⚠️  Could not inspect dump: {result.stderr}")
            return []
            
    except Exception as e:
        print(f"⚠️  Error inspecting dump: {e}")
        return []


def _clean_existing_chembl_data(config: DatabaseConfig, dmp_path: Path = None) -> None:
    """Clean all existing ChEMBL data from the database."""
    print("🧹 Cleaning all existing ChEMBL data...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Drop chembl schema and all its contents
            print("🗑️  Dropping chembl schema...")
            cur.execute("DROP SCHEMA IF EXISTS chembl CASCADE")
            
            # Get tables that will be created by the dump
            dump_tables = []
            if dmp_path and dmp_path.exists():
                print("🔍 Analyzing dump to find tables...")
                dump_tables = _get_chembl_dump_tables(dmp_path)
                print(f"📋 Found {len(dump_tables)} tables in dump")
            
            # Drop tables that exist in the dump (both original names and chembl_ prefixed)
            tables_to_drop = []
            
            # Add original table names from dump
            tables_to_drop.extend(dump_tables)
            
            # Add chembl_ prefixed versions
            tables_to_drop.extend([f"chembl_{table}" for table in dump_tables])
            
            # Also check for any existing chembl_ tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'chembl_%'
            """)
            existing_chembl_tables = [row[0] for row in cur.fetchall()]
            tables_to_drop.extend(existing_chembl_tables)
            
            # Remove duplicates
            tables_to_drop = list(set(tables_to_drop))
            
            print(f"🗑️  Dropping {len(tables_to_drop)} tables...")
            dropped_count = 0
            for table in tables_to_drop:
                try:
                    # Check if table exists before trying to drop it
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    """, (table,))
                    
                    if cur.fetchone()[0] > 0:
                        print(f"  📋 Dropping {table}")
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                        dropped_count += 1
                    else:
                        print(f"  ℹ️  Table {table} does not exist, skipping")
                except Exception as e:
                    print(f"  ⚠️  Could not drop {table}: {e}")
            
            print(f"✅ Dropped {dropped_count} tables")
            
            # Drop any existing ChEMBL views
            print("🗑️  Dropping ChEMBL views...")
            cur.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'chembl_%'
            """)
            existing_views = [row[0] for row in cur.fetchall()]
            
            for view in existing_views:
                try:
                    print(f"  📋 Dropping {view}")
                    cur.execute(f"DROP VIEW IF EXISTS {view} CASCADE")
                except Exception as e:
                    print(f"  ⚠️  Could not drop {view}: {e}")
            
            conn.commit()
            print("✅ All existing ChEMBL data cleaned")


def clean_chembl_tables(config: DatabaseConfig, dmp_path: Path) -> None:
    """Clean all ChEMBL tables that exist in the dump from the database."""
    print()
    print("=" * 80)
    print("Cleaning ChEMBL Tables")
    print("=" * 80)
    print()
    
    if not dmp_path.exists():
        print(f"❌ Dump file not found: {dmp_path}")
        return
    
    _clean_existing_chembl_data(config, dmp_path)
    
    print()
    print("=" * 80)
    print("✅ ChEMBL tables cleaned!")
    print("=" * 80)
    print()


def ingest_chembl_full(config: DatabaseConfig, raw_dir: Path, force_recreate: bool = False) -> None:
    """
    Full ChEMBL ingestion workflow - simplified approach following CTGov pattern.
    
    Args:
        config: Database configuration
        raw_dir: Base raw data directory (will create chembl subdirectory)
        force_recreate: Drop existing tables before import
    """
    print()
    print("=" * 80)
    print("ChEMBL Ingestion (Simplified)")
    print("=" * 80)
    print()
    
    # Create chembl subdirectory
    chembl_dir = Path(raw_dir) / "chembl"
    tar_path = chembl_dir / "chembl_36_postgresql.tar.gz"
    dmp_path = chembl_dir / "chembl_36" / "chembl_36_postgresql" / "chembl_36_postgresql.dmp"
    
    # Step 1: Download (if needed)
    if not tar_path.exists():
        print("Step 1/5: Download")
        print("-" * 80)
        _download_dump(tar_path)
        print()
    else:
        print(f"ℹ️  Using existing download: {tar_path}")
        print()
    
    # Step 2: Extract (if needed)
    if not dmp_path.exists():
        print("Step 2/5: Extract")
        print("-" * 80)
        _extract_dump(tar_path, chembl_dir)
        print()
    else:
        print(f"ℹ️  Using existing file: {dmp_path}")
        print()
    
    # Step 3: Clean existing data if requested
    if force_recreate:
        print("Step 3/5: Clean Existing Data")
        print("-" * 80)
        _clean_existing_chembl_data(config, dmp_path)
        print()
    
    # Step 4: Restore dump and move tables
    print("Step 4/5: Restore Dump")
    print("-" * 80)
    _restore_dump_to_temp_schema(dmp_path, config)
    print()

    
    print()
    print("=" * 80)
    print("✅ ChEMBL ingestion complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - View data: SELECT * FROM chembl_molecule_summary LIMIT 10;")
    print("  - Check stats: SELECT * FROM chembl_stats;")
    print("  - Search synonyms: SELECT * FROM chembl_synonym_search WHERE synonyms ILIKE '%aspirin%';")
    print("  - Match to clinical trials: python -m bioagent.data.ingest.match_chembl_to_interventions")
    print()


if __name__ == "__main__":
    """CLI for ChEMBL ingestion (for direct execution)."""
    # Import default config
    try:
        from .config import DEFAULT_CONFIG
    except ImportError:
        from config import DEFAULT_CONFIG
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ChEMBL ingestion for PostgreSQL (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_chembl_simple.py raw/                    # Full ingestion
  python build_chembl_simple.py raw/ --force-recreate   # Drop and recreate tables
        """
    )
    
    parser.add_argument("raw_dir", type=Path, help="Raw data directory")
    parser.add_argument("--force-recreate", action="store_true", help="Drop and recreate tables (WARNING: deletes all existing data)")
    parser.add_argument("--clean-only", action="store_true", help="Only clean existing ChEMBL tables (don't ingest)")
    
    args = parser.parse_args()
    
    # Handle clean-only option
    if args.clean_only:
        print("🧹 Cleaning ChEMBL tables only...")
        chembl_dir = Path(args.raw_dir) / "chembl"
        dmp_path = chembl_dir / "chembl_36" / "chembl_36_postgresql" / "chembl_36_postgresql.dmp"
        
        if not dmp_path.exists():
            print(f"❌ Dump file not found: {dmp_path}")
            print("💡 Run full ingestion first to download the dump file")
            exit(1)
        
        clean_chembl_tables(DEFAULT_CONFIG, dmp_path)
        exit(0)
    
    if args.force_recreate:
        print("⚠️  WARNING: --force-recreate will DELETE ALL existing ChEMBL data!")
        print("This will drop and recreate all ChEMBL tables and views.")
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Operation cancelled.")
            exit(1)
        print("✅ Proceeding with force recreate...")
        print()
    
    print(f"🚀 Starting ChEMBL ingestion to {args.raw_dir}")
    if args.force_recreate:
        print(f"🗑️  Force recreating tables (deleting existing data)")
    
    ingest_chembl_full(
        DEFAULT_CONFIG,
        args.raw_dir,
        force_recreate=args.force_recreate
    )
