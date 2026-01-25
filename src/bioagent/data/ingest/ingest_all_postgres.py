#!/usr/bin/env python3
"""
Unified ingestion script for all data sources into PostgreSQL.

This script coordinates the ingestion of:
1. OpenFDA drug labels (normalized structure)
2. Orange Book products
3. ClinicalTrials.gov trials
4. DailyMed SPL labels
5. BindingDB molecular targets and binding affinity data
6. ChEMBL biochemical annotations
7. dm_target canonical target mapping (combines ChEMBL + BindingDB)
8. DrugCentral molecular structures
9. dm_molecule unified molecule table + CT.gov mappings (combines all sources)

All data goes into the same PostgreSQL database for integrated querying.
"""

import subprocess
import sys
import time
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    # When run as module (through CLI)
    from .build_bindingdb import ingest_bindingdb_full
    from .build_chembl import ingest_chembl_full
    from .build_ctgov import ingest_ctgov_full, populate_rag_corpus, populate_rag_keys
    from .build_dailymed import run_dailymed_ingestion_pipeline
    from .build_drugcentral import build_drugcentral
    from .build_molecular_mappings import main as build_molecular_mappings
    from .build_openfda import build_fda_normalized
    from .build_orange_book import ingest_orange_book
    from .create_and_populate_dm_target import main as populate_dm_target
    from .generate_ctgov_enriched_search import (
        create_table as create_enriched_search_table,
        populate_table as populate_enriched_search_table,
        refresh_table as refresh_enriched_search_table,
        verify_table as verify_enriched_search_table,
        test_search as test_enriched_search,
    )
    from .config import (
        DEFAULT_CONFIG,
        DatabaseConfig,
        get_all_tables,
        get_connection,
        get_table_stats,
        reset_database,
        show_database_info,
        vacuum_database,
    )
except ImportError:
    # When run directly
    from build_bindingdb import ingest_bindingdb_full
    from build_chembl import ingest_chembl_full
    from build_ctgov import ingest_ctgov_full, populate_rag_corpus, populate_rag_keys
    from build_dailymed import run_dailymed_ingestion_pipeline
    from build_drugcentral import build_drugcentral
    from build_molecular_mappings import main as build_molecular_mappings
    from build_openfda import build_fda_normalized
    from build_orange_book import ingest_orange_book
    from create_and_populate_dm_target import main as populate_dm_target
    from generate_ctgov_enriched_search import (
        create_table as create_enriched_search_table,
        populate_table as populate_enriched_search_table,
        refresh_table as refresh_enriched_search_table,
        verify_table as verify_enriched_search_table,
        test_search as test_enriched_search,
    )
    from config import (
        DEFAULT_CONFIG,
        DatabaseConfig,
        get_all_tables,
        get_connection,
        get_table_stats,
        reset_database,
        show_database_info,
        vacuum_database,
    )


def run_drugcentral_ingestion(config: DatabaseConfig, raw_dir: Path) -> bool:
    """Run DrugCentral ingestion."""
    print("\n" + "=" * 60)
    print("🧪 DRUG CENTRAL INGESTION")
    print("=" * 60)

    try:
        start_time = time.time()
        build_drugcentral(config, raw_dir)
        elapsed = time.time() - start_time
        print(f"✅ DrugCentral ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ DrugCentral ingestion failed: {e}")
        return False


def run_openfda_ingestion(config: DatabaseConfig, raw_dir: Path, max_files: int | None = None, 
                          n_max: int | None = None, use_local_files: bool = False) -> bool:
    """Run OpenFDA normalized ingestion."""
    print("\n" + "=" * 60)
    print("🧬 OPENFDA DRUG LABELS INGESTION")
    if use_local_files:
        print("   (Using local files - skipping download)")
    if max_files:
        print(f"   (Limited to {max_files:,} files)")
    else:
        print("   (Processing all available files)")
    if n_max:
        print(f"   (Limited to {n_max:,} entries)")
    else:
        print("   (Processing all available entries)")
    print("=" * 60)

    try:
        start_time = time.time()
        build_fda_normalized(config, raw_dir, max_label_files=max_files, n_max=n_max, 
                            use_local_files=use_local_files)
        elapsed = time.time() - start_time
        print(f"✅ OpenFDA ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ OpenFDA ingestion failed: {e}")
        return False


def run_orange_book_ingestion(config: DatabaseConfig, raw_dir: Path) -> bool:
    """Run Orange Book ingestion."""
    print("\n" + "=" * 60)
    print("📙 ORANGE BOOK PRODUCTS INGESTION")
    print("=" * 60)

    try:
        start_time = time.time()
        ingest_orange_book(config, raw_dir)
        elapsed = time.time() - start_time
        print(f"✅ Orange Book ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Orange Book ingestion failed: {e}")
        return False


def run_ctgov_ingestion(config: DatabaseConfig, raw_dir: Path, n_max: int | None = None, 
                        populate_rag: bool = False, rag_buckets: int = 16,
                        populate_enriched_search: bool = True, enriched_search_batch_size: int = 1000) -> bool:
    """
    Run ClinicalTrials.gov ingestion.
    
    Args:
        config: Database configuration
        raw_dir: Directory for raw data
        n_max: Maximum number of trials (not currently used for AACT dump)
        populate_rag: Whether to populate RAG corpus after ingestion (can take hours!)
        rag_buckets: Number of buckets for RAG corpus population
        populate_enriched_search: Whether to populate enriched search table after ingestion (default: True)
        enriched_search_batch_size: Batch size for enriched search table population
    """
    print("\n" + "=" * 60)
    print("🔬 CLINICALTRIALS.GOV INGESTION")
    if n_max:
        print(f"   (Limited to {n_max:,} trials)")
    else:
        print("   (Processing all available trials)")
    print("=" * 60)

    try:
        start_time = time.time()
        ingest_ctgov_full(config, raw_dir, n_max=n_max)
        elapsed = time.time() - start_time
        print(f"✅ ClinicalTrials.gov ingestion completed in {elapsed:.1f} seconds")
        
        if populate_rag:
            print("\n⚠️  RAG corpus population requested (this can take hours!)")
            if not run_ctgov_rag_corpus(config, rag_buckets):
                print("⚠️  RAG corpus population failed, but base ingestion succeeded")
            elif not run_ctgov_rag_keys(config):
                print("⚠️  RAG keys population failed, but corpus was populated")
        else:
            print("\n💡 Note: RAG corpus not populated. To populate later, run:")
            print(f"   python build_ctgov.py populate-corpus {rag_buckets}")
            print("   python build_ctgov.py populate-keys")
        
        if populate_enriched_search:
            print("\n🔍 Creating and populating enriched search table...")
            if not run_ctgov_enriched_search(config, enriched_search_batch_size):
                print("⚠️  Enriched search table population failed, but base ingestion succeeded")
        else:
            print("\n⏭️  Enriched search table creation skipped (enabled by default, use --skip-ctgov-enriched-search to disable)")
        
        return True
    except Exception as e:
        print(f"❌ ClinicalTrials.gov ingestion failed: {e}")
        return False


def run_ctgov_rag_corpus(config: DatabaseConfig, buckets: int = 16) -> bool:
    """
    Run ClinicalTrials.gov RAG corpus population (long-running operation).
    
    Args:
        config: Database configuration
        buckets: Number of buckets to partition studies into
    """
    print("\n" + "=" * 60)
    print("🧠 CLINICALTRIALS.GOV RAG CORPUS POPULATION")
    print("   (This can take several hours!)")
    print(f"   (Processing in {buckets} buckets)")
    print("=" * 60)
    
    try:
        start_time = time.time()
        populate_rag_corpus(config, buckets)
        elapsed = time.time() - start_time
        print(f"✅ RAG corpus population completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ RAG corpus population failed: {e}")
        return False


def run_ctgov_rag_keys(config: DatabaseConfig) -> bool:
    """
    Run ClinicalTrials.gov RAG keys population and indexing.
    Should be run after RAG corpus is populated.
    
    Args:
        config: Database configuration
    """
    print("\n" + "=" * 60)
    print("🔑 CLINICALTRIALS.GOV RAG KEYS POPULATION")
    print("   (Creating materialized view and indexes)")
    print("=" * 60)
    
    try:
        start_time = time.time()
        populate_rag_keys(config)
        elapsed = time.time() - start_time
        print(f"✅ RAG keys population completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ RAG keys population failed: {e}")
        return False


def run_ctgov_enriched_search(config: DatabaseConfig, batch_size: int = 1000) -> bool:
    """
    Run ClinicalTrials.gov enriched search table creation and population.
    Creates a denormalized table optimized for flexible searching with:
    - Aggregated conditions, interventions, sponsors
    - Normalized text for trigram similarity
    - Full-text search vectors
    - Filterable metadata columns
    - Proper indexes for all search patterns
    
    Args:
        config: Database configuration
        batch_size: Batch size for population (default: 1000)
    """
    print("\n" + "=" * 60)
    print("🔍 CLINICALTRIALS.GOV ENRICHED SEARCH TABLE")
    print("   (Creating denormalized search table)")
    print(f"   (Batch size: {batch_size:,})")
    print("=" * 60)
    
    try:
        start_time = time.time()
        
        # Create table and indexes
        print("\n📋 Creating table structure and indexes...")
        create_enriched_search_table(config)
        
        # Populate data
        print("\n📊 Populating data...")
        populate_enriched_search_table(config, batch_size)
        
        # Verify
        print("\n🔍 Verifying table state...")
        verify_enriched_search_table(config)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Enriched search table completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Enriched search table creation failed: {e}")
        return False


def run_ctgov_enriched_search_only(config: DatabaseConfig, batch_size: int = 1000) -> bool:
    """
    Standalone function to run enriched search table generation.
    Can be called independently after CT.gov ingestion.
    
    Args:
        config: Database configuration
        batch_size: Batch size for population (default: 1000)
    """
    return run_ctgov_enriched_search(config, batch_size)


def run_dailymed_ingestion(config: DatabaseConfig, raw_dir: Path, n_max: int | None = None) -> bool:
    """Run DailyMed ingestion."""
    print("\n" + "=" * 60)
    print("💊 DAILYMED SPL LABELS INGESTION")
    if n_max:
        print(f"   (Limited to {n_max:,} XML files)")
    else:
        print("   (Processing all available XML files)")
    print("=" * 60)

    try:
        start_time = time.time()
        run_dailymed_ingestion_pipeline(config, raw_dir, "full", n_max=n_max)
        elapsed = time.time() - start_time
        print(f"✅ DailyMed ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ DailyMed ingestion failed: {e}")
        return False


def run_bindingdb_ingestion(config: DatabaseConfig, raw_dir: Path, n_max: int | None = None, 
                           all_organisms: bool = False, batch_size: int = 10000, 
                           force_recreate: bool = False) -> bool:
    """Run BindingDB ingestion."""
    print("\n" + "=" * 60)
    print("🧬 BINDINGDB MOLECULAR TARGETS INGESTION")
    if n_max:
        print(f"   (Limited to {n_max:,} records)")
    else:
        print("   (Processing all available records)")
    if not all_organisms:
        print("   (Human targets only)")
    else:
        print("   (All organisms)")
    if force_recreate:
        print("   (Force recreating tables)")
    print(f"   (Batch size: {batch_size:,})")
    print("=" * 60)

    try:
        start_time = time.time()
        ingest_bindingdb_full(
            config, 
            raw_dir, 
            limit=n_max, 
            human_only=not all_organisms,
            batch_size=batch_size,
            force_recreate=force_recreate
        )
        elapsed = time.time() - start_time
        print(f"✅ BindingDB ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ BindingDB ingestion failed: {e}")
        return False


def run_chembl_ingestion(config: DatabaseConfig, raw_dir: Path, force_recreate: bool = False) -> bool:
    """Run ChEMBL ingestion."""
    print("\n" + "=" * 60)
    print("🧪 CHEMBL BIOCHEMICAL ANNOTATIONS INGESTION")
    if force_recreate:
        print("   (Force recreating tables)")
    print("=" * 60)

    try:
        start_time = time.time()
        ingest_chembl_full(
            config, 
            raw_dir, 
            force_recreate=force_recreate
        )
        elapsed = time.time() - start_time
        print(f"✅ ChEMBL ingestion completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ ChEMBL ingestion failed: {e}")
        return False


def run_dm_target_population(config: DatabaseConfig) -> bool:
    """
    Run dm_target population to combine ChEMBL and BindingDB annotations.
    
    This creates canonical target mappings with:
    - Phase 1: ChEMBL base (human SINGLE_PROTEIN targets)
    - Phase 2: BindingDB augmentation (new targets + consensus marking)
    - Phase 3: Gene synonym consolidation
    - Phase 4: UniProt accession mapping
    """
    print("\n" + "=" * 60)
    print("🎯 DM_TARGET CANONICAL TARGET MAPPING")
    print("   (Combining ChEMBL + BindingDB annotations)")
    print("=" * 60)

    try:
        start_time = time.time()
        populate_dm_target()
        elapsed = time.time() - start_time
        print(f"✅ dm_target population completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ dm_target population failed: {e}")
        return False


def run_molecular_mappings(config: DatabaseConfig) -> bool:
    """
    Run molecular mapping pipeline to consolidate molecules and map to clinical trials.
    
    This creates:
    - dm_molecule: Unified molecule table (ChEMBL + DrugCentral + BindingDB)
    - dm_molecule_synonyms: Synonym dictionary for molecule matching
    - map_ctgov_molecules: CT.gov intervention to molecule mappings
    - map_product_molecules: OpenFDA/DailyMed to molecule mappings
    - dm_compound_target_activity: Unified activity view
    
    Requires: ChEMBL, DrugCentral, BindingDB, dm_target, and CT.gov tables
    """
    print("\n" + "=" * 60)
    print("🧬 DM_MOLECULE UNIFIED MOLECULAR MAPPINGS")
    print("   (Consolidating ChEMBL + DrugCentral + BindingDB)")
    print("   (Mapping CT.gov interventions to molecules)")
    print("=" * 60)

    try:
        start_time = time.time()
        build_molecular_mappings()
        elapsed = time.time() - start_time
        print(f"✅ Molecular mappings completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Molecular mappings failed: {e}")
        return False


def dump_database(config: DatabaseConfig, dump_file: Path) -> bool:
    """Create a PostgreSQL dump of the database."""
    print(f"\n💾 Creating database dump: {dump_file}")
    
    try:
        # Build pg_dump command
        cmd = [
            "pg_dump",
            "-h", config.host,
            "-U", config.user,
            "-d", config.database,
            "--verbose",
            "--no-owner",
            "--no-privileges",
            "--clean",
            "--if-exists",
            "--create",
            "-f", str(dump_file)
        ]
        
        # Set password via environment variable
        env = {"PGPASSWORD": config.password}
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ Database dump created successfully: {dump_file}")
        print(f"📊 Dump size: {dump_file.stat().st_size / (1024*1024):.1f} MB")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create database dump: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Failed to create database dump: {e}")
        return False


def restore_database(config: DatabaseConfig, dump_file: Path) -> bool:
    """Restore a PostgreSQL database from a dump file."""
    print(f"\n📥 Restoring database from: {dump_file}")
    
    if not dump_file.exists():
        print(f"❌ Dump file not found: {dump_file}")
        return False
    
    try:
        # Build psql command for restore
        cmd = [
            "psql",
            "-h", config.host,
            "-U", config.user,
            "-d", "postgres",  # Connect to postgres database to drop/create target
            "-f", str(dump_file)
        ]
        
        # Set password via environment variable
        env = {"PGPASSWORD": config.password}
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ Database restored successfully from: {dump_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to restore database: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Failed to restore database: {e}")
        return False


def get_database_sample_data(config: DatabaseConfig, sample_size: int = 3) -> None:
    """
    Get sample data from each table in the database.

    Args:
        config: Database configuration
        sample_size: Number of sample rows to show per table
    """
    print(f"\n🔍 Database Sample Data (showing {sample_size} rows per table):")
    print("=" * 70)

    try:
        tables = get_all_tables(config)
        if not tables:
            print("📋 No tables found in database.")
            return

        with get_connection(config) as con:
            with con.cursor() as cur:
                for table in sorted(tables):
                    try:
                        # Get table info
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        total_rows = cur.fetchone()[0]

                        if total_rows == 0:
                            print(f"\n📋 {table} (0 rows)")
                            continue

                        print(f"\n📋 {table} ({total_rows:,} total rows)")
                        print("-" * 50)

                        # Get sample data
                        cur.execute(f"SELECT * FROM {table} LIMIT {sample_size}")
                        sample_rows = cur.fetchall()

                        # Get column names
                        cur.execute(
                            f"""
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_name = '{table}'
                            ORDER BY ordinal_position
                        """
                        )
                        columns = cur.fetchall()

                        # Print column headers
                        col_names = [col[0] for col in columns]
                        col_types = [col[1] for col in columns]

                        # Truncate long column names for display
                        display_cols = []
                        for name, dtype in zip(col_names, col_types, strict=False):
                            if len(name) > 20:
                                display_name = name[:17] + "..."
                            else:
                                display_name = name
                            display_cols.append(f"{display_name} ({dtype})")

                        print(" | ".join(f"{col:<25}" for col in display_cols))
                        print("-" * (len(display_cols) * 26))

                        # Print sample rows
                        for row in sample_rows:
                            display_row = []
                            for _i, value in enumerate(row):
                                if value is None:
                                    display_value = "NULL"
                                elif isinstance(value, str) and len(value) > 20:
                                    display_value = value[:17] + "..."
                                else:
                                    display_value = str(value)
                                display_row.append(f"{display_value:<25}")
                            print(" | ".join(display_row))

                        if total_rows > sample_size:
                            print(f"... and {total_rows - sample_size:,} more rows")

                    except Exception as e:
                        print(f"\n📋 {table} - Error: {e}")

    except Exception as e:
        print(f"❌ Could not get sample data: {e}")


def main():
    """Main ingestion orchestrator."""
    print("🚀 BiomedicalAgent PostgreSQL Data Ingestion Pipeline")
    print("=" * 70)

    # Configuration
    config = DEFAULT_CONFIG

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="BiomedicalAgent PostgreSQL Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --get-db-info                    # Show database information and sample data
  %(prog)s --get-db-info --sample-size 5    # Show 5 sample rows per table
  %(prog)s --reset --force                  # Reset database (no confirmation)
  %(prog)s --vacuum                         # Optimize database
  %(prog)s --skip-openfda                   # Skip OpenFDA ingestion
  %(prog)s --openfda-files 10 --n-max 1000  # Limit OpenFDA files and entries
  %(prog)s --bindingdb-all-organisms        # Include all organisms in BindingDB
  %(prog)s --bindingdb-batch-size 5000      # Use smaller batch size for low memory
  %(prog)s --bindingdb-force-recreate       # Force recreate BindingDB tables
  %(prog)s --chembl-force-recreate          # Force recreate ChEMBL tables
  %(prog)s --ctgov-populate-rag             # Populate CT.gov RAG during ingestion (slow!)
  %(prog)s --ctgov-rag-corpus-only          # Only populate CT.gov RAG corpus
  %(prog)s --ctgov-rag-keys-only            # Only populate CT.gov RAG keys
  %(prog)s --ctgov-rag-buckets 32           # Use 32 buckets for RAG population
  %(prog)s --skip-ctgov-enriched-search     # Skip CT.gov enriched search table (enabled by default)
  %(prog)s --ctgov-populate-enriched-search # Explicitly enable enriched search (enabled by default)
  %(prog)s --ctgov-enriched-search-only     # Only populate enriched search table (standalone)
  %(prog)s --ctgov-enriched-search-batch-size 2000  # Batch size for enriched search
  %(prog)s --reset --force --vacuum         # Complete reset: drop all, reimport, optimize
  %(prog)s --dump-db backup.sql             # Create database dump
  %(prog)s --restore-db backup.sql          # Restore database from dump
        """,
    )
    parser.add_argument("--reset", action="store_true", help="Reset database before ingestion")
    parser.add_argument("--skip-openfda", action="store_true", help="Skip OpenFDA ingestion")
    parser.add_argument("--skip-orange-book", action="store_true", help="Skip Orange Book ingestion")
    parser.add_argument("--skip-ctgov", action="store_true", help="Skip ClinicalTrials.gov ingestion")
    parser.add_argument("--skip-dailymed", action="store_true", help="Skip DailyMed ingestion")
    parser.add_argument("--skip-bindingdb", action="store_true", help="Skip BindingDB ingestion")
    parser.add_argument("--skip-drugcentral", action="store_true", help="Skip DrugCentral ingestion")
    parser.add_argument("--skip-chembl", action="store_true", help="Skip ChEMBL ingestion")
    parser.add_argument("--skip-dm-target", action="store_true", help="Skip dm_target population (requires ChEMBL + BindingDB)")
    parser.add_argument("--skip-dm-molecule", action="store_true", help="Skip dm_molecule/molecular mappings (requires all molecule sources + dm_target)")
    parser.add_argument("--ctgov-populate-rag", action="store_true", help="Populate CT.gov RAG corpus after ingestion (WARNING: takes hours!)")
    parser.add_argument("--ctgov-rag-buckets", type=int, default=16, help="Number of buckets for CT.gov RAG corpus (default: 16)")
    parser.add_argument("--ctgov-rag-corpus-only", action="store_true", help="Only populate CT.gov RAG corpus (requires prior ingestion)")
    parser.add_argument("--ctgov-rag-keys-only", action="store_true", help="Only populate CT.gov RAG keys (requires corpus to be populated)")
    parser.add_argument("--ctgov-populate-enriched-search", action="store_true", help="Populate CT.gov enriched search table after ingestion (enabled by default)")
    parser.add_argument("--skip-ctgov-enriched-search", action="store_true", help="Skip CT.gov enriched search table creation (disabled by default)")
    parser.add_argument("--ctgov-enriched-search-batch-size", type=int, default=1000, help="Batch size for enriched search table population (default: 1000)")
    parser.add_argument("--ctgov-enriched-search-only", action="store_true", help="Only populate CT.gov enriched search table (requires prior CT.gov ingestion)")
    parser.add_argument("--bindingdb-all-organisms", action="store_true", help="Include all organisms in BindingDB (default: human only)")
    parser.add_argument("--bindingdb-batch-size", type=int, default=10000, help="Batch size for BindingDB processing (default: 10000)")
    parser.add_argument("--bindingdb-force-recreate", action="store_true", help="Force recreate BindingDB tables (WARNING: deletes existing data)")
    parser.add_argument("--chembl-force-recreate", action="store_true", help="Force recreate ChEMBL tables (WARNING: deletes existing data)")
    parser.add_argument("--openfda-files", type=int, help="Max OpenFDA files to process (default: all)")
    parser.add_argument("--openfda-use-local-files", action="store_true", help="Use existing local files in raw_dir/openfda (skip download)")
    parser.add_argument("--n-max", type=int, help="Max entries to process")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after ingestion")
    parser.add_argument("--raw-dir", type=Path, help="Raw directory", default=Path("./raw"))
    parser.add_argument("--get-db-info", action="store_true", help="Show database information and sample data")
    parser.add_argument("--sample-size", type=int, default=3, help="Number of sample rows to show per table (default: 3)")
    parser.add_argument("--tables", action="store_true", help="List all tables in the database")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts (use with --reset)")
    parser.add_argument("--exit-on-reset", action="store_true", help="Exit after resetting database")
    parser.add_argument("--dump-db", type=str, help="Create database dump to specified file (e.g., backup.sql)")
    parser.add_argument("--restore-db", type=str, help="Restore database from specified dump file")

    args = parser.parse_args()

    # Handle --get-db-info command (standalone)
    if args.get_db_info:
        print("🔍 BiomedicalAgent Database Information")
        print("=" * 50)
        print(f"🔗 Connected to: {config.host}:{config.port}/{config.database}")
        print(f"👤 User: {config.user}")

        # Show database info
        show_database_info(config)

        # Show sample data
        get_database_sample_data(config, args.sample_size)

        print("\n💡 Usage examples:")
        print("   python ingest_all_postgres.py --get-db-info --sample-size 5")
        print("   python ingest_all_postgres.py --reset --force")
        print("   python ingest_all_postgres.py --vacuum")
        print("   python ingest_all_postgres.py --tables")
        return

    # Handle --tables command (standalone)
    if args.tables:
        print("📋 Tables in database:")
        try:
            tables = get_all_tables(config)
            if tables:
                stats = get_table_stats(config)
                for table in sorted(tables):
                    count = stats.get(table, "Unknown")
                    if isinstance(count, int):
                        print(f"   - {table:<35} {count:>10,} rows")
                    else:
                        print(f"   - {table:<35} {count}")
            else:
                print("   No tables found.")
        except Exception as e:
            print(f"❌ Could not list tables: {e}")
        return

    # Handle --dump-db command (standalone)
    if args.dump_db:
        dump_file = Path(args.dump_db)
        print("💾 BiomedicalAgent Database Dump")
        print("=" * 50)
        print(f"🔗 Connected to: {config.host}:{config.port}/{config.database}")
        print(f"👤 User: {config.user}")
        print(f"📁 Dump file: {dump_file}")
        
        if dump_database(config, dump_file):
            print("\n🎉 Database dump completed successfully!")
        else:
            print("\n❌ Database dump failed!")
            sys.exit(1)
        return

    # Handle --restore-db command (standalone)
    if args.restore_db:
        dump_file = Path(args.restore_db)
        print("📥 BiomedicalAgent Database Restore")
        print("=" * 50)
        print(f"🔗 Connected to: {config.host}:{config.port}")
        print(f"👤 User: {config.user}")
        print(f"📁 Restore from: {dump_file}")
        
        if restore_database(config, dump_file):
            print("\n🎉 Database restore completed successfully!")
            print("🔍 Verifying restored database...")
            show_database_info(config)
        else:
            print("\n❌ Database restore failed!")
            sys.exit(1)
        return

    # Handle --ctgov-rag-corpus-only command (standalone)
    if args.ctgov_rag_corpus_only:
        print("🧠 ClinicalTrials.gov RAG Corpus Population (Standalone)")
        print("=" * 50)
        if run_ctgov_rag_corpus(config, args.ctgov_rag_buckets):
            print("\n🎉 RAG corpus population completed!")
            print("💡 Next step: Run with --ctgov-rag-keys-only to populate keys")
        else:
            print("\n❌ RAG corpus population failed!")
            sys.exit(1)
        return

    # Handle --ctgov-rag-keys-only command (standalone)
    if args.ctgov_rag_keys_only:
        print("🔑 ClinicalTrials.gov RAG Keys Population (Standalone)")
        print("=" * 50)
        if run_ctgov_rag_keys(config):
            print("\n🎉 RAG keys population completed!")
        else:
            print("\n❌ RAG keys population failed!")
            sys.exit(1)
        return

    # Handle --ctgov-enriched-search-only command (standalone)
    if args.ctgov_enriched_search_only:
        print("🔍 ClinicalTrials.gov Enriched Search Table (Standalone)")
        print("=" * 50)
        if run_ctgov_enriched_search_only(config, args.ctgov_enriched_search_batch_size):
            print("\n🎉 Enriched search table population completed!")
        else:
            print("\n❌ Enriched search table population failed!")
            sys.exit(1)
        return

    # Show initial database info
    print("\n📊 Initial Database State:")
    show_database_info(config)

    # Reset database if requested
    if args.reset:
        print("\n🗑️  Resetting database...")
        if not reset_database(config, confirm=args.force):
            print("❌ Database reset cancelled or failed")
            sys.exit(1)
        if args.exit_on_reset:
            return

    # Track results
    results = {}
    total_start_time = time.time()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. OpenFDA (normalized structure) - foundation for other data
    if not args.skip_openfda:
        max_files = args.openfda_files  # None means process all files
        results["openfda"] = run_openfda_ingestion(
            config, raw_dir, max_files, args.n_max, 
            use_local_files=args.openfda_use_local_files
        )
    else:
        print("\n⏭️  Skipping OpenFDA ingestion")
        results["openfda"] = True

    # 2. Orange Book (therapeutic equivalence data)
    if not args.skip_orange_book:
        results["orange_book"] = run_orange_book_ingestion(config, raw_dir)
    else:
        print("\n⏭️  Skipping Orange Book ingestion")
        results["orange_book"] = True

    # 3. ClinicalTrials.gov (clinical trials data)
    if not args.skip_ctgov:
        # Enriched search is enabled by default, unless explicitly skipped
        # --ctgov-populate-enriched-search flag is kept for backward compatibility
        populate_enriched_search = not args.skip_ctgov_enriched_search
        if args.ctgov_populate_enriched_search:
            # Explicit flag overrides skip flag
            populate_enriched_search = True
        
        results["ctgov"] = run_ctgov_ingestion(
            config, 
            raw_dir, 
            args.n_max,
            populate_rag=args.ctgov_populate_rag,
            rag_buckets=args.ctgov_rag_buckets,
            populate_enriched_search=populate_enriched_search,
            enriched_search_batch_size=args.ctgov_enriched_search_batch_size
        )
    else:
        print("\n⏭️  Skipping ClinicalTrials.gov ingestion")
        results["ctgov"] = True

    # 4. DailyMed (structured product labels)
    if not args.skip_dailymed:
        results["dailymed"] = run_dailymed_ingestion(config, raw_dir, args.n_max)
    else:
        print("\n⏭️  Skipping DailyMed ingestion")
        results["dailymed"] = True

    # 5. BindingDB (molecular targets and binding affinity data)
    if not args.skip_bindingdb:
        results["bindingdb"] = run_bindingdb_ingestion(
            config, 
            raw_dir, 
            args.n_max, 
            args.bindingdb_all_organisms,
            args.bindingdb_batch_size,
            args.bindingdb_force_recreate
        )
    else:
        print("\n⏭️  Skipping BindingDB ingestion")
        results["bindingdb"] = True

    # 6. ChEMBL (biochemical annotations and molecule naming)
    if not args.skip_chembl:
        results["chembl"] = run_chembl_ingestion(
            config, 
            raw_dir, 
            args.chembl_force_recreate
        )
    else:
        print("\n⏭️  Skipping ChEMBL ingestion")
        results["chembl"] = True

    # 7. dm_target (canonical target mapping combining ChEMBL + BindingDB)
    if not args.skip_dm_target:
        # Only run if both ChEMBL and BindingDB were successful
        if results.get("chembl", False) and results.get("bindingdb", False):
            results["dm_target"] = run_dm_target_population(config)
        else:
            print("\n⏭️  Skipping dm_target (requires successful ChEMBL + BindingDB)")
            results["dm_target"] = False
    else:
        print("\n⏭️  Skipping dm_target population")
        results["dm_target"] = True

    # 8. DrugCentral (molecular structures and chemical data)
    if not args.skip_drugcentral:
        results["drugcentral"] = run_drugcentral_ingestion(config, raw_dir)
    else:
        print("\n⏭️  Skipping DrugCentral ingestion")
        results["drugcentral"] = True

    # 9. dm_molecule (unified molecular mappings + CT.gov intervention mapping)
    if not args.skip_dm_molecule:
        # Only run if all required sources were successful
        required_sources = ["chembl", "drugcentral", "bindingdb", "dm_target", "ctgov"]
        all_required_ok = all(results.get(src, False) for src in required_sources)
        
        if all_required_ok:
            results["dm_molecule"] = run_molecular_mappings(config)
        else:
            missing = [src for src in required_sources if not results.get(src, False)]
            print(f"\n⏭️  Skipping dm_molecule (requires: {', '.join(missing)})")
            results["dm_molecule"] = False
    else:
        print("\n⏭️  Skipping dm_molecule/molecular mappings")
        results["dm_molecule"] = True

    # Post-processing
    if args.vacuum:
        print("\n🧹 Running database vacuum...")
        vacuum_database(config)

    # Final summary
    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("📋 INGESTION SUMMARY")
    print("=" * 70)

    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)

    for source, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{source.upper():<20} {status}")

    print(f"\nOverall: {success_count}/{total_count} data sources ingested successfully")
    print(f"Total time: {total_elapsed:.1f} seconds")

    # Show final database state
    print("\n📊 Final Database State:")
    show_database_info(config)

    if success_count == total_count:
        print("\n🎉 All ingestions completed successfully!")
        print("🔍 Your PostgreSQL database is ready for advanced queries.")
        print("\n📊 Key tables created:")
        print("   - bindingdb_molecules: Molecular structures and identifiers")
        print("   - bindingdb_targets: Protein targets and gene information")
        print("   - bindingdb_activities: Binding affinity measurements")
        print("   - bindingdb_human_targets: Human-specific target interactions")
        print("   - molecule_dictionary: ChEMBL molecule definitions and metadata")
        print("   - compound_structures: ChEMBL chemical structures and identifiers")
        print("   - molecule_synonyms: ChEMBL alternative molecule names")
        print("   - target_dictionary: ChEMBL protein targets")
        print("   - activities: ChEMBL bioactivity measurements")
        print("   - dm_target: Canonical target mappings (ChEMBL + BindingDB consensus)")
        print("   - dm_target_gene_synonyms: Gene symbol synonyms for targets")
        print("   - dm_target_uniprot_mappings: UniProt accession mappings")
        print("   - dm_molecule: Unified molecules (ChEMBL + DrugCentral + BindingDB)")
        print("   - dm_molecule_synonyms: Comprehensive synonym dictionary")
        print("   - map_ctgov_molecules: CT.gov intervention to molecule mappings")
        print("   - map_product_molecules: OpenFDA/DailyMed to molecule mappings")
        print("   - dm_compound_target_activity: Unified molecule-target activity view")
        print("\nExample cross-source queries you can now run:")
        print("1. Find OpenFDA drugs with Orange Book therapeutic equivalents")
        print("2. Link clinical trials to FDA-approved drugs")
        print("3. Compare OpenFDA vs DailyMed drug information")
        print("4. Cross-reference DrugCentral molecular structures with FDA data")
        print("5. Analyze trial outcomes for specific drug classes")
        print("6. Find chemical similarities using DrugCentral SMILES data")
        print("7. Find drugs targeting specific proteins using BindingDB data")
        print("8. Analyze binding affinity patterns across drug classes")
        print("9. Query dm_target for unified ChEMBL+BindingDB target annotations")
        print("10. Use dm_target_gene_synonyms to find targets by alternative names")
        print("11. Query dm_molecule for unified molecule structures by InChIKey")
        print("12. Find clinical trials for molecules targeting specific genes:")
        print("    SELECT m.pref_name, c.nct_id FROM dm_compound_target_activity a")
        print("    JOIN map_ctgov_molecules map ON a.mol_id = map.mol_id")
        print("    JOIN ctgov_studies c ON map.nct_id = c.nct_id")
        print("    WHERE a.gene_symbol = 'EGFR' AND a.pchembl_value > 7;")
    else:
        print("\n⚠️  Some ingestions failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
