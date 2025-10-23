#!/usr/bin/env python3
"""
Unified ingestion script for all data sources into PostgreSQL.

This script coordinates the ingestion of:
1. OpenFDA drug labels (normalized structure)
2. Orange Book products
3. ClinicalTrials.gov trials
4. DailyMed SPL labels
5. DrugCentral molecular structures

All data goes into the same PostgreSQL database for integrated querying.
"""

import subprocess
import sys
import time
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    # When run as module (through CLI)
    from .build_ctgov import ingest_ctgov_full
    from .build_dailymed import run_dailymed_ingestion_pipeline
    from .build_drugcentral import build_drugcentral
    from .build_openfda import build_fda_normalized
    from .build_orange_book import ingest_orange_book
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
    from build_ctgov import ingest_ctgov_full
    from build_dailymed import run_dailymed_ingestion_pipeline
    from build_drugcentral import build_drugcentral
    from build_openfda import build_fda_normalized
    from build_orange_book import ingest_orange_book
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


def run_openfda_ingestion(config: DatabaseConfig, raw_dir: Path, max_files: int | None = None, n_max: int | None = None) -> bool:
    """Run OpenFDA normalized ingestion."""
    print("\n" + "=" * 60)
    print("🧬 OPENFDA DRUG LABELS INGESTION")
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
        build_fda_normalized(config, raw_dir, max_label_files=max_files, n_max=n_max)
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


def run_ctgov_ingestion(config: DatabaseConfig, raw_dir: Path, n_max: int | None = None) -> bool:
    """Run ClinicalTrials.gov ingestion."""
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
        return True
    except Exception as e:
        print(f"❌ ClinicalTrials.gov ingestion failed: {e}")
        return False


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


# TODO: Implement drug mapping combination across all data sources
# def run_combine_mapping(config: DatabaseConfig) -> bool:
#     """Run drug mapping combination across all data sources."""
#     print("\n" + "=" * 60)
#     print("🔗 DRUG MAPPING COMBINATION")
#     print("=" * 60)
#
#     try:
#         start_time = time.time()
#         create_mapping(config)  # Function not yet implemented
#         elapsed = time.time() - start_time
#         print(f"✅ Drug mapping combination completed in {elapsed:.1f} seconds")
#         return True
#     except Exception as e:
#         print(f"❌ Drug mapping combination failed: {e}")
#         return False


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
  %(prog)s --dump-db backup.sql             # Create database dump
  %(prog)s --restore-db backup.sql          # Restore database from dump
        """,
    )
    parser.add_argument("--reset", action="store_true", help="Reset database before ingestion")
    parser.add_argument("--skip-openfda", action="store_true", help="Skip OpenFDA ingestion")
    parser.add_argument("--skip-orange-book", action="store_true", help="Skip Orange Book ingestion")
    parser.add_argument("--skip-ctgov", action="store_true", help="Skip ClinicalTrials.gov ingestion")
    parser.add_argument("--skip-dailymed", action="store_true", help="Skip DailyMed ingestion")
    parser.add_argument("--skip-drugcentral", action="store_true", help="Skip DrugCentral ingestion")
    parser.add_argument("--openfda-files", type=int, help="Max OpenFDA files to process (default: all)")
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
        results["openfda"] = run_openfda_ingestion(config, raw_dir, max_files, args.n_max)
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
        results["ctgov"] = run_ctgov_ingestion(config, raw_dir, args.n_max)
    else:
        print("\n⏭️  Skipping ClinicalTrials.gov ingestion")
        results["ctgov"] = True

    # 4. DailyMed (structured product labels)
    if not args.skip_dailymed:
        results["dailymed"] = run_dailymed_ingestion(config, raw_dir, args.n_max)
    else:
        print("\n⏭️  Skipping DailyMed ingestion")
        results["dailymed"] = True

    # 5. DrugCentral (molecular structures and chemical data)
    if not args.skip_drugcentral:
        results["drugcentral"] = run_drugcentral_ingestion(config, raw_dir)
    else:
        print("\n⏭️  Skipping DrugCentral ingestion")
        results["drugcentral"] = True

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
        print("   - drug_mapping_final: Cross-source drug mappings with match types")
        print("   - drug_mapping_summary: Aggregated view of drug mappings")
        print("\nExample cross-source queries you can now run:")
        print("1. Find OpenFDA drugs with Orange Book therapeutic equivalents")
        print("2. Link clinical trials to FDA-approved drugs")
        print("3. Compare OpenFDA vs DailyMed drug information")
        print("4. Cross-reference DrugCentral molecular structures with FDA data")
        print("5. Analyze trial outcomes for specific drug classes")
        print("6. Find chemical similarities using DrugCentral SMILES data")
        print("7. Query drug_mapping_final for exact matches across all sources")
        print("8. Use drug_mapping_summary for aggregated drug relationships")
    else:
        print("\n⚠️  Some ingestions failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
