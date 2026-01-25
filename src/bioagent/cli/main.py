#!/usr/bin/env python3
"""
Main CLI dispatcher for biomedagent-db commands.

This module routes different subcommands to their respective handlers.
"""

import sys
from typing import NoReturn


def show_help() -> None:
    """Show the main help message."""
    print("biomedagent-db - Database management CLI")
    print("=" * 40)
    print()
    print("Database Setup Commands:")
    print("  setup_postgres           - Run full PostgreSQL setup")
    print("  install-pgvector         - Install pgvector extension (requires sudo)")
    print("  create-vector-ext        - Create vector extension in database (requires sudo)")
    print("  install-rdkit            - Install RDKit extension (requires sudo)")
    print("  create-rdkit-ext         - Create RDKit extension in database (requires sudo)")
    print("  info                     - Show database information")
    print("  reset                    - Reset database (with confirmation)")
    print("  reset --force            - Reset database (no confirmation)")
    print("  vacuum                   - Vacuum database")
    print("  tables                   - List all tables")
    print("  create-schema            - Create public schema if missing")
    print("  fix-permissions          - Fix user permissions on public schema")
    print("  verify-deps              - Verify Python dependencies are installed")
    print()
    print("Data Ingestion Commands:")
    print("  ingest                   - Run full data ingestion pipeline")
    print("  ingest --help            - Show detailed ingestion options")
    print()
    print("  Database Operations:")
    print("  ingest --get-db-info     - Show database info and sample data")
    print("  ingest --sample-size 5   - Number of sample rows per table (default: 3)")
    print("  ingest --tables          - List all tables with row counts")
    print("  ingest --reset           - Reset database before ingestion (with confirmation)")
    print("  ingest --reset --force   - Reset database and run full ingestion (no confirmation)")
    print("  ingest --exit-on-reset   - Exit after resetting database")
    print("  ingest --vacuum          - Run VACUUM after ingestion")
    print("  ingest --dump-db file.sql - Create database backup")
    print("  ingest --restore-db file.sql - Restore database from backup")
    print("  extract-schema           - Extract database schema and sample data to a text file")
    print()
    print("  Data Source Selection:")
    print("  ingest --skip-openfda    - Skip OpenFDA ingestion")
    print("  ingest --skip-orange-book - Skip Orange Book ingestion")
    print("  ingest --skip-ctgov      - Skip ClinicalTrials.gov ingestion")
    print("  ingest --skip-dailymed   - Skip DailyMed ingestion")
    print("  ingest --skip-bindingdb  - Skip BindingDB ingestion")
    print("  ingest --skip-chembl     - Skip ChEMBL ingestion")
    print("  ingest --skip-dm-target  - Skip dm_target population (requires ChEMBL + BindingDB)")
    print("  ingest --skip-dm-molecule - Skip dm_molecule/molecular mappings (requires all sources)")
    print("  ingest --skip-drugcentral - Skip DrugCentral ingestion")
    print()
    print("  General Options:")
    print("  ingest --n-max 1000      - Max entries to process per source")
    print("  ingest --raw-dir ./data  - Directory for raw data files (default: ./raw)")
    print()
    print("  OpenFDA Options:")
    print("  ingest --openfda-files 10 - Max OpenFDA files to process (default: all)")
    print("  ingest --openfda-use-local-files - Use existing local files (skip download)")
    print()
    print("  ClinicalTrials.gov Options:")
    print("  ingest --ctgov-populate-rag - Populate RAG corpus after ingestion (WARNING: takes hours!)")
    print("  ingest --ctgov-rag-buckets 32 - Number of buckets for RAG corpus (default: 16)")
    print("  ingest --ctgov-rag-corpus-only - Only populate CT.gov RAG corpus (standalone)")
    print("  ingest --ctgov-rag-keys-only - Only populate CT.gov RAG keys (standalone)")
    print("  ingest --skip-ctgov-enriched-search - Skip enriched search table (enabled by default)")
    print("  ingest --ctgov-populate-enriched-search - Explicitly enable enriched search (enabled by default)")
    print("  ingest --ctgov-enriched-search-batch-size 2000 - Batch size for enriched search (default: 1000)")
    print("  ingest --ctgov-enriched-search-only - Only populate enriched search table (standalone)")
    print()
    print("  BindingDB Options:")
    print("  ingest --bindingdb-all-organisms - Include all organisms (default: human only)")
    print("  ingest --bindingdb-batch-size 5000 - Batch size for processing (default: 10000)")
    print("  ingest --bindingdb-force-recreate - Force recreate BindingDB tables")
    print()
    print("  ChEMBL Options:")
    print("  ingest --chembl-force-recreate - Force recreate ChEMBL tables")
    print()
    print("Search Index Commands:")
    print("  generate-search          - Create full-text search indexes (tsvector + GIN)")
    print("  generate-ctgov-search    - Create enriched CT.gov search table")
    print("    create                 - Create table and indexes")
    print("    populate [batch_size]  - Populate data (default batch: 1000)")
    print("    refresh [batch_size]   - Refresh data (truncate + reload)")
    print("    verify                 - Verify table state")
    print("    test                   - Test search queries")
    print("    drop                   - Drop table and indexes")
    print("    full [batch_size]      - Create + populate + verify + test")
    print()
    print("Molecular Mapping Commands:")
    print("  create-dm-target         - Create canonical target mappings (ChEMBL + BindingDB)")
    print("  create-dm-molecule       - Create unified molecular mappings (all sources)")
    print()
    print()
    print("Usage:")
    print("  biomedagent-db <command> [options]")
    print()
    print("Examples:")
    print("  biomedagent-db setup_postgres")
    print("  biomedagent-db info")
    print("  biomedagent-db reset --force")
    print("  biomedagent-db ingest")
    print("  biomedagent-db ingest --help")
    print("  biomedagent-db ingest --get-db-info --sample-size 5")
    print("  biomedagent-db ingest --tables")
    print("  biomedagent-db ingest --reset --force")
    print("  biomedagent-db ingest --skip-openfda --skip-ctgov")
    print("  biomedagent-db ingest --openfda-files 10 --n-max 1000")
    print("  biomedagent-db ingest --openfda-use-local-files")
    print("  biomedagent-db ingest --bindingdb-all-organisms --bindingdb-batch-size 5000")
    print("  biomedagent-db ingest --bindingdb-force-recreate")
    print("  biomedagent-db ingest --chembl-force-recreate")
    print("  biomedagent-db ingest --ctgov-populate-rag --ctgov-rag-buckets 32")
    print("  biomedagent-db ingest --ctgov-rag-corpus-only")
    print("  biomedagent-db ingest --ctgov-rag-keys-only")
    print("  biomedagent-db ingest --skip-ctgov-enriched-search  # Skip enriched search (enabled by default)")
    print("  biomedagent-db ingest --ctgov-enriched-search-only --ctgov-enriched-search-batch-size 2000")
    print("  biomedagent-db ingest --vacuum")
    print("  biomedagent-db ingest --dump-db backup.sql")
    print("  biomedagent-db ingest --restore-db backup.sql")
    print("  biomedagent-db generate-search")
    print("  biomedagent-db generate-ctgov-search create")
    print("  biomedagent-db generate-ctgov-search populate 2000")
    print("  biomedagent-db generate-ctgov-search full")
    print("  biomedagent-db create-dm-target")
    print("  biomedagent-db create-dm-molecule")
    print("  biomedagent-db extract-schema --output schema.txt --sample-rows 5")


def route_setup_postgres() -> NoReturn:
    """Route to PostgreSQL setup."""
    try:
        from bioagent.data.ingest.setup_postgres import main
        main()
    except ImportError as e:
        print(f"❌ Error importing setup_postgres: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running setup_postgres: {e}")
        sys.exit(1)
    sys.exit(0)


def route_ingest_command() -> NoReturn:
    """Route to data ingestion pipeline."""
    try:
        from bioagent.data.ingest.ingest_all_postgres import main
        # Replace sys.argv to pass arguments to the ingestion script
        # Keep the original command but replace 'ingest' with the script name
        original_argv = sys.argv.copy()
        sys.argv = ['ingest_all_postgres.py'] + sys.argv[2:]  # Skip 'biomedagent-db' and 'ingest'
        main()
    except ImportError as e:
        print(f"❌ Error importing ingestion pipeline: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running ingestion pipeline: {e}")
        sys.exit(1)
    finally:
        # Restore original sys.argv
        if 'original_argv' in locals():
            sys.argv = original_argv
    sys.exit(0)


def route_generate_search() -> NoReturn:
    """Route to full-text search index generation."""
    try:
        from bioagent.data.ingest.generate_search import create_full_text_search_indexes_sync, DEFAULT_CONFIG
        print("🔍 Starting full-text search index generation...")
        create_full_text_search_indexes_sync(DEFAULT_CONFIG)
        print("✅ Full-text search indexes created successfully!")
    except ImportError as e:
        print(f"❌ Error importing generate_search: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating search indexes: {e}")
        sys.exit(1)
    sys.exit(0)


def route_extract_schema() -> NoReturn:
    """Route to schema and examples extraction utility."""
    try:
        from bioagent.data.ingest.extract_schema_and_examples import main as extract_main
        # Preserve original argv while forwarding only subcommand args
        original_argv = sys.argv.copy()
        # Replace the script name so argparse help looks correct
        sys.argv = ["extract_schema_and_examples.py"] + sys.argv[2:]
        exit_code = extract_main()
        sys.exit(exit_code)
    except ImportError as e:
        print(f"❌ Error importing extract_schema_and_examples: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running schema extraction: {e}")
        sys.exit(1)
    finally:
        if "original_argv" in locals():
            sys.argv = original_argv


def route_generate_ctgov_search() -> NoReturn:
    """Route to CT.gov enriched search table generation."""
    try:
        from bioagent.data.ingest.generate_ctgov_enriched_search import main
        # Preserve original argv while forwarding only subcommand args
        original_argv = sys.argv.copy()
        # Replace the script name so argparse help looks correct
        sys.argv = ["generate_ctgov_enriched_search.py"] + sys.argv[2:]
        main()
    except ImportError as e:
        print(f"❌ Error importing generate_ctgov_enriched_search: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CT.gov enriched search generation: {e}")
        sys.exit(1)
    finally:
        if "original_argv" in locals():
            sys.argv = original_argv
    sys.exit(0)







def route_create_dm_target() -> NoReturn:
    """Route to dm_target canonical target mapping creation."""
    try:
        from bioagent.data.ingest.create_and_populate_dm_target import main
        print("🎯 Starting canonical target mapping creation...")
        main()
        print("✅ Canonical target mapping completed successfully!")
    except ImportError as e:
        print(f"❌ Error importing create_and_populate_dm_target: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating canonical target mapping: {e}")
        sys.exit(1)
    sys.exit(0)


def route_create_dm_molecule() -> NoReturn:
    """Route to dm_molecule unified molecular mappings creation."""
    try:
        from bioagent.data.ingest.build_molecular_mappings import main
        print("🧬 Starting unified molecular mappings creation...")
        main()
        print("✅ Unified molecular mappings completed successfully!")
    except ImportError as e:
        print(f"❌ Error importing build_molecular_mappings: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating unified molecular mappings: {e}")
        sys.exit(1)
    sys.exit(0)


def route_db_command(command: str, args: list[str]) -> NoReturn:
    """Route database management commands."""
    try:
        from bioagent.data.ingest.config import (
            DEFAULT_CONFIG,
            show_database_info,
            reset_database,
            vacuum_database,
            get_all_tables,
            ensure_public_schema,
            fix_user_permissions,
        )
        from bioagent.data.ingest.setup_postgres import (
            verify_python_dependencies,
            install_pgvector_extension,
            create_vector_extension_in_database,
            install_rdkit_extension,
            create_rdkit_extension_in_database,
        )
        
        if command == "info":
            show_database_info(DEFAULT_CONFIG)
        elif command == "reset":
            force = "--force" in args
            reset_database(DEFAULT_CONFIG, confirm=force)
        elif command == "vacuum":
            vacuum_database(DEFAULT_CONFIG)
        elif command == "tables":
            tables = get_all_tables(DEFAULT_CONFIG)
            print(f"📋 Tables in {DEFAULT_CONFIG.database}:")
            for table in tables:
                print(f"   - {table}")
        elif command == "create-schema":
            ensure_public_schema(DEFAULT_CONFIG)
        elif command == "fix-permissions":
            fix_user_permissions(DEFAULT_CONFIG)
        elif command == "verify-deps":
            success = verify_python_dependencies()
            sys.exit(0 if success else 1)
        elif command == "install-pgvector":
            success = install_pgvector_extension()
            sys.exit(0 if success else 1)
        elif command == "create-vector-ext":
            success = create_vector_extension_in_database()
            sys.exit(0 if success else 1)
        elif command == "install-rdkit":
            success = install_rdkit_extension()
            sys.exit(0 if success else 1)
        elif command == "create-rdkit-ext":
            success = create_rdkit_extension_in_database()
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Unknown database command: {command}")
            show_help()
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Error importing database functions: {e}")
        print("Make sure all dependencies are installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running {command}: {e}")
        sys.exit(1)
    sys.exit(0)


def main() -> NoReturn:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    command = sys.argv[1].lower()
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    # Route commands to their handlers
    if command == "setup_postgres":
        route_setup_postgres()
    elif command == "ingest":
        route_ingest_command()
    elif command == "extract-schema":
        route_extract_schema()
    elif command == "generate-search":
        route_generate_search()
    elif command == "generate-ctgov-search":
        route_generate_ctgov_search()
    elif command == "create-dm-target":
        route_create_dm_target()
    elif command == "create-dm-molecule":
        route_create_dm_molecule()
    elif command in ["info", "reset", "vacuum", "tables", "create-schema", "fix-permissions", "verify-deps", "install-pgvector", "create-vector-ext", "install-rdkit", "create-rdkit-ext"]:
        route_db_command(command, args)
    elif command in ["help", "--help", "-h"]:
        show_help()
        sys.exit(0)
    else:
        print(f"❌ Unknown command: {command}")
        print()
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
