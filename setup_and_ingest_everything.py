#!/usr/bin/env python3
"""
Complete Biomedical Database Setup and Ingestion Script

This script handles the complete setup and data ingestion process from scratch:

1. Verify Python dependencies
2. Install PostgreSQL extensions (pgvector, RDKit)
3. Setup PostgreSQL database and user
4. Ingest all biomedical data sources
5. Create search indexes and molecular mappings
6. Final verification and summary

Usage:
    python setup_and_ingest_everything.py              # Full setup and ingestion
    python setup_and_ingest_everything.py --reset-db   # Reset database first
    python setup_and_ingest_everything.py --fast       # Fast mode (limited data)
    python setup_and_ingest_everything.py --skip-ext   # Skip extension installation
    python setup_and_ingest_everything.py --populate-rag  # Include RAG corpus (takes hours!)

Requirements:
    - Ubuntu/Debian system with sudo access
    - Python 3.8+ with required packages installed
    - PostgreSQL (will be installed if missing)

Author: CureBench Team
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def run_command(cmd: str, description: str, cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def run_cli_command(command: str, description: str) -> bool:
    """Run a biomedagent-db CLI command."""
    cmd = f"python -m bioagent.cli.main {command}"
    return run_command(cmd, description, cwd=script_dir)


def main():
    """Main setup and ingestion orchestrator."""
    print("🚀 CureBench Complete Database Setup and Ingestion")
    print("=" * 70)

    parser = argparse.ArgumentParser(
        description="Complete biomedical database setup and ingestion"
    )
    parser.add_argument("--reset-db", action="store_true",
                       help="Reset database before ingestion")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode with limited data (for testing)")
    parser.add_argument("--skip-ext", action="store_true",
                       help="Skip PostgreSQL extension installation")
    parser.add_argument("--skip-ingest", action="store_true",
                       help="Skip data ingestion (setup only)")
    parser.add_argument("--populate-rag", action="store_true",
                       help="Populate CT.gov RAG corpus (WARNING: takes hours!)")
    parser.add_argument("--rag-buckets", type=int, default=16,
                       help="Number of buckets for RAG corpus population (default: 16)")

    args = parser.parse_args()

    start_time = time.time()
    success_count = 0
    total_steps = 0

    def step(description: str):
        nonlocal total_steps
        total_steps += 1
        print(f"\n📋 STEP {total_steps}: {description}")
        print("-" * 60)

    # ============================================================================
    # STEP 1: Verify Python Dependencies
    # ============================================================================
    step("Verify Python Dependencies")
    if run_cli_command("verify-deps", "Checking Python dependencies"):
        success_count += 1
    else:
        print("❌ Python dependencies check failed. Please run: pip install -e .")
        sys.exit(1)

    # ============================================================================
    # STEP 2: Install PostgreSQL Extensions
    # ============================================================================
    if not args.skip_ext:
        step("Install PostgreSQL Extensions")

        # Install pgvector
        if run_cli_command("install-pgvector", "Installing pgvector extension"):
            success_count += 1
        else:
            print("⚠️  pgvector installation failed - continuing anyway")

        # Install RDKit
        if run_cli_command("install-rdkit", "Installing RDKit extension"):
            success_count += 1
        else:
            print("⚠️  RDKit installation failed - continuing anyway")

        # Create extensions in database
        if run_cli_command("create-vector-ext", "Creating vector extension in database"):
            success_count += 1
        else:
            print("⚠️  Vector extension creation failed")

        if run_cli_command("create-rdkit-ext", "Creating RDKit extension in database"):
            success_count += 1
        else:
            print("⚠️  RDKit extension creation failed")

    # ============================================================================
    # STEP 3: Setup PostgreSQL Database
    # ============================================================================
    step("Setup PostgreSQL Database")
    if run_cli_command("setup_postgres", "Setting up PostgreSQL database and user"):
        success_count += 1
    else:
        print("❌ Database setup failed")
        sys.exit(1)

    # ============================================================================
    # STEP 4: Reset Database (if requested)
    # ============================================================================
    if args.reset_db:
        step("Reset Database")
        if run_cli_command("reset --force", "Resetting database"):
            success_count += 1
        else:
            print("❌ Database reset failed")
            sys.exit(1)

    # ============================================================================
    # STEP 5: Data Ingestion
    # ============================================================================
    if not args.skip_ingest:
        step("Data Ingestion Pipeline")

        # Build ingestion command
        ingest_cmd = "ingest --vacuum"

        if args.fast:
            # Fast mode: limit data for testing
            ingest_cmd += " --n-max 1000 --openfda-files 5"

        # Skip advanced molecular mappings in fast mode
        if args.fast:
            ingest_cmd += " --skip-dm-target --skip-dm-molecule"
        
        # Add RAG corpus population if requested (takes hours!)
        if args.populate_rag:
            ingest_cmd += f" --ctgov-populate-rag --ctgov-rag-buckets {args.rag_buckets}"

        if run_cli_command(ingest_cmd, "Running complete data ingestion pipeline"):
            success_count += 1
        else:
            print("❌ Data ingestion failed")
            # Don't exit - continue with post-processing

        # ============================================================================
        # STEP 6: Create Search Indexes
        # ============================================================================
        step("Create Search Indexes")
        if run_cli_command("generate-search", "Creating full-text search indexes"):
            success_count += 1
        else:
            print("⚠️  Search index creation failed - continuing")

        # ============================================================================
        # STEP 7: Create CT.gov Enriched Search
        # ============================================================================
        step("Create CT.gov Enriched Search")
        if run_cli_command("generate-ctgov-search full", "Creating CT.gov enriched search table"):
            success_count += 1
        else:
            print("⚠️  CT.gov enriched search creation failed - continuing")
        
        # Note: RAG corpus is populated during ingestion if --populate-rag flag is used
        # (This happens in STEP 5 as part of the ingestion command)

        # ============================================================================
        # STEP 8: Create Molecular Mappings (if not skipped)
        # ============================================================================
        if not args.fast:
            step("Create Molecular Mappings")

                # Use new CLI commands for molecular mappings
            if run_cli_command("create-dm-target", "Creating canonical target mappings (dm_target)"):
                success_count += 1
            else:
                print("⚠️  Canonical target mapping failed")

            if run_cli_command("create-dm-molecule", "Creating molecular mappings (dm_molecule)"):
                success_count += 1
            else:
                print("⚠️  Molecular mapping creation failed")

    # ============================================================================
    # STEP 9: Final Verification
    # ============================================================================
    step("Final Verification")
    if run_cli_command("info", "Database information and verification"):
        success_count += 1

    # ============================================================================
    # STEP 10: Extract Schema Documentation
    # ============================================================================
    step("Extract Schema Documentation")
    if run_cli_command("extract-schema --output docs/database_schema.txt --sample-rows 5",
                      "Extracting database schema and examples"):
        success_count += 1
    else:
        print("⚠️  Schema extraction failed - continuing")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("📋 SETUP AND INGESTION SUMMARY")
    print("=" * 70)

    print(f"Total steps attempted: {total_steps}")
    print(f"Steps completed successfully: {success_count}")
    print(f"Total time: {elapsed:.1f} seconds")

    if success_count >= total_steps * 0.8:  # 80% success rate
        print("\n🎉 Setup and ingestion completed successfully!")
        print("\n📊 Your biomedical database is ready for advanced queries.")
        print("\nExample commands you can now run:")
        print("  biomedagent-db info                    # Database information")
        print("  biomedagent-db tables                  # List all tables")
        print("  biomedagent-db ingest --get-db-info    # Sample data from all tables")
        print("\n📁 Schema documentation saved to: docs/database_schema.txt")
    else:
        print(f"\n⚠️  Setup completed with {success_count}/{total_steps} successful steps.")
        print("Check the logs above for any failed steps.")

    return success_count == total_steps


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)