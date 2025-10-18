#!/usr/bin/env python3
"""
PostgreSQL setup script for data ingestion.

This script helps set up the PostgreSQL database and user for the data pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def check_postgresql_installed() -> bool:
    """Check if PostgreSQL is installed and accessible."""
    print("🔍 Checking PostgreSQL installation...")

    # Check if psql is available
    if not run_command("which psql", "Checking for psql command"):
        print("❌ PostgreSQL client (psql) not found. Please install PostgreSQL first.")
        print("   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
        print("   macOS: brew install postgresql")
        print("   Windows: Download from https://www.postgresql.org/download/")
        return False

    # Check if PostgreSQL service is running
    if not run_command("pg_isready", "Checking PostgreSQL service"):
        print("❌ PostgreSQL service is not running. Please start it:")
        print("   Ubuntu/Debian: sudo systemctl start postgresql")
        print("   macOS: brew services start postgresql")
        print("   Windows: Start PostgreSQL service from Services manager")
        return False

    return True


def setup_database():
    """Set up the database and user."""
    print("\n🏗️  Setting up database...")

    db_name = "database"
    user_name = "database_user"
    password = "database_password"

    # Create database and user
    sql_commands = f"""
-- Create user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{user_name}') THEN
        CREATE USER {user_name} WITH PASSWORD '{password}';
    END IF;
END
$$;

-- Create database if not exists
SELECT 'CREATE DATABASE {db_name}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{db_name}')\\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user_name};

-- Connect to the database and enable pgvector extension
\\c {db_name}
CREATE EXTENSION IF NOT EXISTS vector;
"""

    # Write SQL to temporary file
    sql_file = Path("/tmp/setup_database.sql")
    with open(sql_file, "w") as f:
        f.write(sql_commands)

    # Execute SQL as postgres user
    if run_command(f"sudo -u postgres psql -f {sql_file}", "Creating database and user"):
        print(f"✅ Database '{db_name}' and user '{user_name}' created successfully")

        # Test connection
        test_cmd = f"PGPASSWORD={password} psql -h localhost -U {user_name} -d {db_name} -c 'SELECT version();'"
        if run_command(test_cmd, "Testing database connection"):
            print("✅ Database connection test successful")
            return True
        else:
            print("❌ Database connection test failed")
            return False
    else:
        return False


def install_python_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing Python dependencies...")

    packages = [
        "psycopg2-binary",  # PostgreSQL adapter
        "tqdm",  # Progress bars
        "requests",  # HTTP requests
        "beautifulsoup4",  # HTML parsing
        "lxml",  # XML/HTML parser
        "python-dateutil",  # Date parsing
        "typer",  # CLI framework
        "numpy",  # Numerical computing
        "sentence-transformers",  # Semantic embeddings
        "pgvector",  # PostgreSQL vector extension
        "ijson",  # JSON streaming parser
    ]

    for package in packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            continue
        else:
            print(f"❌ Failed to install {package}")
            return False

    print("✅ All Python dependencies installed")
    return True


def create_config_file():
    """Create a configuration file for database connection."""
    print("\n⚙️  Creating configuration file...")

    config_content = '''#!/usr/bin/env python3
"""
Database configuration for data ingestion.
Modify these settings to match your PostgreSQL setup.
"""
from typing import Any

import psycopg2
import psycopg2.extras
from tqdm import tqdm


class DatabaseConfig:
    """Database configuration settings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "database",
        user: str = "database_user",
        password: str = "database_password",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def get_connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_psycopg2_params(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "database": self.database, "user": self.user, "password": self.password}


# Default database configuration
DEFAULT_CONFIG = DatabaseConfig(
    host="localhost", port=5432, database="database", user="database_user", password="database_password"
)


def get_connection(config: DatabaseConfig) -> psycopg2.extensions.connection:
    """Get a PostgreSQL connection."""
    return psycopg2.connect(**config.get_psycopg2_params())


def ensure_public_schema(config: DatabaseConfig) -> bool:
    """
    Ensure the public schema exists in the database.

    Args:
        config: Database configuration

    Returns:
        True if schema exists or was created successfully, False if failed
    """
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                # Check if public schema exists
                cur.execute(
                    """
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name = 'public'
                    """
                )

                if cur.fetchone():
                    print("✅ Public schema already exists")
                    return True

                # Create public schema
                print("🔧 Creating public schema...")
                cur.execute("CREATE SCHEMA public")
                con.commit()

                print("✅ Public schema created successfully")
                return True

    except Exception as e:
        print(f"❌ Failed to create public schema: {e}")
        return False


def get_all_tables(config: DatabaseConfig) -> list[str]:
    """Get list of all tables in the database (excluding system tables)."""
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            return [row[0] for row in cur.fetchall()]


def get_table_stats(config: DatabaseConfig) -> dict[str, int]:
    """Get row counts for all tables."""
    tables = get_all_tables(config)
    stats = {}

    with get_connection(config) as con:
        with con.cursor() as cur:
            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    stats[table] = count
                except Exception as e:
                    stats[table] = f"Error: {e}"

    return stats


def reset_database(config: DatabaseConfig, confirm: bool = False) -> bool:
    """
    Reset the database by dropping all tables and schemas.

    Args:
        config: Database configuration
        confirm: If True, skip confirmation prompt (use with caution!)

    Returns:
        True if reset was successful, False if cancelled or failed
    """
    if not confirm:
        print("⚠️  WARNING: This will permanently delete ALL data in the database!")
        print(f"   Database: {config.database}")
        print(f"   Host: {config.host}:{config.port}")

        # Show current table stats
        try:
            stats = get_table_stats(config)
            if stats:
                print("\\n📊 Current tables and row counts:")
                for table, count in stats.items():
                    print(f"   - {table}: {count:,} rows" if isinstance(count, int) else f"   - {table}: {count}")
            else:
                print("\\n📊 No tables found in database.")
        except Exception as e:
            print(f"\\n❌ Could not get table stats: {e}")

        response = input("\\n❓ Are you sure you want to reset the database? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Database reset cancelled.")
            return False

    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                # First, drop the ctgov schema if it exists
                cur.execute("DROP SCHEMA IF EXISTS ctgov CASCADE")
                print("🗑️  Dropped ctgov schema (if it existed)")

                # Get all tables in public schema
                tables = get_all_tables(config)

                if not tables:
                    print("✅ No tables found in public schema to drop.")
                else:
                    print(f"🗑️  Dropping {len(tables)} tables from public schema...")

                    # Drop tables with progress bar
                    with tqdm(tables, desc="Dropping tables", unit="table") as pbar:
                        for table in pbar:
                            pbar.set_postfix_str(f"Dropping {table}")

                            # Drop table and its sequences/indexes (CASCADE handles dependencies)
                            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

                            # Also drop any related sequences (for SERIAL columns)
                            cur.execute(
                                f"""
                                DROP SEQUENCE IF EXISTS {table}_id_seq CASCADE
                                """
                            )

                # Drop any remaining FTS tables (virtual tables in SQLite, but views/indexes in PostgreSQL)
                fts_patterns = ['search_%', '%_fts']
                for pattern in fts_patterns:
                    cur.execute(
                        f"""
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public' AND tablename LIKE '{pattern}'
                        """
                    )
                    fts_tables = [row[0] for row in cur.fetchall()]

                    for fts_table in fts_tables:
                        cur.execute(f"DROP TABLE IF EXISTS {fts_table} CASCADE")

                # Drop any custom functions we created
                cur.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")

                con.commit()

        print("✅ Database reset completed successfully!")
        print("🔧 ctgov schema dropped and all tables from public schema removed.")
        print("💡 You can now run the ingestion script to recreate the schema.")
        return True

    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False


def vacuum_database(config: DatabaseConfig) -> bool:
    """
    Run VACUUM ANALYZE to reclaim space and update statistics.
    Useful after large data operations.
    """
    try:
        print("🧹 Running VACUUM ANALYZE to optimize database...")

        # VACUUM needs to be run outside a transaction
        con = psycopg2.connect(**config.get_psycopg2_params())
        con.autocommit = True

        with con.cursor() as cur:
            cur.execute("VACUUM ANALYZE")

        con.close()
        print("✅ Database vacuum completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Database vacuum failed: {e}")
        return False


def show_database_info(config: DatabaseConfig) -> None:
    """Show comprehensive database information."""
    print("📊 Database Information")
    print("=" * 50)
    print(f"Host: {config.host}:{config.port}")
    print(f"Database: {config.database}")
    print(f"User: {config.user}")

    try:
        with get_connection(config) as con:
            with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Database size
                cur.execute(
                    f"""
                    SELECT pg_size_pretty(pg_database_size('{config.database}')) as size
                    """
                )
                db_size = cur.fetchone()['size']
                print(f"Database Size: {db_size}")

                # Table stats
                stats = get_table_stats(config)
                if stats:
                    print(f"\\n📋 Tables ({len(stats)}):")
                    total_rows = 0
                    for table, count in sorted(stats.items()):
                        if isinstance(count, int):
                            print(f"   - {table:<30} {count:>10,} rows")
                            total_rows += count
                        else:
                            print(f"   - {table:<30} {count}")

                    if total_rows > 0:
                        print(f"   {'Total':<30} {total_rows:>10,} rows")
                else:
                    print("\\n📋 No tables found.")

                # Connection info
                cur.execute("SELECT version()")
                version = cur.fetchone()['version']
                print(f"\\nPostgreSQL Version: {version.split(',')[0]}")

    except Exception as e:
        print(f"\\n❌ Could not get database info: {e}")


if __name__ == "__main__":
    # Command line interface for database operations
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python config.py info          - Show database information")
        print("  python config.py reset         - Reset database (with confirmation)")
        print("  python config.py reset --force - Reset database (no confirmation)")
        print("  python config.py vacuum        - Vacuum database")
        print("  python config.py tables        - List all tables")
        print("  python config.py create-schema - Create public schema if missing")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "info":
        show_database_info(DEFAULT_CONFIG)
    elif command == "reset":
        force = "--force" in sys.argv
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
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
'''

    config_file = Path("config.py")
    with open(config_file, "w") as f:
        f.write(config_content)

    print(f"✅ Configuration file created: {config_file.absolute()}")
    print("   Edit this file to customize your database settings")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 PostgreSQL setup complete!")
    print("\n📋 Next steps:")
    print("1. Run the full ingestion pipeline:")
    print("   python ingest_all_postgres.py")
    print("")
    print("2. Or use the CLI:")
    print("   ingest")
    print("")
    print("3. Or run individual data sources:")
    print("   python ingest_all_postgres.py --skip-openfda --skip-dailymed")
    print("")
    print("4. Database management:")
    print("   curebench db info                    # Show database info")
    print("   curebench db reset --confirm         # Reset database")
    print("   curebench db dump --file backup.sql  # Create backup")
    print("")
    print("5. Search across data sources:")
    print("   curebench search openfda 'aspirin' --type substance")
    print("   curebench search ctgov 'diabetes' --type condition")
    print("   curebench search drugcentral 'caffeine' --type name")
    print("")
    print("📊 Data sources available:")
    print("   ✅ OpenFDA drug labels (normalized)")
    print("   ✅ Orange Book therapeutic equivalence")
    print("   ✅ ClinicalTrials.gov trials")
    print("   ✅ DailyMed SPL labels")
    print("   ✅ DrugCentral molecular structures")
    print("   ✅ Cross-source drug mapping")
    print("")
    print("🔍 Advanced features:")
    print("   ✅ Full-text search with ranking")
    print("   ✅ Cross-source drug mapping")
    print("   ✅ Molecular structure queries")
    print("   ✅ Clinical trial analysis")
    print("   ✅ Therapeutic equivalence lookup")


def main():
    """Main setup function."""
    print("🚀 PostgreSQL Setup")
    print("=" * 40)

    if not check_postgresql_installed():
        sys.exit(1)

    if not setup_database():
        sys.exit(1)

    if not install_python_dependencies():
        sys.exit(1)

    if not create_config_file():
        sys.exit(1)

    print_next_steps()


if __name__ == "__main__":
    main()
