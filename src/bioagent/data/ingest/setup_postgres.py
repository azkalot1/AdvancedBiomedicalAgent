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


def get_postgresql_version() -> str | None:
    """Get the PostgreSQL server version number."""
    try:
        result = subprocess.run(
            "psql --version", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            # Parse version from output like "psql (PostgreSQL) 16.4"
            import re
            match = re.search(r'(\d+)\.', result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def install_pgvector_extension() -> bool:
    """Install pgvector extension at the system level."""
    print("\n🔧 Installing pgvector extension...")
    
    pg_version = get_postgresql_version()
    if not pg_version:
        print("⚠️  Could not detect PostgreSQL version, assuming version 16")
        pg_version = "16"
    
    print(f"📦 Detected PostgreSQL version: {pg_version}")
    
    # Try to install pgvector package
    install_commands = [
        # Ubuntu/Debian with specific version
        f"sudo apt-get update && sudo apt-get install -y postgresql-{pg_version}-pgvector",
        # Fallback: try without version
        "sudo apt-get install -y postgresql-pgvector",
        # Alternative package name
        f"sudo apt-get install -y postgresql-{pg_version}-vector",
    ]
    
    for cmd in install_commands:
        print(f"   Trying: {cmd.split('&&')[-1].strip()}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ pgvector installed successfully!")
            return True
        elif "Unable to locate package" not in result.stderr:
            # Some other error, might still have worked
            if "is already the newest version" in result.stdout or "is already installed" in result.stdout:
                print("✅ pgvector is already installed!")
                return True
    
    # If apt failed, provide manual instructions
    print("\n❌ Automatic installation failed. Please install pgvector manually:")
    print(f"\n   For PostgreSQL {pg_version} on Ubuntu/Debian:")
    print(f"   sudo apt-get install postgresql-{pg_version}-pgvector")
    print("\n   For macOS with Homebrew:")
    print("   brew install pgvector")
    print("\n   For other systems, see: https://github.com/pgvector/pgvector#installation")
    print("\n   After installation, restart PostgreSQL:")
    print("   sudo systemctl restart postgresql")
    
    return False


def create_vector_extension_in_database(database: str = "database") -> bool:
    """Create the vector extension in a specific database using superuser privileges."""
    print(f"\n🔧 Creating vector extension in database '{database}'...")
    
    # Run as postgres superuser
    cmd = f"sudo -u postgres psql -d {database} -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Vector extension created in database '{database}'")
        return True
    else:
        print(f"❌ Failed to create vector extension: {result.stderr}")
        
        # Check if it's because extension is not installed at system level
        if "could not open extension control file" in result.stderr.lower():
            print("\n💡 The pgvector package is not installed at the system level.")
            print("   Run: biomedagent-db install-pgvector")
        
        return False


def setup_database():
    """Set up the database and user."""
    print("\n🏗️  Setting up database...")

    # First, ensure pgvector is installed at system level
    if not install_pgvector_extension():
        print("\n⚠️  pgvector installation failed or needs manual intervention.")
        print("   You can continue, but vector-based features (semantic search) won't work.")
        response = input("   Continue anyway? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            return False

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

-- Grant privileges on database
GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user_name};

-- Connect to the database and set up permissions
\\c {db_name}

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant all privileges on public schema
GRANT ALL ON SCHEMA public TO {user_name};

-- Grant privileges on all existing tables in public schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {user_name};

-- Grant privileges on all existing sequences in public schema
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {user_name};

-- Grant default privileges for future tables and sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {user_name};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {user_name};

-- Ensure user can create tables in public schema
GRANT CREATE ON SCHEMA public TO {user_name};
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


def verify_python_dependencies():
    """Verify that required Python packages are installed."""
    print("\n📦 Verifying Python dependencies...")

    # Core packages required for database operations
    required_packages = [
        "psycopg2",  # PostgreSQL adapter (psycopg2-binary)
        "tqdm",  # Progress bars
        "requests",  # HTTP requests
        "bs4",  # BeautifulSoup4 (HTML parsing)
        "lxml",  # XML/HTML parser
        "dateutil",  # python-dateutil (Date parsing)
        "typer",  # CLI framework
        "numpy",  # Numerical computing
        "sentence_transformers",  # Semantic embeddings
        "pgvector",  # PostgreSQL vector extension
        "ijson",  # JSON streaming parser
    ]

    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing dependencies: {', '.join(missing_packages)}")
        print("\n🔧 To install missing dependencies, run:")
        print("   pip install -e .")
        print("   # or")
        print("   pip install -e .[dev]  # for development dependencies")
        return False

    print("✅ All required Python dependencies are installed")
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


def fix_user_permissions(config: DatabaseConfig) -> bool:
    """
    Fix user permissions for the public schema.
    This function can be used to fix permission issues on existing databases.

    Args:
        config: Database configuration

    Returns:
        True if permissions were fixed successfully, False if failed
    """
    try:
        print(f"🔧 Fixing permissions for user '{config.user}' on database '{config.database}'...")
        
        with get_connection(config) as con:
            with con.cursor() as cur:
                # Grant all privileges on public schema
                cur.execute(f"GRANT ALL ON SCHEMA public TO {config.user}")
                print("✅ Granted schema permissions")

                # Grant privileges on all existing tables in public schema
                cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {config.user}")
                print("✅ Granted table permissions")

                # Grant privileges on all existing sequences in public schema
                cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {config.user}")
                print("✅ Granted sequence permissions")

                # Grant default privileges for future tables and sequences
                cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {config.user}")
                cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {config.user}")
                print("✅ Set default privileges for future objects")

                # Ensure user can create tables in public schema
                cur.execute(f"GRANT CREATE ON SCHEMA public TO {config.user}")
                print("✅ Granted CREATE permission on schema")

                con.commit()

        print("✅ All permissions fixed successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to fix user permissions: {e}")
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


def cli_main():
    """Command line interface for database operations."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  biomedagent-db setup_postgres           - Run full PostgreSQL setup")
        print("  biomedagent-db info                     - Show database information")
        print("  biomedagent-db reset                    - Reset database (with confirmation)")
        print("  biomedagent-db reset --force            - Reset database (no confirmation)")
        print("  biomedagent-db vacuum                   - Vacuum database")
        print("  biomedagent-db tables                   - List all tables")
        print("  biomedagent-db create-schema            - Create public schema if missing")
        print("  biomedagent-db fix-permissions          - Fix user permissions on public schema")
        print("  biomedagent-db verify-deps              - Verify Python dependencies are installed")
        print("  biomedagent-db install-pgvector         - Install pgvector extension (requires sudo)")
        print("  biomedagent-db create-vector-ext        - Create vector extension in database (requires sudo)")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "setup_postgres":
        main()
    elif command == "info":
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
    elif command == "fix-permissions":
        fix_user_permissions(DEFAULT_CONFIG)
    elif command == "verify-deps":
        verify_python_dependencies()
    elif command == "install-pgvector":
        install_pgvector_extension()
    elif command == "create-vector-ext":
        create_vector_extension_in_database()
    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands:")
        print("  setup_postgres, info, reset, vacuum, tables, create-schema, fix-permissions, verify-deps, install-pgvector, create-vector-ext")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
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
    print("1. Install project dependencies (if not already done):")
    print("   pip install -e .                    # Install project in development mode")
    print("   # or")
    print("   pip install -e .[dev]               # Include development dependencies")
    print("")
    print("2. Run the full ingestion pipeline:")
    print("   python ingest_all_postgres.py")
    print("")
    print("3. Or use the CLI:")
    print("   ingest")
    print("")
    print("4. Or run individual data sources:")
    print("   python ingest_all_postgres.py --skip-openfda --skip-dailymed")
    print("")
    print("5. Database management:")
    print("   biomedagent-db info                  # Show database info")
    print("   biomedagent-db reset                 # Reset database")
    print("   biomedagent-db fix-permissions       # Fix user permissions")
    print("   biomedagent-db verify-deps           # Verify dependencies")
    print("   biomedagent-db vacuum                # Optimize database")
    print("")
    print("6. Search across data sources:")
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

    if not verify_python_dependencies():
        sys.exit(1)

    if not create_config_file():
        sys.exit(1)

    print_next_steps()


if __name__ == "__main__":
    main()