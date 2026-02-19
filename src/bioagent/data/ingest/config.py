#!/usr/bin/env python3
"""
Database configuration for data ingestion.
Modify these settings to match your PostgreSQL setup.
"""
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

DATA_POSTGRES_URI_ENV_KEYS = ("DATA_POSTGRES_URI", "DATA_POSTGRES_URL", "DATA_DATABASE_URL")


def _load_repo_env() -> None:
    """Load nearest .env from this file's parent tree (repo root preferred)."""
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    load_dotenv(override=False)


# Load environment variables from .env file at module import.
_load_repo_env()


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
        safe_user = quote(self.user, safe="")
        safe_password = quote(self.password, safe="")
        return f"postgresql://{safe_user}:{safe_password}@{self.host}:{self.port}/{self.database}"

    def get_psycopg2_params(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "database": self.database, "user": self.user, "password": self.password}


def _legacy_uri_from_parts() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "database")
    user = quote(os.getenv("DB_USER", "database_user"), safe="")
    password = quote(os.getenv("DB_PASSWORD", "database_password"), safe="")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def resolve_data_postgres_uri() -> str:
    """Resolve ingestion DB URI from env, preferring DATA_POSTGRES_URI."""
    for key in DATA_POSTGRES_URI_ENV_KEYS:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def parse_data_postgres_uri(uri: str) -> DatabaseConfig:
    """Parse a PostgreSQL URI into DatabaseConfig."""
    if not uri:
        raise ValueError("Empty DATA_POSTGRES_URI")
    parsed = urlparse(uri)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("Unsupported DB URI scheme. Use postgres:// or postgresql://")
    if not parsed.hostname:
        raise ValueError("DATA_POSTGRES_URI missing host")
    if parsed.username is None:
        raise ValueError("DATA_POSTGRES_URI missing username")
    if parsed.password is None:
        raise ValueError("DATA_POSTGRES_URI missing password")
    if not parsed.path or parsed.path == "/":
        raise ValueError("DATA_POSTGRES_URI missing database name")

    host = parsed.hostname
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")
    user = unquote(parsed.username)
    password = unquote(parsed.password)

    return DatabaseConfig(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password=password,
    )


def load_config_from_env() -> DatabaseConfig:
    """
    Load database configuration from environment variables.

    Primary source: DATA_POSTGRES_URI (or legacy DATA_DATABASE_URL).
    Legacy fallback: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.

    Returns:
        DatabaseConfig instance with environment variable values
    """
    uri = resolve_data_postgres_uri()
    if uri:
        return parse_data_postgres_uri(uri)
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "database"),
        user=os.getenv("DB_USER", "database_user"),
        password=os.getenv("DB_PASSWORD", "database_password")
    )


def _sync_legacy_db_env(config: DatabaseConfig) -> None:
    """
    Populate legacy DB_* keys for modules that still reference them.
    DATA_POSTGRES_URI remains the source of truth.
    """
    os.environ.setdefault("DB_HOST", config.host)
    os.environ.setdefault("DB_PORT", str(config.port))
    os.environ.setdefault("DB_NAME", config.database)
    os.environ.setdefault("DB_USER", config.user)
    os.environ.setdefault("DB_PASSWORD", config.password)


# Default database configuration - reads from DATA_POSTGRES_URI with legacy fallbacks.
DEFAULT_CONFIG = load_config_from_env()
_sync_legacy_db_env(DEFAULT_CONFIG)


def get_connection(config: DatabaseConfig) -> psycopg2.extensions.connection:
    """Get a PostgreSQL connection."""
    return psycopg2.connect(**config.get_psycopg2_params())


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
                print("\n📊 Current tables and row counts:")
                for table, count in stats.items():
                    print(f"   - {table}: {count:,} rows" if isinstance(count, int) else f"   - {table}: {count}")
            else:
                print("\n📊 No tables found in database.")
        except Exception as e:
            print(f"\n❌ Could not get table stats: {e}")

        response = input("\n❓ Are you sure you want to reset the database? (yes/no): ").lower().strip()
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
                    from tqdm import tqdm

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
                    print(f"\n📋 Tables ({len(stats)}):")
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
                    print("\n📋 No tables found.")

                # Connection info
                cur.execute("SELECT version()")
                version = cur.fetchone()['version']
                print(f"\nPostgreSQL Version: {version.split(',')[0]}")

    except Exception as e:
        print(f"\n❌ Could not get database info: {e}")


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
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
