#!/usr/bin/env python3
"""
PostgreSQL setup script for data ingestion.

This script helps set up the PostgreSQL database and user for the data pipeline.
"""

import getpass
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from dotenv import load_dotenv

# Keys we read/write for ingestion DB config
DATA_DB_ENV_KEY = "DATA_POSTGRES_URI"
DATA_DB_ENV_ALIASES = (DATA_DB_ENV_KEY, "DATA_POSTGRES_URL", "DATA_DATABASE_URL")
LEGACY_DB_ENV_KEYS = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
ALL_DB_ENV_KEYS = (*DATA_DB_ENV_ALIASES, *LEGACY_DB_ENV_KEYS)


def get_repo_root() -> Path:
    """Return the repository root directory (directory containing pyproject.toml)."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume repo root is 4 levels up from this file (src/bioagent/data/ingest/)
    return path.parents[3]


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


def install_rdkit_extension() -> bool:
    """Install RDKit PostgreSQL extension at the system level."""
    print("\n🔧 Installing RDKit PostgreSQL extension...")
    
    pg_version = get_postgresql_version()
    if not pg_version:
        print("⚠️  Could not detect PostgreSQL version, assuming version 16")
        pg_version = "16"
    
    print(f"📦 Detected PostgreSQL version: {pg_version}")
    
    # Try to install RDKit package
    install_commands = [
        # Ubuntu/Debian with specific version
        f"sudo apt-get update && sudo apt-get install -y postgresql-{pg_version}-rdkit",
        # Fallback: try without version
        "sudo apt-get install -y postgresql-rdkit",
        # Alternative package name
        f"sudo apt-get install -y postgresql-{pg_version}-rdkit-python",
    ]
    
    for cmd in install_commands:
        print(f"   Trying: {cmd.split('&&')[-1].strip()}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ RDKit installed successfully!")
            return True
        elif "Unable to locate package" not in result.stderr:
            # Some other error, might still have worked
            if "is already the newest version" in result.stdout or "is already installed" in result.stdout:
                print("✅ RDKit is already installed!")
                return True
    
    # If apt failed, provide manual instructions
    print("\n❌ Automatic installation failed. Please install RDKit manually:")
    print(f"\n   For PostgreSQL {pg_version} on Ubuntu/Debian:")
    print(f"   sudo apt-get install postgresql-{pg_version}-rdkit")
    print("\n   For macOS with Homebrew:")
    print("   brew install rdkit")
    print("\n   For other systems, see: https://github.com/rdkit/rdkit")
    print("\n   Note: RDKit PostgreSQL extension may need to be compiled from source.")
    print("   See: https://github.com/rdkit/rdkit/tree/master/Contrib/rdkit_pgsql")
    print("\n   After installation, restart PostgreSQL:")
    print("   sudo systemctl restart postgresql")
    
    return False


def create_rdkit_extension_in_database(database: str = "database") -> bool:
    """Create the RDKit extension in a specific database using superuser privileges."""
    print(f"\n🔧 Creating RDKit extension in database '{database}'...")
    
    # Run as postgres superuser
    cmd = f"sudo -u postgres psql -d {database} -c 'CREATE EXTENSION IF NOT EXISTS rdkit;'"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ RDKit extension created in database '{database}'")
        return True
    else:
        print(f"❌ Failed to create RDKit extension: {result.stderr}")
        
        # Check if it's because extension is not installed at system level
        if "could not open extension control file" in result.stderr.lower():
            print("\n💡 The RDKit package is not installed at the system level.")
            print("   Run: biomedagent-db install-rdkit")
        
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


def _parse_env_lines(path: Path) -> list[tuple[str, str]]:
    """Read .env file into list of (key, value) for non-empty KEY=VALUE lines. Comments and empty lines omitted."""
    if not path.exists():
        return []
    pairs = []
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            pairs.append((key.strip(), value.strip()))
    return pairs


def _build_data_postgres_uri(
    db_host: str,
    db_port: str,
    db_name: str,
    user_name: str,
    password: str,
) -> str:
    safe_user = quote(user_name, safe="")
    safe_password = quote(password, safe="")
    return f"postgresql://{safe_user}:{safe_password}@{db_host}:{db_port}/{db_name}"


def _parse_data_postgres_uri(uri: str) -> tuple[str, str, str, str, str]:
    parsed = urlparse(uri)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("Unsupported DB URI scheme. Use postgres:// or postgresql://")

    db_host = parsed.hostname or "localhost"
    db_port = str(parsed.port or 5432)
    db_name = (parsed.path or "/database").lstrip("/") or "database"
    user_name = unquote(parsed.username or "database_user")
    password = unquote(parsed.password or "database_password")
    return db_host, db_port, db_name, user_name, password


def _env_db_updates(
    db_host: str, db_port: str, db_name: str, user_name: str, password: str
) -> dict[str, str]:
    """Return dict of DATA_* keys to values for writing to .env."""
    return {
        DATA_DB_ENV_KEY: _build_data_postgres_uri(db_host, db_port, db_name, user_name, password),
    }


def save_config_to_env(
    db_host: str,
    db_port: str,
    db_name: str,
    user_name: str,
    password: str,
    env_file: Path,
) -> None:
    """Create or update .env file with DATA_POSTGRES_URI. Other keys/comments are preserved."""
    updates = _env_db_updates(db_host, db_port, db_name, user_name, password)
    existing = {k: v for k, v in _parse_env_lines(env_file)}

    # Merge: existing keys stay unless we're updating managed ingestion DB keys.
    for k, v in updates.items():
        existing[k] = v

    # Rebuild file: keep original order/comments for non-DB lines, then ensure normalized data DB key.
    lines_out = []
    had_db_keys = False
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines_out.append(line)
                continue
            if "=" in line:
                key = line.partition("=")[0].strip()
                if key in ALL_DB_ENV_KEYS:
                    had_db_keys = True
                    continue  # drop old DB config lines; we'll append normalized key
            lines_out.append(line)

    # Append DB section (with comment if we're adding for first time)
    if not had_db_keys:
        lines_out.append("")
        lines_out.append("# Data ingestion PostgreSQL URI (setup_postgres.py)")
    lines_out.append(f"{DATA_DB_ENV_KEY}={existing[DATA_DB_ENV_KEY]}")

    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines_out) + "\n")
    print(f"✅ Database configuration saved to: {env_file.absolute()}")


def get_env_config() -> tuple[str, str, str, str, str] | None:
    """Get database configuration from environment variables or repo-root .env file.

    Returns:
        Tuple of (db_host, db_port, db_name, user_name, password) if all are found, None otherwise.
    """
    env_file = get_repo_root() / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    data_uri = ""
    for key in DATA_DB_ENV_ALIASES:
        value = (os.getenv(key) or "").strip()
        if value:
            data_uri = value
            break
    if data_uri:
        try:
            return _parse_data_postgres_uri(data_uri)
        except ValueError:
            print(f"⚠️  Invalid {DATA_DB_ENV_KEY} value in .env. Falling back to legacy DB_* vars.")

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    user_name = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if db_host and db_port and db_name and user_name and password:
        return db_host, db_port, db_name, user_name, password

    return None


def prompt_for_config() -> tuple[str, str, str, str, str]:
    """Prompt user for database configuration.

    Returns:
        Tuple of (db_host, db_port, db_name, user_name, password)
    """
    print("\n🔧 Database Configuration")
    print("=" * 40)

    default_host = "localhost"
    default_port = "5432"
    default_db = "database"
    default_user = "database_user"

    db_host = input(f"DB host [{default_host}]: ").strip() or default_host
    db_port = input(f"DB port [{default_port}]: ").strip() or default_port
    db_name = input(f"Database name [{default_db}]: ").strip() or default_db
    user_name = input(f"Database user [{default_user}]: ").strip() or default_user

    # Use getpass for password to hide input
    while True:
        password = getpass.getpass("Database password: ")
        if password:
            break
        print("❌ Password cannot be empty. Please try again.")

    return db_host, db_port, db_name, user_name, password


def setup_database():
    """Set up the database and user."""
    print("\n🏗️  Setting up database...")

    # First, ensure pgvector and rdkit are installed at system level
    if not install_pgvector_extension():
        print("\n⚠️  pgvector installation failed or needs manual intervention.")
        print("   You can continue, but vector-based features (semantic search) won't work.")
        response = input("   Continue anyway? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            return False

    if not install_rdkit_extension():
        print("\n⚠️  RDKit installation failed or needs manual intervention.")
        print("   You can continue, but RDKit-based features (e.g. structure search) won't work.")
        response = input("   Continue anyway? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            return False

    # Check for existing configuration in repo-root .env
    env_file = get_repo_root() / ".env"
    env_config = get_env_config()
    if env_config:
        db_host, db_port, db_name, user_name, password = env_config
        print(f"✅ Using existing configuration from {env_file}:")
        print(f"   Host: {db_host}, Port: {db_port}")
        print(f"   Database: {db_name}")
        print(f"   User: {user_name}")
        if not any((os.getenv(key) or "").strip() for key in DATA_DB_ENV_ALIASES):
            save_config_to_env(db_host, db_port, db_name, user_name, password, env_file)
            print(f"ℹ️  Migrated legacy DB_* settings to {DATA_DB_ENV_KEY} in {env_file}.")
    else:
        print(f"\n⚙️  No .env in repo root or {DATA_DB_ENV_KEY} missing. Let's set up your database:")
        db_host, db_port, db_name, user_name, password = prompt_for_config()
        save_config_to_env(db_host, db_port, db_name, user_name, password, env_file)

    # Create database and user (names parsed from DATA_POSTGRES_URI)
    # Script runs as postgres superuser; "You are now connected... as user postgres" is expected.
    sql_commands = f"""
-- Create user if not exists (user_name parsed from DATA_POSTGRES_URI)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{user_name}') THEN
        CREATE USER {user_name} WITH PASSWORD '{password}';
    END IF;
END
$$;

-- Create database if not exists (db_name parsed from DATA_POSTGRES_URI)
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

    # Execute SQL as postgres user (session runs as postgres; we create user_name and db_name from .env)
    if run_command(f"sudo -u postgres psql -f {sql_file}", "Creating database and user"):
        # Create vector and rdkit extensions in the database (idempotent)
        create_vector_extension_in_database(db_name)
        create_rdkit_extension_in_database(db_name)

        # Verify that the role and database from .env actually exist
        verify_cmd = f"sudo -u postgres psql -t -c \"SELECT 1 FROM pg_roles WHERE rolname = '{user_name}'\" -c \"SELECT 1 FROM pg_database WHERE datname = '{db_name}'\""
        result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and "1" in result.stdout:
            print(f"✅ Verified: role '{user_name}' and database '{db_name}' exist (from .env)")
        else:
            print(f"⚠️  Could not verify role/database; continuing anyway.")

        print(f"✅ Database '{db_name}' and user '{user_name}' created successfully")

        # Test connection as the app user (from .env)
        test_cmd = f"PGPASSWORD={password} psql -h {db_host} -p {db_port} -U {user_name} -d {db_name} -c 'SELECT version();'"
        if run_command(test_cmd, "Testing database connection"):
            print("✅ Database connection test successful (logged in as .env user)")
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


def main() -> None:
    """Entry point for CLI and direct execution: check PostgreSQL, then run database setup."""
    if not check_postgresql_installed():
        sys.exit(1)
    if not setup_database():
        sys.exit(1)


if __name__ == "__main__":
    main()
