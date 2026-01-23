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

from dotenv import load_dotenv


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


def save_config_to_env(db_name: str, user_name: str, password: str, env_file: Path) -> None:
    """Save database configuration to .env file."""
    env_content = f"""# PostgreSQL Database Configuration
# Generated by setup_postgres.py

# Database connection details
DB_HOST=localhost
DB_PORT=5432
DB_NAME={db_name}
DB_USER={user_name}
DB_PASSWORD={password}
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"✅ Database configuration saved to: {env_file.absolute()}")


def get_env_config() -> tuple[str, str, str] | None:
    """Get database configuration from environment variables or .env file.

    Returns:
        Tuple of (db_name, user_name, password) if all are found, None otherwise.
    """
    # Try to load from .env file in the ingest directory
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Check if all required variables are set
    db_name = os.getenv("DB_NAME")
    user_name = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if db_name and user_name and password:
        return db_name, user_name, password

    return None


def prompt_for_config() -> tuple[str, str, str]:
    """Prompt user for database configuration.

    Returns:
        Tuple of (db_name, user_name, password)
    """
    print("\n🔧 Database Configuration")
    print("=" * 40)

    # Default values
    default_db = "database"
    default_user = "database_user"

    db_name = input(f"Database name [{default_db}]: ").strip() or default_db
    user_name = input(f"Database user [{default_user}]: ").strip() or default_user

    # Use getpass for password to hide input
    while True:
        password = getpass.getpass("Database password: ")
        if password:
            break
        print("❌ Password cannot be empty. Please try again.")

    return db_name, user_name, password


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

    # Check for existing configuration
    env_config = get_env_config()
    if env_config:
        db_name, user_name, password = env_config
        print(f"✅ Using existing configuration from .env file:")
        print(f"   Database: {db_name}")
        print(f"   User: {user_name}")
    else:
        print("\n⚙️  No .env configuration found. Let's set up your database:")
        db_name, user_name, password = prompt_for_config()

        # Ask if they want to save to .env
        env_file = Path(__file__).parent / ".env"
        if not env_file.exists():
            response = input("\n💾 Save configuration to .env file? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                save_config_to_env(db_name, user_name, password, env_file)

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


