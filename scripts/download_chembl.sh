#!/bin/bash
# Download and install ChEMBL database

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data/chembl"

echo "========================================"
echo "ChEMBL Database Download & Installation"
echo "========================================"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Check if already downloaded
if [ -f "chembl_34_postgresql.dmp" ]; then
    echo "✓ ChEMBL dump already exists: chembl_34_postgresql.dmp"
    read -p "Download again? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
    else
        rm -f chembl_34_postgresql.tar.gz chembl_34_postgresql.dmp
    fi
fi

# Download if needed
if [ ! -f "chembl_34_postgresql.dmp" ]; then
    echo "Downloading ChEMBL 34 PostgreSQL dump..."
    echo "URL: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_postgresql.tar.gz"
    echo "Size: ~15 GB (this will take 30-60 minutes)"
    echo ""
    
    wget -c https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_postgresql.tar.gz
    
    echo ""
    echo "✓ Download complete!"
    echo ""
    echo "Extracting archive..."
    tar -xzf chembl_34_postgresql.tar.gz
    
    echo "✓ Extraction complete!"
    echo ""
fi

# Check disk space
REQUIRED_SPACE_GB=100
AVAILABLE_SPACE=$(df -BG "$DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')

echo "Disk space check:"
echo "  Required: ${REQUIRED_SPACE_GB} GB"
echo "  Available: ${AVAILABLE_SPACE} GB"

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE_GB" ]; then
    echo ""
    echo "⚠️  WARNING: Insufficient disk space!"
    echo "   ChEMBL requires ~80 GB after loading plus temp space."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "Database Creation"
echo "========================================"
echo ""

# Check if database exists
DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='chembl_34'" 2>/dev/null || echo "0")

if [ "$DB_EXISTS" = "1" ]; then
    echo "⚠️  Database 'chembl_34' already exists!"
    read -p "Drop and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Dropping existing database..."
        sudo -u postgres dropdb chembl_34
        echo "✓ Database dropped"
    else
        echo "Keeping existing database. Skipping restore."
        exit 0
    fi
fi

echo "Creating database 'chembl_34'..."
sudo -u postgres createdb chembl_34
echo "✓ Database created"

echo ""
echo "========================================"
echo "Restoring ChEMBL Data"
echo "========================================"
echo ""
echo "This will take 30-60 minutes..."
echo "Using 4 parallel jobs for faster restore."
echo ""

# Restore with progress
sudo -u postgres pg_restore \
    --dbname=chembl_34 \
    --jobs=4 \
    --no-owner \
    --no-acl \
    --verbose \
    chembl_34_postgresql.dmp 2>&1 | grep -E "processing|restoring|creating" || true

echo ""
echo "✓ Restore complete!"

echo ""
echo "========================================"
echo "Granting Permissions"
echo "========================================"
echo ""

# Grant permissions to database_user
sudo -u postgres psql -d chembl_34 <<EOF
GRANT CONNECT ON DATABASE chembl_34 TO database_user;
GRANT USAGE ON SCHEMA public TO database_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO database_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO database_user;

-- Grant for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO database_user;
EOF

echo "✓ Permissions granted to database_user"

echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo ""

# Verify installation
echo "Checking table counts..."
sudo -u postgres psql -d chembl_34 -c "
SELECT 
    schemaname,
    tablename,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE n_live_tup > 0
ORDER BY n_live_tup DESC
LIMIT 15;" 2>&1

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "ChEMBL 34 is now installed in database: chembl_34"
echo ""
echo "Key tables:"
echo "  - molecule_dictionary: Compounds (~2.4M)"
echo "  - activities: Bioactivity data (~21M)"
echo "  - target_dictionary: Protein targets (~14K)"
echo "  - compound_structures: SMILES, InChI"
echo ""
echo "Next steps:"
echo "  1. Create mapping table in your main database:"
echo "     psql -U database_user -d database -f scripts/create_chembl_mapping_tables.sql"
echo ""
echo "  2. Map interventions to ChEMBL molecules:"
echo "     python src/bioagent/data/ingest/map_chembl.py"
echo ""
echo "  3. Query drug-target binding data:"
echo "     See examples in CHEMBL_INGESTION_GUIDE.md"
echo ""




