# BiomedicalAgent Database Setup Guide

Complete guide for setting up the PostgreSQL database for BiomedicalAgent.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [PostgreSQL Installation](#postgresql-installation)
3. [Database Setup](#database-setup)
4. [pgvector Extension](#pgvector-extension)
5. [Data Ingestion](#data-ingestion)
6. [Post-Ingestion Steps](#post-ingestion-steps)
7. [Database Management](#database-management)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **PostgreSQL 14+** (recommended: 16)
- **Python 3.10+**
- **~50 GB disk space** for full database
- **8+ GB RAM** recommended for ingestion

### Python Dependencies

```bash
# Install the package in development mode
pip install -e .

# Verify dependencies
biomedagent-db verify-deps
```

---

## PostgreSQL Installation

### Ubuntu/Debian

```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify installation
pg_isready
```

### macOS (Homebrew)

```bash
brew install postgresql
brew services start postgresql
```

### Windows

Download from https://www.postgresql.org/download/windows/

---

## Database Setup

### Option 1: Automatic Setup (Recommended)

```bash
biomedagent-db setup_postgres
```

This will:
1. Create the database (`database`)
2. Create the user (`database_user`)
3. Set up permissions
4. Install pgvector extension
5. Generate config file

### Option 2: Manual Setup

```bash
# Connect as postgres superuser
sudo -u postgres psql

# Run these SQL commands:
```

```sql
-- Create user
CREATE USER database_user WITH PASSWORD 'database_password';

-- Create database
CREATE DATABASE database;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE database TO database_user;

-- Connect to database
\c database

-- Create pgvector extension (requires system-level install)
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO database_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO database_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO database_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO database_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO database_user;
GRANT CREATE ON SCHEMA public TO database_user;
```

### Configuration

Edit `src/bioagent/data/ingest/config.py` if you need different credentials:

```python
DEFAULT_CONFIG = DatabaseConfig(
    host="localhost",
    port=5432,
    database="database",
    user="database_user",
    password="database_password"
)
```

---

## pgvector Extension

The pgvector extension is required for semantic search features (DailyMed, OpenFDA embeddings).

### Step 1: Install at System Level

```bash
# Automatic installation (requires sudo)
biomedagent-db install-pgvector

# Or manual installation for PostgreSQL 16:
sudo apt-get install postgresql-16-pgvector

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Step 2: Create Extension in Database

```bash
# Create the extension (requires superuser)
biomedagent-db create-vector-ext
```

Or manually:
```bash
sudo -u postgres psql -d database -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

### Verify Installation

```sql
-- Check extension exists
SELECT * FROM pg_extension WHERE extname = 'vector';
```

---

## Data Ingestion

### Full Ingestion (All Sources)

```bash
# Run complete ingestion pipeline
biomedagent-db ingest

# With optimization after
biomedagent-db ingest --vacuum
```

**⏱️ Time Estimate:** 2-4 hours for full ingestion

### Selective Ingestion

```bash
# Skip specific sources
biomedagent-db ingest --skip-openfda --skip-dailymed

# Only specific sources (skip everything else)
biomedagent-db ingest \
    --skip-openfda \
    --skip-orange-book \
    --skip-dailymed \
    --skip-bindingdb \
    --skip-chembl \
    --skip-dm-target \
    --skip-drugcentral
# This would only run CT.gov
```

### Using Local Files (Skip Downloads)

If you've already downloaded the data files:

```bash
# Use existing OpenFDA files
biomedagent-db ingest --openfda-use-local-files

# Specify custom raw directory
biomedagent-db ingest --raw-dir /path/to/data
```

### Testing with Limited Data

```bash
# Limit records for testing
biomedagent-db ingest --n-max 1000

# Limit OpenFDA files
biomedagent-db ingest --openfda-files 5 --n-max 1000
```

### Ingestion Order

The pipeline runs in this order:

1. **OpenFDA** - Drug labels (foundation)
2. **Orange Book** - Therapeutic equivalence
3. **ClinicalTrials.gov** - Clinical trials
4. **DailyMed** - SPL labels
5. **BindingDB** - Molecular targets
6. **ChEMBL** - Biochemical annotations
7. **dm_target** - Target mapping (requires 5 & 6)
8. **DrugCentral** - Molecular structures

---

## Post-Ingestion Steps

### ClinicalTrials.gov RAG Corpus

The RAG (Retrieval-Augmented Generation) corpus enables semantic search over clinical trials. This is a **separate, long-running process**.

#### Option 1: During Ingestion (Not Recommended)

```bash
# WARNING: This adds several hours to ingestion!
biomedagent-db ingest --ctgov-populate-rag
```

#### Option 2: After Ingestion (Recommended)

```bash
# Step 1: Populate the RAG corpus (2-6 hours)
biomedagent-db ingest --ctgov-rag-corpus-only

# Step 2: Create keys and indexes (30-60 minutes)
biomedagent-db ingest --ctgov-rag-keys-only
```

#### RAG Options

```bash
# Use more buckets for larger databases (default: 16)
biomedagent-db ingest --ctgov-rag-corpus-only --ctgov-rag-buckets 32

# Check progress
biomedagent-db info
```

#### What RAG Creates

| Table/View | Description |
|------------|-------------|
| `rag_study_corpus` | Full study JSON for each NCT ID |
| `rag_study_keys` | Materialized view with searchable keys |
| Trigram indexes | Fast fuzzy text search |

### Force Recreate Tables

If you need to reimport specific data:

```bash
# Recreate BindingDB tables
biomedagent-db ingest --skip-all --bindingdb-force-recreate

# Recreate ChEMBL tables
biomedagent-db ingest --skip-all --chembl-force-recreate
```

### Database Optimization

```bash
# Run VACUUM ANALYZE after ingestion
biomedagent-db vacuum
```

---

## Database Management

### View Status

```bash
# Show database info and stats
biomedagent-db info

# List all tables
biomedagent-db tables

# Detailed sample data
biomedagent-db ingest --get-db-info --sample-size 5
```

### Backup & Restore

```bash
# Create backup
biomedagent-db ingest --dump-db backup.sql

# Restore from backup
biomedagent-db ingest --restore-db backup.sql
```

### Reset Database

```bash
# With confirmation prompt
biomedagent-db reset

# Force reset (no confirmation)
biomedagent-db reset --force
```

### Fix Permissions

```bash
# Fix user permissions on public schema
biomedagent-db fix-permissions

# Ensure public schema exists
biomedagent-db create-schema
```

---

## Troubleshooting

### "extension 'vector' is not available"

The pgvector system package is not installed:

```bash
# Install pgvector
biomedagent-db install-pgvector

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### "permission denied to create extension 'vector'"

You need superuser privileges to create extensions:

```bash
# Create extension as superuser
biomedagent-db create-vector-ext

# Or manually
sudo -u postgres psql -d database -c 'CREATE EXTENSION vector;'
```

### "relation does not exist"

Tables haven't been created. Run ingestion:

```bash
biomedagent-db ingest
```

### "permission denied for schema public"

Fix user permissions:

```bash
biomedagent-db fix-permissions
```

### Out of Memory During Ingestion

Use smaller batch sizes:

```bash
biomedagent-db ingest --bindingdb-batch-size 5000
```

### Slow OpenFDA Download

Use local files if already downloaded:

```bash
biomedagent-db ingest --openfda-use-local-files
```

### Import Errors

Reinstall the package:

```bash
# Clear cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Reinstall
pip install -e .
```

---

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `biomedagent-db setup_postgres` | Full automatic setup |
| `biomedagent-db install-pgvector` | Install pgvector system package |
| `biomedagent-db create-vector-ext` | Create vector extension in DB |
| `biomedagent-db ingest` | Run full ingestion |
| `biomedagent-db info` | Show database status |
| `biomedagent-db tables` | List all tables |
| `biomedagent-db vacuum` | Optimize database |
| `biomedagent-db reset --force` | Reset database |

### Ingestion Flags

| Flag | Description |
|------|-------------|
| `--skip-<source>` | Skip a data source |
| `--n-max N` | Limit records per source |
| `--openfda-use-local-files` | Use existing downloads |
| `--ctgov-rag-corpus-only` | Only populate RAG corpus |
| `--ctgov-rag-keys-only` | Only create RAG keys |
| `--vacuum` | Optimize after ingestion |
| `--dump-db FILE` | Create backup |
| `--restore-db FILE` | Restore backup |

### Data Sources

| Source | Prefix | Records | Time |
|--------|--------|---------|------|
| OpenFDA | `labels_meta`, `sections` | ~150K | 30-60 min |
| Orange Book | `orange_book_*` | ~35K | 1 min |
| ClinicalTrials.gov | `ctgov_*` | ~500K | 15-30 min |
| DailyMed | `dailymed_*` | ~150K | 30-60 min |
| BindingDB | `bindingdb_*` | ~2M | 30-60 min |
| ChEMBL | `chembl.*` | ~2M | 20-40 min |
| dm_target | `dm_target*` | ~3.5K | 5-10 min |
| DrugCentral | `drugcentral_*` | ~5K | 2-5 min |

---

## Complete Setup Workflow

```bash
# 1. Install PostgreSQL (if needed)
sudo apt-get install postgresql postgresql-contrib

# 2. Install Python package
pip install -e .

# 3. Install pgvector
biomedagent-db install-pgvector

# 4. Run setup
biomedagent-db setup_postgres

# 5. Create vector extension
biomedagent-db create-vector-ext

# 6. Run full ingestion
biomedagent-db ingest --vacuum

# 7. (Optional) Populate RAG corpus for CT.gov
biomedagent-db ingest --ctgov-rag-corpus-only
biomedagent-db ingest --ctgov-rag-keys-only

# 8. Verify
biomedagent-db info
```

🎉 **Your database is now ready!**

