# Advanced Biomedical Agent

Biomedical research agent that integrates OpenFDA, DailyMed, ClinicalTrials.gov, ChEMBL, BindingDB, DrugCentral, Orange Book and related sources into a unified PostgreSQL database, with LLM-friendly search tools and a **chat interface**.

## Quick Start

**Prerequisites:** PostgreSQL 14+, Python 3.12+, ~50 GB disk, 8+ GB RAM for full ingestion.

### 1. Install PostgreSQL

- **Ubuntu/Debian:** `sudo apt-get install postgresql postgresql-contrib`
- **macOS:** `brew install postgresql`
- **Windows:** [Download from postgresql.org](https://www.postgresql.org/download/)

### 2. After that

```bash
pip install -e ./
biomedagent-db verify-deps
biomedagent-db setup_postgres
biomedagent-db ingest
biomedagent-db ingest --ctgov-rag-corpus-only
biomedagent-db ingest --ctgov-rag-keys-only
biomedagent-db ingest --generate-search
```

**Data available.** The database unifies clinical trials (ClinicalTrials.gov), molecular structures and biotherapeutics (ChEMBL, DrugCentral, BindingDB), biological targets, drug labels (OpenFDA, DailyMed), and regulatory data (Orange Book), with mappings from trials and products to molecules. Full schema: [docs/DATABASE_SCHEMA.md](docs/DATABASE_SCHEMA.md).

**Tools available.** Retrieval tools cover clinical trial search, drug labels, molecule–trial connectivity, adverse events, outcomes, Orange Book, cross-database lookup, biotherapeutic sequence search, and target/drug pharmacology; agent tools wrap these with LLM-friendly formatting. Full reference: [docs/tools.md](docs/tools.md).

For pipeline order and script roles, see [docs/INGESTION.md](docs/INGESTION.md).
