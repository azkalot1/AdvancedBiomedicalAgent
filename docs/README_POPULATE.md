# BiomedicalAgent Data Population Guide

## Overview

The BiomedicalAgent PostgreSQL database integrates **8 data sources** into a unified schema for biomedical queries. All data is stored in the `public` schema with source-specific prefixes.

### Data Sources

| # | Source | Tables Prefix | Description |
|---|--------|---------------|-------------|
| 1 | **OpenFDA** | `labels_meta`, `sections`, `mapping_*` | Drug labels with normalized structure |
| 2 | **Orange Book** | `orange_book_*` | FDA therapeutic equivalence data |
| 3 | **ClinicalTrials.gov** | `ctgov_*` | Clinical trials from CTTI AACT dump |
| 4 | **DailyMed** | `dailymed_*` | SPL drug labels with semantic search |
| 5 | **BindingDB** | `bindingdb_*` | Molecular targets & binding affinity |
| 6 | **ChEMBL** | `chembl_*` (in chembl schema) | Biochemical annotations |
| 7 | **dm_target** | `dm_target*` | Canonical target mapping (ChEMBL + BindingDB) |
| 8 | **DrugCentral** | `drugcentral_*` | Molecular structures & chemical data |

---

## Quick Start

### Full Ingestion (All Sources)

```bash
cd /home/sergey/projects/AdvancedBiomedicalAgent

# Run full ingestion pipeline
biomedagent-db ingest

# Or with options
biomedagent-db ingest --vacuum  # Optimize after ingestion
```

### Selective Ingestion

```bash
# Skip specific sources
biomedagent-db ingest --skip-openfda --skip-ctgov

# Use local files (skip download)
biomedagent-db ingest --openfda-use-local-files

# Limit records for testing
biomedagent-db ingest --n-max 1000
```

---

## Data Source Details

### 1. OpenFDA Drug Labels

**Script:** `build_openfda.py`  
**Tables Created:**
- `set_ids` - Reference table for all set_ids
- `section_names` - Canonical section names with embeddings
- `labels_meta` - Drug label metadata
- `sections` - Normalized section content with tsvector
- `sections_per_set_id` - Set ID to section mapping
- `mapping_*` - Various identifier mappings (NDC, RxCUI, etc.)

**Key Features:**
- Semantic search via section name embeddings (384-dim vectors)
- Full-text search on section content
- Normalized structure for LLM parsing

**Options:**
```bash
biomedagent-db ingest --openfda-files 10      # Limit files
biomedagent-db ingest --openfda-use-local-files  # Use existing downloads
```

---

### 2. Orange Book Products

**Script:** `build_orange_book.py`  
**Tables Created:**
- `orange_book_products` - FDA approved drug products

**Key Features:**
- Therapeutic equivalence codes (TE codes)
- Reference Listed Drugs (RLD)
- Application numbers for FDA approval lookup

**Data:**
- ~35,000 product records
- Updated monthly from FDA

---

### 3. ClinicalTrials.gov (CTTI AACT)

**Script:** `build_ctgov.py`  
**Tables Created:**
- `ctgov_studies` - Main study records
- `ctgov_interventions` - Drug/treatment interventions
- `ctgov_conditions` - Medical conditions studied
- `ctgov_outcomes` - Study outcome measures
- `ctgov_*` - 40+ additional tables
- `rag_study_corpus` - RAG-ready study JSON
- `rag_study_keys` - Materialized view for RAG lookup

**Key Features:**
- Full CTTI AACT PostgreSQL dump (~500K studies)
- Full-text search indexes on key tables
- RAG functionality for study retrieval

**Options:**
```bash
# Populate RAG corpus (takes hours!)
biomedagent-db ingest --ctgov-populate-rag
biomedagent-db ingest --ctgov-rag-buckets 32

# Standalone RAG population
biomedagent-db ingest --ctgov-rag-corpus-only
biomedagent-db ingest --ctgov-rag-keys-only
```

---

### 4. DailyMed SPL Labels

**Script:** `build_dailymed.py`  
**Tables Created:**
- `dailymed_products` - Drug product info
- `dailymed_sections` - Label sections with tsvector
- `dailymed_section_names` - Section names with embeddings

**Key Features:**
- Semantic search via embeddings
- Canonical title normalization
- Full-text search on content

**Data:**
- ~150,000 drug labels
- Downloaded from DailyMed FTP

---

### 5. BindingDB Molecular Targets

**Script:** `build_bindingdb.py`  
**Tables Created:**
- `bindingdb_molecules` - Molecular structures & identifiers
- `bindingdb_targets` - Protein targets
- `bindingdb_activities` - Binding measurements (Ki, IC50, Kd, EC50)

**Views:**
- `bindingdb_summary` - Joined molecule-target-activity view
- `bindingdb_human_targets` - Human targets only
- `bindingdb_stats` - Coverage statistics

**Key Features:**
- InChIKey, SMILES, PubChem CID, ChEMBL ID
- pChEMBL values for potency comparison
- Human-only filtering (default)

**Options:**
```bash
biomedagent-db ingest --bindingdb-all-organisms    # Include all organisms
biomedagent-db ingest --bindingdb-batch-size 5000  # Lower memory usage
biomedagent-db ingest --bindingdb-force-recreate   # Drop and recreate
```

--- 
### 6. ChEMBL Biochemical Annotations

**Script:** `build_chembl.py`  
**Schema:** Tables restored to `chembl` schema, views accessible from public

**Key Tables:**
- `molecule_dictionary` - Molecule definitions
- `compound_structures` - Chemical structures
- `molecule_synonyms` - Alternative names
- `target_dictionary` - Protein targets
- `activities` - Bioactivity measurements
- `component_sequences` - Target protein sequences

**Key Features:**
- Full ChEMBL PostgreSQL dump (~2M molecules)
- Pre-built views for common queries
- Cross-references to UniProt, PubChem

**Options:**
```bash
biomedagent-db ingest --chembl-force-recreate  # Drop and recreate
```

---

### 7. dm_target Canonical Target Mapping

**Script:** `create_and_populate_dm_target.py`  
**Tables Created:**
- `dm_target` - Unified target table
- `dm_target_gene_synonyms` - Gene symbol aliases
- `dm_target_uniprot_mappings` - UniProt accession mappings

**Population Phases:**

| Phase | Description | Records |
|-------|-------------|---------|
| 1 | ChEMBL base (human SINGLE_PROTEIN) | ~2,000 |
| 2A | New BindingDB targets | ~400 |
| 2B | Consensus marking | ~1,200 |
| 3 | Gene synonym consolidation | ~8,000 |
| 4 | UniProt accession mapping | ~3,500 |

**Key Features:**
- Unified target representation across ChEMBL + BindingDB
- Confidence scoring (0-10)
- Source tracking (CHEMBL, BINDINGDB, CONSENSUS)

**Options:**
```bash
biomedagent-db ingest --skip-dm-target  # Skip dm_target population
```

**Prerequisites:**
- ChEMBL must be ingested first
- BindingDB must be ingested first

---

### 8. DrugCentral Molecular Structures

**Script:** `build_drugcentral.py`  
**Tables Created:**
- `drugcentral_drugs` - Drug structures and identifiers

**Key Features:**
- SMILES, InChIKey, molecular formula
- CAS Registry Numbers
- Full-text search on names and synonyms
- RDKit computed properties (if available)

**Data:**
- ~5,000 approved drug structures
- From DrugCentral 2023 release

---

## Database Management

### Check Database Status

```bash
biomedagent-db info              # Show database info
biomedagent-db tables            # List all tables with row counts
biomedagent-db ingest --get-db-info --sample-size 5  # Sample data
```

### Reset Database

```bash
biomedagent-db reset             # With confirmation
biomedagent-db reset --force     # No confirmation
```

### Backup & Restore

```bash
biomedagent-db ingest --dump-db backup.sql     # Create backup
biomedagent-db ingest --restore-db backup.sql  # Restore from backup
```

### Optimize Performance

```bash
biomedagent-db vacuum            # Run VACUUM ANALYZE
biomedagent-db ingest --vacuum   # Vacuum after ingestion
```

---

## File Locations

All raw data files are stored in `raw_dir` (default: `./raw`):

```
raw/
├── openfda/              # OpenFDA JSON zip files
│   └── drug-label-*.json.zip
├── orange_book_latest.zip
├── ctgov/
│   ├── ctti_aact_dump.zip
│   └── extracted/
│       └── *.dmp
├── dailymed/
│   └── full/
│       └── dm_spl_release_human_rx_part*.zip
├── bindingdb/
│   ├── BindingDB_All_202510_tsv.zip
│   └── BindingDB_All.tsv
├── chembl/
│   ├── chembl_36_postgresql.tar.gz
│   └── chembl_36/
│       └── chembl_36_postgresql/
│           └── chembl_36_postgresql.dmp
└── drugcentral/
    └── structures.molV3.sdf.gz
```

Files are reused if they exist - use `--openfda-use-local-files` or similar to skip downloads entirely.

---

## Validation Queries

### Check All Tables

```sql
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns 
     WHERE table_name = t.table_name) as columns
FROM information_schema.tables t
WHERE table_schema = 'public'
ORDER BY table_name;
```

### Check Data Source Coverage

```sql
-- OpenFDA
SELECT COUNT(*) as labels FROM labels_meta;
SELECT COUNT(*) as sections FROM sections;

-- Orange Book
SELECT COUNT(*) as products FROM orange_book_products;

-- ClinicalTrials.gov
SELECT COUNT(*) as studies FROM ctgov_studies;

-- DailyMed
SELECT COUNT(*) as products FROM dailymed_products;
SELECT COUNT(*) as sections FROM dailymed_sections;

-- BindingDB
SELECT * FROM bindingdb_stats;

-- dm_target
SELECT 
    COUNT(*) as total,
    COUNT(CASE WHEN primary_source = 'CHEMBL' THEN 1 END) as chembl,
    COUNT(CASE WHEN primary_source = 'BINDINGDB' THEN 1 END) as bindingdb,
    COUNT(CASE WHEN primary_source = 'CONSENSUS' THEN 1 END) as consensus
FROM dm_target;

-- DrugCentral
SELECT COUNT(*) as drugs FROM drugcentral_drugs;
```

### Cross-Source Query Example

```sql
-- Find drugs with FDA labels, clinical trials, and binding data
SELECT DISTINCT
    lm.title as drug_name,
    ob.trade_name,
    ct.nct_id,
    bs.gene_symbol,
    bs.pchembl_value
FROM labels_meta lm
JOIN mapping_brand_name mbn ON lm.set_id_id = mbn.set_id_id
LEFT JOIN orange_book_products ob ON LOWER(ob.trade_name) = LOWER(mbn.brand_name)
LEFT JOIN ctgov_interventions ci ON LOWER(ci.name) ILIKE '%' || LOWER(mbn.brand_name) || '%'
LEFT JOIN ctgov_studies ct ON ci.nct_id = ct.nct_id
LEFT JOIN bindingdb_summary bs ON bs.ligand_name ILIKE '%' || LOWER(mbn.brand_name) || '%'
WHERE lm.title IS NOT NULL
LIMIT 10;
```

---

## Troubleshooting

### "Table does not exist"

```bash
# Check if ingestion completed
biomedagent-db tables

# Re-run specific source
biomedagent-db ingest --skip-all-except-chembl  # Not implemented, use skip flags
```

### "Download failed"

```bash
# Use existing local files
biomedagent-db ingest --openfda-use-local-files

# Or manually download and place in raw/ directory
```

### "Out of memory"

```bash
# Use smaller batch sizes
biomedagent-db ingest --bindingdb-batch-size 5000

# Or limit records for testing
biomedagent-db ingest --n-max 10000
```

### "dm_target population failed"

dm_target requires both ChEMBL and BindingDB to be ingested first:
```bash
# Check prerequisites
biomedagent-db ingest --skip-openfda --skip-orange-book --skip-ctgov --skip-dailymed --skip-drugcentral
```

---

## Performance Notes

| Source | Download Size | Records | Time |
|--------|--------------|---------|------|
| OpenFDA | ~3 GB | ~150K labels | 30-60 min |
| Orange Book | ~2 MB | ~35K products | 1 min |
| ClinicalTrials.gov | ~1 GB | ~500K studies | 15-30 min |
| DailyMed | ~4 GB | ~150K labels | 30-60 min |
| BindingDB | ~3 GB | ~2M activities | 30-60 min |
| ChEMBL | ~1 GB | ~2M molecules | 20-40 min |
| dm_target | N/A | ~3.5K targets | 5-10 min |
| DrugCentral | ~50 MB | ~5K drugs | 2-5 min |

**Total:** ~12 GB download, 2-4 hours full ingestion

---

## CLI Reference

```bash
biomedagent-db ingest --help
```

Key options:
- `--skip-<source>` - Skip specific data source
- `--n-max N` - Limit records per source
- `--raw-dir PATH` - Custom data directory
- `--vacuum` - Optimize after ingestion
- `--get-db-info` - Show database status
- `--tables` - List all tables
- `--reset --force` - Reset database
- `--dump-db FILE` - Backup database
- `--restore-db FILE` - Restore database
