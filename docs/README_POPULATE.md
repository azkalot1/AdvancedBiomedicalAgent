# BiomedicalAgent Data Population Guide

## Overview

The BiomedicalAgent PostgreSQL database integrates **9 data sources** into a unified schema for biomedical queries. All data is stored in the `public` schema with source-specific prefixes.

### Data Sources

| # | Source | Tables Prefix | Description |
|---|--------|---------------|-------------|
| 1 | **OpenFDA** | `labels_meta`, `sections`, `mapping_*` | Drug labels with normalized structure |
| 2 | **Orange Book** | `orange_book_*` | FDA therapeutic equivalence data |
| 3 | **ClinicalTrials.gov** | `ctgov_*`, `ctgov_enriched_search` | Clinical trials from CTTI AACT dump |
| 4 | **DailyMed** | `dailymed_*` | SPL drug labels with semantic search |
| 5 | **BindingDB** | `bindingdb_*` | Molecular targets & binding affinity |
| 6 | **ChEMBL** | `chembl_*` (in chembl schema) | Biochemical annotations |
| 7 | **dm_target** | `dm_target*` | Canonical target mapping (ChEMBL + BindingDB) |
| 8 | **DrugCentral** | `drugcentral_*` | Molecular structures & chemical data |
| 9 | **dm_molecule** | `dm_molecule*`, `map_*` | Unified molecules + CT.gov/product mappings |

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

**Script:** `build_ctgov.py`, `generate_ctgov_enriched_search.py`  
**Tables Created:**
- `ctgov_studies` - Main study records
- `ctgov_interventions` - Drug/treatment interventions
- `ctgov_conditions` - Medical conditions studied
- `ctgov_outcomes` - Study outcome measures
- `ctgov_*` - 40+ additional tables
- `rag_study_corpus` - RAG-ready study JSON
- `rag_study_keys` - Materialized view for RAG lookup
- `ctgov_enriched_search` - Denormalized search table with trigram indexes

**Key Features:**
- Full CTTI AACT PostgreSQL dump (~500K studies)
- Full-text search indexes on key tables
- RAG functionality for study retrieval
- Enriched search table with aggregated conditions, interventions, sponsors
- Trigram similarity indexes for flexible searching

**Options:**
```bash
# Populate RAG corpus (takes hours!)
biomedagent-db ingest --ctgov-populate-rag
biomedagent-db ingest --ctgov-rag-buckets 32

# Standalone RAG population
biomedagent-db ingest --ctgov-rag-corpus-only
biomedagent-db ingest --ctgov-rag-keys-only

# Populate enriched search table (faster flexible search)
biomedagent-db ingest --ctgov-populate-enriched-search
biomedagent-db ingest --ctgov-enriched-search-batch-size 2000

# Standalone enriched search table
biomedagent-db ingest --ctgov-enriched-search-only

# Or via generate-ctgov-search command
biomedagent-db generate-ctgov-search create
biomedagent-db generate-ctgov-search populate 1000
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

### 9. dm_molecule Unified Molecular Mappings

**Script:** `build_molecular_mappings.py`  
**Tables Created:**
- `dm_molecule` - Unified molecule table (all specific forms)
- `dm_molecule_concept` - Drug concept grouping (Level 1: all forms of same drug)
- `dm_molecule_stereo` - Stereo forms grouping (Level 2: same stereochemistry)
- `dm_molecule_synonyms` - Comprehensive synonym dictionary
- `map_ctgov_molecules` - CT.gov intervention to molecule mappings
- `map_product_molecules` - OpenFDA/DailyMed to molecule mappings
- `dm_compound_target_activity` - Unified activity materialized view

**Key Features:**
- 3-level molecular hierarchy (concept → stereo → molecule)
- RDKit-based normalization (parent_smiles, parent_inchi_key_14, parent_stereo_inchi_key)
- Salt/form detection and classification
- Stereochemistry analysis (ACHIRAL, DEFINED, RACEMIC, etc.)
- Cross-database unification (ChEMBL + DrugCentral + BindingDB)
- Morgan fingerprints for similarity search (mfp2, ffp2)
- RDKit mol column for structure-based queries
- CT.gov intervention to molecule mapping with confidence scoring
- Product (OpenFDA/DailyMed) to molecule mapping

**Population:**
| Phase | Description | Records |
|-------|-------------|---------|
| 1 | Create 3-level hierarchy tables | N/A |
| 2 | Populate from ChEMBL | ~2M molecules |
| 3 | Add DrugCentral molecules | ~5K |
| 4 | Add BindingDB molecules | ~1.5M |
| 5 | Build synonym dictionary | ~10M synonyms |
| 6 | Map CT.gov interventions | ~200K mappings |
| 7 | Map product labels | ~50K mappings |
| 8 | Create activity view | ~5M activities |

**Prerequisites:**
- ChEMBL must be ingested first
- DrugCentral must be ingested first
- BindingDB must be ingested first
- dm_target must be populated first
- CT.gov must be ingested for intervention mappings

**Options:**
```bash
biomedagent-db ingest --skip-dm-molecule  # Skip molecular mappings
```

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
SELECT COUNT(*) as enriched_rows FROM ctgov_enriched_search;

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

-- dm_molecule hierarchy
SELECT 
    (SELECT COUNT(*) FROM dm_molecule_concept) as concepts,
    (SELECT COUNT(*) FROM dm_molecule_stereo) as stereo_forms,
    (SELECT COUNT(*) FROM dm_molecule) as molecules,
    (SELECT COUNT(*) FROM dm_molecule_synonyms) as synonyms;

-- CT.gov molecule mappings
SELECT 
    COUNT(*) as total_mappings,
    COUNT(DISTINCT nct_id) as trials_mapped,
    COUNT(DISTINCT mol_id) as molecules_mapped,
    AVG(confidence) as avg_confidence
FROM map_ctgov_molecules;

-- Product molecule mappings
SELECT 
    source_table,
    COUNT(*) as mappings,
    COUNT(DISTINCT mol_id) as molecules
FROM map_product_molecules
GROUP BY source_table;

-- Activity data coverage
SELECT 
    COUNT(*) as total_activities,
    COUNT(DISTINCT mol_id) as molecules,
    COUNT(DISTINCT gene_symbol) as targets,
    AVG(pchembl_value) as avg_pchembl
FROM dm_compound_target_activity;
```

### Cross-Source Query Examples

```sql
-- 1. Find drugs with FDA labels, clinical trials, and binding data (using dm_molecule)
SELECT DISTINCT
    c.preferred_name AS drug_name,
    m.canonical_smiles,
    s.nct_id,
    s.brief_title,
    a.gene_symbol,
    a.pchembl_value
FROM dm_molecule_concept c
JOIN dm_molecule m ON c.concept_id = m.concept_id
LEFT JOIN map_ctgov_molecules map ON c.concept_id = map.concept_id
LEFT JOIN ctgov_studies s ON map.nct_id = s.nct_id
LEFT JOIN dm_compound_target_activity a ON m.mol_id = a.mol_id
WHERE c.preferred_name ILIKE '%imatinib%'
  AND a.pchembl_value > 7
LIMIT 20;

-- 2. Find trials for drugs targeting a specific gene
SELECT DISTINCT
    s.nct_id,
    s.brief_title,
    s.phase,
    s.overall_status,
    c.preferred_name AS drug,
    a.pchembl_value
FROM dm_compound_target_activity a
JOIN dm_molecule_concept c ON a.concept_id = c.concept_id
JOIN map_ctgov_molecules map ON a.mol_id = map.mol_id
JOIN ctgov_studies s ON map.nct_id = s.nct_id
WHERE a.gene_symbol = 'EGFR'
  AND a.pchembl_value > 7
  AND s.overall_status = 'Recruiting'
ORDER BY a.pchembl_value DESC;

-- 3. Connect FDA products to clinical trials via molecule
SELECT DISTINCT
    lm.title AS fda_product,
    c.preferred_name AS drug_concept,
    s.nct_id,
    s.phase
FROM map_product_molecules pm
JOIN dm_molecule_concept c ON pm.concept_id = c.concept_id
JOIN map_ctgov_molecules cm ON c.concept_id = cm.concept_id
JOIN ctgov_studies s ON cm.nct_id = s.nct_id
JOIN labels_meta lm ON pm.set_id = lm.set_id_id::text
WHERE s.overall_status = 'Completed'
  AND s.has_results = true
LIMIT 20;
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

### "dm_molecule population failed"

dm_molecule requires ChEMBL, DrugCentral, BindingDB, dm_target, and CT.gov:
```bash
# Check prerequisites exist
biomedagent-db tables | grep -E "(chembl|drugcentral|bindingdb|dm_target|ctgov)"

# Re-run only dm_molecule
biomedagent-db ingest --skip-openfda --skip-orange-book --skip-ctgov --skip-dailymed --skip-bindingdb --skip-chembl --skip-dm-target --skip-drugcentral
```

### "RDKit extension not available"

The RDKit PostgreSQL extension is optional but required for similarity/substructure search:
```bash
# Install RDKit
biomedagent-db install-rdkit

# Restart PostgreSQL
sudo systemctl restart postgresql

# Create extension
biomedagent-db create-rdkit-ext
```

---

## Performance Notes

| Source | Download Size | Records | Time |
|--------|--------------|---------|------|
| OpenFDA | ~3 GB | ~150K labels | 30-60 min |
| Orange Book | ~2 MB | ~35K products | 1 min |
| ClinicalTrials.gov | ~1 GB | ~500K studies | 15-30 min |
| CT.gov Enriched Search | N/A | ~500K rows | 10-20 min |
| DailyMed | ~4 GB | ~150K labels | 30-60 min |
| BindingDB | ~3 GB | ~2M activities | 30-60 min |
| ChEMBL | ~1 GB | ~2M molecules | 20-40 min |
| dm_target | N/A | ~3.5K targets | 5-10 min |
| DrugCentral | ~50 MB | ~5K drugs | 2-5 min |
| dm_molecule | N/A | ~3M molecules | 30-60 min |

**Total:** ~12 GB download, 3-5 hours full ingestion (including molecular mappings)

---

## CLI Reference

```bash
biomedagent-db ingest --help
```

Key options:
- `--skip-<source>` - Skip specific data source (openfda, orange-book, ctgov, dailymed, bindingdb, chembl, dm-target, drugcentral, dm-molecule)
- `--n-max N` - Limit records per source
- `--raw-dir PATH` - Custom data directory
- `--vacuum` - Optimize after ingestion
- `--get-db-info` - Show database status
- `--tables` - List all tables
- `--reset --force` - Reset database
- `--dump-db FILE` - Backup database
- `--restore-db FILE` - Restore database

CT.gov RAG options:
- `--ctgov-populate-rag` - Populate RAG during ingestion
- `--ctgov-rag-corpus-only` - Only populate RAG corpus
- `--ctgov-rag-keys-only` - Only populate RAG keys
- `--ctgov-rag-buckets N` - Number of buckets (default: 16)

CT.gov Enriched Search options:
- `--ctgov-populate-enriched-search` - Populate during ingestion
- `--ctgov-enriched-search-only` - Standalone population
- `--ctgov-enriched-search-batch-size N` - Batch size (default: 1000)

BindingDB options:
- `--bindingdb-all-organisms` - Include all organisms (default: human only)
- `--bindingdb-batch-size N` - Batch size (default: 10000)
- `--bindingdb-force-recreate` - Drop and recreate tables

ChEMBL options:
- `--chembl-force-recreate` - Drop and recreate tables

OpenFDA options:
- `--openfda-files N` - Limit number of files
- `--openfda-use-local-files` - Use existing downloads
