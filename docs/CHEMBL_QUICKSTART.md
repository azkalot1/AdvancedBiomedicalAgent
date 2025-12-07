# ChEMBL Integration - Quick Start Guide

## Overview

This guide walks you through integrating ChEMBL with your ClinicalTrials.gov database to enable **molecule-protein binding annotation** for clinical trial interventions.

## What You'll Get

✅ **Drug structures** (SMILES, InChI) for clinical trial interventions  
✅ **Drug-target binding data** (IC50, Ki, EC50, Kd values)  
✅ **Protein target annotations** (UniProt IDs, gene names)  
✅ **~21M bioactivity measurements** from ChEMBL  
✅ **FDA approval status** for drugs (max_phase field)  

---

## Prerequisites

- **PostgreSQL** installed with ~100 GB free disk space
- **Database user** with create database permissions
- **Internet connection** for download (~15 GB)
- **Time**: ~2-3 hours total (mostly automated)

---

## Step-by-Step Installation

### Step 1: Download and Install ChEMBL Database (~90 minutes)

```bash
cd /home/sergey/projects/AdvancedBiomedicalAgent

# Run the download and installation script
bash scripts/download_chembl.sh
```

**What this does:**
1. Downloads ChEMBL 34 PostgreSQL dump (~15 GB, 30-60 min)
2. Creates `chembl_34` database
3. Restores all ChEMBL tables (~30-60 min)
4. Grants permissions to your user

**Output:** New database `chembl_34` with ~2.4M molecules and 21M bioactivities

---

### Step 2: Create Mapping Tables (~1 minute)

```bash
# Connect to your main database and create mapping tables
psql -U database_user -d database -f scripts/create_chembl_mapping_tables.sql
```

**What this creates:**
- `intervention_chembl_mapping` - Maps CTG interventions → ChEMBL molecules
- `intervention_target_binding` - Cached drug-target binding data
- Views for common queries
- Helper functions

---

### Step 3: Map Interventions to ChEMBL (~30-60 minutes)

```bash
cd /home/sergey/projects/AdvancedBiomedicalAgent

# Activate environment
conda activate biomedagent

# Test with first 100 interventions (dry run)
python -m bioagent.data.ingest.map_chembl --dry-run --limit 100

# Map all interventions
python -m bioagent.data.ingest.map_chembl

# Extract drug-target binding data
python -m bioagent.data.ingest.map_chembl --extract-binding
```

**Expected mapping success rate:** ~70-85% for actual drugs

**Output:**
- Mapped interventions in `intervention_chembl_mapping`
- Binding data in `intervention_target_binding`

---

## Verify Installation

```sql
-- Connect to main database
psql -U database_user -d database

-- Check mappings
SELECT 
    COUNT(*) as total_interventions,
    COUNT(DISTINCT icm.intervention_id) as mapped_interventions,
    COUNT(DISTINCT icm.chembl_id) as unique_molecules
FROM ctgov_interventions i
LEFT JOIN intervention_chembl_mapping icm ON i.id = icm.intervention_id
WHERE i.intervention_type = 'DRUG';

-- Check binding data
SELECT COUNT(*) as total_binding_records
FROM intervention_target_binding;

-- Example: View approved drugs in trials
SELECT * FROM intervention_approved_drugs LIMIT 10;
```

---

## Example Queries

### Query 1: Find all targets for a drug in a clinical trial

```sql
SELECT * FROM get_intervention_targets('NCT00000620');
```

**Output:**
```
target_name                  | uniprot_id | activity_type | best_value | units
-----------------------------|------------|---------------|------------|------
Insulin receptor             | P06213     | IC50          | 5.2        | nM
Glucagon receptor            | P47871     | Ki            | 12.3       | nM
```

### Query 2: Find trials testing drugs that target a specific protein

```sql
SELECT 
    i.nct_id,
    i.name as drug_name,
    itb.activity_type,
    itb.standard_value,
    itb.standard_units
FROM intervention_target_binding itb
JOIN ctgov_interventions i ON itb.intervention_id = i.id
WHERE itb.uniprot_id = 'P00533'  -- EGFR
  AND itb.activity_type = 'IC50'
  AND itb.standard_value < 100   -- High affinity
ORDER BY itb.standard_value
LIMIT 10;
```

### Query 3: High-affinity drug-target interactions

```sql
SELECT * FROM intervention_potent_binding LIMIT 20;
```

**Output:**
```
nct_id      | drug_name  | target_name | uniprot_id | activity_type | value | units
------------|------------|-------------|------------|---------------|-------|------
NCT00123456 | Gefitinib  | EGFR        | P00533     | IC50          | 0.8   | nM
NCT00789012 | Erlotinib  | EGFR        | P00533     | IC50          | 2.1   | nM
```

### Query 4: Find drug structures (SMILES) for a trial

```sql
SELECT 
    i.name as drug_name,
    icm.chembl_id,
    icm.canonical_smiles,
    icm.max_phase as approval_status
FROM ctgov_interventions i
JOIN intervention_chembl_mapping icm ON i.id = icm.intervention_id
WHERE i.nct_id = 'NCT00000620';
```

---

## Database Schema Overview

### Key Tables in `chembl_34` Database

| Table | Rows | Description |
|-------|------|-------------|
| `molecule_dictionary` | ~2.4M | Compound IDs, names, approval status |
| `compound_structures` | ~2.3M | SMILES, InChI structures |
| `activities` | ~21M | Bioactivity measurements (IC50, Ki, etc.) |
| `target_dictionary` | ~14K | Protein targets |
| `component_sequences` | ~15K | UniProt mappings |
| `molecule_synonyms` | ~1.3M | Drug names, brand names |

### Key Tables in Your Main Database

| Table | Description |
|-------|-------------|
| `intervention_chembl_mapping` | CTG intervention → ChEMBL molecule |
| `intervention_target_binding` | Cached binding data |
| `intervention_approved_drugs` | View of FDA-approved drugs |
| `intervention_potent_binding` | View of high-affinity interactions |

---

## Maintenance and Updates

### Update ChEMBL to Newer Version

ChEMBL releases new versions quarterly. To update:

```bash
# Download new version
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_postgresql.tar.gz

# Drop old database and create new
sudo -u postgres dropdb chembl_34
sudo -u postgres createdb chembl_35
sudo -u postgres pg_restore --dbname=chembl_35 chembl_35_postgresql.dmp

# Remap interventions
python -m bioagent.data.ingest.map_chembl --extract-binding
```

### Mapping Statistics

After mapping, check stats:

```sql
SELECT 
    mapping_method,
    COUNT(*) as count,
    ROUND(AVG(confidence_score), 2) as avg_confidence
FROM intervention_chembl_mapping
GROUP BY mapping_method
ORDER BY count DESC;
```

---

## Troubleshooting

### Issue: Low mapping rate (<50%)

**Cause:** Many CTG interventions are procedures, devices, or have non-standard names

**Solution:**
- Check intervention types: `SELECT intervention_type, COUNT(*) FROM ctgov_interventions GROUP BY intervention_type;`
- Only drugs/biologicals should map to ChEMBL

### Issue: pg_restore fails

**Cause:** Insufficient disk space or memory

**Solution:**
```bash
# Check space
df -h /var/lib/postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Reduce parallel jobs if memory limited
sudo -u postgres pg_restore --jobs=2 ...  # Instead of --jobs=4
```

### Issue: Slow mapping

**Cause:** No indexes on ChEMBL tables

**Solution:**
```sql
-- ChEMBL dump should include indexes, but verify:
\di molecule_dictionary*
\di molecule_synonyms*

-- If missing, ChEMBL provides index creation scripts
```

---

## Performance Notes

### Disk Space

- **ChEMBL download**: 15 GB
- **ChEMBL database**: 80 GB
- **Temp space during restore**: 20 GB
- **Total needed**: ~120 GB

### Speed

- **Download**: 30-60 min (depends on connection)
- **Restore**: 30-60 min (depends on CPU/disk)
- **Mapping**: 30-60 min (depends on intervention count)
- **Binding extraction**: 10-20 min

### Optimization

```sql
-- Add indexes if querying is slow
CREATE INDEX IF NOT EXISTS idx_activities_molregno_type 
    ON activities(molregno, standard_type) 
    WHERE standard_value IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_molecule_synonyms_lower 
    ON molecule_synonyms(LOWER(synonyms));
```

---

## Summary

After completing these steps, you'll have:

1. ✅ ChEMBL database with 2.4M molecules locally installed
2. ✅ Interventions mapped to ChEMBL structures (SMILES, InChI)
3. ✅ Drug-target binding data (IC50, Ki, etc.) cached locally
4. ✅ Protein targets with UniProt IDs
5. ✅ SQL views and functions for easy querying

**Total time investment:** ~2-3 hours  
**Result:** Full molecule-protein binding annotation capability! 🎉

---

## Next Steps

1. **Integrate with search**: Update `clinical_trial_searches.py` to include target information
2. **Add to RAG**: Include binding data in `rag_study_json.sql`
3. **Visualize structures**: Use RDKit to render molecular structures
4. **Protein analysis**: Link to AlphaFold structures, pathways

See `CHEMBL_INGESTION_GUIDE.md` for detailed technical documentation.




