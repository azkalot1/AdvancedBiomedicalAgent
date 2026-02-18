# dm_target: Canonical Target Mapping - Quick Start

## Overview

The `dm_target` table is a unified representation of biological targets (proteins) that consolidates data from multiple sources:

- **ChEMBL**: 17,000+ targets (most curated for drug targets)
- **BindingDB**: 3,406 targets (comprehensive binding affinity data)
- **Clinical Trials**: Target context for drug indication

## Files in This Guide

1. **DM_TARGET_SCHEMA_REVIEW.md** - Comprehensive review with rationale
2. **scripts/create_dm_target.py** - Schema creation script
3. **scripts/populate_dm_target.py** - Data population script (next step)

## Quick Start: 3 Steps

### Step 1: Review the Schema

```bash
# Read the comprehensive review
cat DM_TARGET_SCHEMA_REVIEW.md
```

Key improvements over your original proposal:
- ✅ 24 columns vs 8 (added names, cross-refs, audit trail, quality tracking)
- ✅ 8 indexes vs 2 (optimized query performance)
- ✅ 2 supporting tables (synonyms, UniProt mappings)
- ✅ Data quality tracking (source, confidence_score)

### Step 2: Create Tables

```bash
cd /home/sergey/projects/AdvancedBiomedicalAgent
python scripts/create_dm_target.py
```

This will:
1. Print the full schema review
2. Ask for confirmation
3. Create 3 tables:
   - `dm_target` (canonical targets)
   - `dm_target_gene_synonyms` (gene name variants)
   - `dm_target_uniprot_mappings` (protein isoforms & secondary IDs)

### Step 3: Populate with Data

```bash
# Coming next: population script
python scripts/populate_dm_target.py
```

This will populate 4 phases:
1. ChEMBL base loading (~2,000 human targets)
2. BindingDB augmentation (~400 new targets + cross-refs)
3. Gene synonym consolidation (~8,000 synonyms)
4. UniProt mapping (~3,000 accessions)

## Key Design Decisions

### Why UniProt as Unique Key?

- One UniProt ID = one canonical protein sequence
- Stable, curated by UniProt consortium
- Enables precise protein identification
- Example: `P04637` = TP53 (tumor suppressor protein)

### Why Support Tables?

**dm_target_gene_synonyms**: Gene symbols aren't unique
```
CD4 -> human T-cell antigen (P01730)
CD4 -> mouse T-cell antigen (P06332)
TP53 -> "p53", "LFS1", "tumor protein p53"
```

**dm_target_uniprot_mappings**: UniProt has isoforms & secondary IDs
```
P04637      -> canonical
P04637-1    -> isoform α
P04637-2    -> isoform β
A1C5L3      -> secondary (after record merge)
```

### Why Confidence Score?

When ChEMBL and BindingDB disagree, score indicates reliability:
- 10: High confidence (ChEMBL curated, or consensus)
- 8: BindingDB only (less editorial review)
- 7: Conflicting data (flagged for manual review)

## Query Examples

### Find all identifiers for a protein

```sql
SELECT 
    dt.uniprot_id,
    dt.gene_symbol,
    dgs.gene_symbol as synonym,
    dum.uniprot_accession,
    dt.chembl_id,
    dt.bindingdb_target_id
FROM dm_target dt
LEFT JOIN dm_target_gene_synonyms dgs ON dt.target_id = dgs.target_id
LEFT JOIN dm_target_uniprot_mappings dum ON dt.target_id = dum.target_id
WHERE dt.gene_symbol = 'TP53'
  AND dt.organism = 'Homo sapiens';
```

### Find all kinase targets

```sql
SELECT dt.gene_symbol, dt.uniprot_id, dt.chembl_id
FROM dm_target dt
JOIN protein_classification pc ON dt.protein_class_id = pc.protein_class_id
WHERE pc.protein_class_desc LIKE '%kinase%'
  AND dt.organism = 'Homo sapiens'
ORDER BY dt.gene_symbol;
```

### Data quality check - find conflicts

```sql
SELECT 
    dt.gene_symbol,
    dt.uniprot_id,
    dt.primary_source,
    dt.confidence_score,
    dt.notes
FROM dm_target dt
WHERE dt.confidence_score < 8
  AND dt.data_sources && ARRAY['CHEMBL', 'BINDINGDB']
ORDER BY dt.confidence_score;
```

## Expected Results

After full population:

| Metric | Count |
|--------|-------|
| Human protein targets | ~2,500 |
| ChEMBL-matched | ~2,300 |
| BindingDB-new | ~200 |
| Gene synonyms | ~8,000+ |
| UniProt mappings | ~3,000+ |

## Next Steps

1. ✅ Review schema (this document + DM_TARGET_SCHEMA_REVIEW.md)
2. 📋 Run `create_dm_target.py` to create tables
3. 📊 Run `populate_dm_target.py` to load data
4. ✓ Validate with data quality checks
5. 📚 Document any manual deduplication decisions

## Troubleshooting

### Tables not created?
- Check database connection in `config.py`
- Verify user has CREATE TABLE permissions
- Run: `biomedagent-db setup_postgres` (or ensure it was run initially)

### Need to reset?
```bash
# Full database reset (WARNING: deletes all data)
python config.py reset --force

# Then recreate
python scripts/create_dm_target.py
```

## References

- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **BindingDB**: https://www.bindingdb.org/
- **UniProt**: https://www.uniprot.org/
- **NCBI Taxonomy**: https://www.ncbi.nlm.nih.gov/taxonomy/
- **Ensembl**: https://www.ensembl.org/

---

**Created:** 2025-10-26  
**Database**: AdvancedBiomedicalAgent  
**Schema Version**: 1.0
