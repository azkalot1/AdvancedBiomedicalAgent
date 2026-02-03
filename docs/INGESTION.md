# Ingestion Pipeline

This document describes the order and roles of the ingestion scripts used to populate the Advanced Biomedical Agent PostgreSQL database. The main entry point is `biomedagent-db ingest` (or `python -m bioagent.data.ingest.ingest_all_postgres`), which runs the steps below in order. Individual steps can be skipped via flags (e.g. `--skip-openfda`, `--skip-chembl`).

## High-level order

1. **OpenFDA** → normalized label/product structure (foundation for mappings)
2. **Orange Book** → FDA approval and therapeutic equivalence
3. **ClinicalTrials.gov** → trials, conditions, interventions; optional RAG corpus/keys; optional enriched search table
4. **DailyMed** → structured drug labels (SPL XML)
5. **BindingDB** → molecules, targets, binding activities
6. **ChEMBL** → molecules, targets, activities, **biotherapeutics** (antibodies, peptides, etc.)
7. **dm_target** → canonical target mapping (ChEMBL + BindingDB); requires ChEMBL and BindingDB
8. **DrugCentral** → drug structures and identifiers
9. **Molecular mappings** → dm_molecule hierarchy, synonyms, **biotherapeutics**, map_ctgov_molecules, map_product_molecules, dm_compound_target_activity; requires ChEMBL, DrugCentral, BindingDB, dm_target, and CT.gov

Optional post-processing:

- **CT.gov RAG** (separate or during ingest): populate `rag_study_corpus`, then `rag_study_keys`
- **CT.gov enriched search**: create/populate `ctgov_enriched_search` (denormalized search table)
- **Full-text search indexes**: tsvector/GIN on raw ctgov_* tables (`generate_search.py`)

---

## Scripts and roles

| Script / step | Purpose | Main outputs |
|---------------|--------|--------------|
| **build_openfda.py** | Ingest OpenFDA drug labels into normalized schema (set_ids, labels_meta, sections, section_names, mappings). | OpenFDA tables; foundation for map_product_molecules. |
| **build_orange_book.py** | Ingest FDA Orange Book (approvals, products, ingredients, TE codes). | orange_book_products. |
| **build_ctgov.py** | Ingest ClinicalTrials.gov (AACT): studies, conditions, interventions, outcomes, etc. Optional: RAG corpus, RAG keys. | ctgov_* tables; optionally rag_study_corpus, rag_study_keys. |
| **build_dailymed.py** | Ingest DailyMed SPL XML into dailymed_products, dailymed_sections, dailymed_section_names. | dailymed_* tables. |
| **build_bindingdb.py** | Ingest BindingDB TSV: molecules, targets, activities (binding affinity). | bindingdb_molecules, bindingdb_targets, bindingdb_activities. |
| **build_chembl.py** | Ingest ChEMBL (molecule_dictionary, compound_structures, target_dictionary, activities, **biotherapeutics**, biotherapeutic_components, etc.). | ChEMBL schema; source for dm_molecule and **dm_biotherapeutic**. |
| **create_and_populate_dm_target.py** | Build canonical target table from ChEMBL + BindingDB. Phases: (1) ChEMBL base, (2) BindingDB augmentation, (3) gene synonyms, (4) UniProt mappings. | dm_target, dm_target_gene_synonyms, dm_target_uniprot_mappings. |
| **build_drugcentral.py** | Ingest DrugCentral (drugs, structures, identifiers). | drugcentral_drugs, etc. |
| **build_molecular_mappings.py** | Build unified molecule layer and trial/product mappings. Creates dm_molecule_concept, dm_molecule_stereo, dm_molecule, dm_molecule_synonyms; **Phase 2B: dm_biotherapeutic, dm_biotherapeutic_component, dm_biotherapeutic_synonyms** (from ChEMBL); map_ctgov_molecules (small molecules + biotherapeutics, mol_id nullable for bio-only); map_product_molecules; dm_compound_target_activity. | dm_* molecule/biotherapeutic tables, map_ctgov_molecules, map_product_molecules, dm_compound_target_activity. |
| **generate_ctgov_enriched_search.py** | Create and populate denormalized CT.gov search table (conditions, interventions, sponsors, full-text vectors, filters). | ctgov_enriched_search. |
| **generate_search.py** | Add tsvector columns and GIN indexes to raw ctgov_* tables for full-text search. | ctgov_* vector columns and indexes. |
| **extract_schema_and_examples.py** | Utility: dump table definitions and sample rows to a text file (for documentation or debugging). | Text file with schema and samples. |

---

## Dependencies

- **dm_target** requires: ChEMBL, BindingDB.
- **Molecular mappings** (build_molecular_mappings) requires: ChEMBL, DrugCentral, BindingDB, dm_target, ClinicalTrials.gov (for map_ctgov_molecules).
- **Biotherapeutics** (dm_biotherapeutic, dm_biotherapeutic_component, dm_biotherapeutic_synonyms) are populated inside build_molecular_mappings (Phase 2B) from ChEMBL biotherapeutics; they share dm_molecule_concept with small molecules and are used for trial intervention matching (map_ctgov_molecules with concept_id and mol_id NULL when matched only via dm_biotherapeutic_synonyms).

---

## Quick reference

- Full pipeline: `biomedagent-db ingest` (see README for prerequisites).
- Skip steps: `--skip-openfda`, `--skip-orange-book`, `--skip-ctgov`, `--skip-dailymed`, `--skip-drugcentral`, `--skip-chembl`, `--skip-bindingdb`, `--skip-dm-target`, `--skip-dm-molecule`.
- CT.gov RAG only: `biomedagent-db ingest --ctgov-rag-corpus-only` then `--ctgov-rag-keys-only`.
- CT.gov enriched search only: `biomedagent-db ingest --ctgov-enriched-search-only`.
- Full-text indexes only: `biomedagent-db ingest --generate-search`.
