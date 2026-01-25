# BioAgent Database Schema Documentation

This document describes the PostgreSQL database structure for the Advanced Biomedical Agent. The database integrates clinical trial data, molecular structures, drug information, and regulatory data into a unified system.

## Database Architecture Overview

The database is organized into several logical layers:

1. **Clinical Trial Data Layer** - Clinical trial information from ClinicalTrials.gov
2. **Molecular Data Layer** - Chemical structures and molecular information from multiple sources
3. **Target/Protein Data Layer** - Biological targets and protein information
4. **Drug/Product Data Layer** - FDA-regulated drug and supplement information
5. **Regulatory/Label Data Layer** - FDA drug labels and regulatory information
6. **Integrated Data Layer** - Unified views and mappings connecting all layers

---

## 1. CLINICAL TRIAL DATA LAYER

### Core Clinical Trial Tables

#### `ctgov_studies`
**Purpose:** Main clinical trial records from ClinicalTrials.gov (CTTI AACT database)

**Data Stored:**
- Trial identifiers (nct_id, org_study_id)
- Trial descriptors (brief_title, official_title)
- Trial status (overall_status, recruitment_status)
- Trial phase information (phase)
- Dates (start_date, completion_date, primary_completion_date)
- Population and design information (enrollment, study_type)
- Study design details

**Key Columns:** nct_id (PRIMARY), brief_title, overall_status, phase, enrollment, start_date

---

#### `ctgov_conditions`
**Purpose:** Medical conditions studied in clinical trials

**Data Stored:**
- Condition names and terminology
- Links to specific trials (via nct_id)
- Mesh term classifications (downcase_mesh_term)

**Key Columns:** id (PRIMARY), nct_id (FOREIGN KEY → ctgov_studies), name, downcase_mesh_term

**Relationship:** One trial can have multiple conditions (1:N)

---

#### `ctgov_interventions`
**Purpose:** Treatments and interventions tested in clinical trials

**Data Stored:**
- Intervention names (name)
- Intervention type classification (intervention_type: DRUG, BIOLOGICAL, DEVICE, etc.)
- Links to specific trials (via nct_id)
- Detailed descriptions (description)

**Key Columns:** id (PRIMARY), nct_id (FOREIGN KEY → ctgov_studies), name, intervention_type

**Relationship:** One trial can have multiple interventions (1:N)

---

#### `ctgov_browse_conditions`
**Purpose:** Standardized condition hierarchy and mesh terms

**Data Stored:**
- Mesh condition terminology
- Standardized condition classifications
- Links to trials (via nct_id)

**Key Columns:** id (PRIMARY), nct_id (FOREIGN KEY → ctgov_studies), downcase_mesh_term

---

#### `ctgov_browse_interventions`
**Purpose:** Standardized intervention hierarchy and mesh terms

**Data Stored:**
- Mesh intervention terminology
- Standardized intervention classifications
- Links to trials (via nct_id)

**Key Columns:** id (PRIMARY), nct_id (FOREIGN KEY → ctgov_studies), downcase_mesh_term

---

#### `ctgov_design_groups`, `ctgov_design_outcomes`, `ctgov_outcome_measurements`
**Purpose:** Trial design structure and outcome measurements

**Data Stored:**
- Study group definitions and descriptions
- Planned outcome measures
- Outcome measurement details and results

**Relationships:** All linked to trials through study hierarchy

---

#### `ctgov_detailed_descriptions`, `ctgov_brief_summaries`
**Purpose:** Full-text trial descriptions and summaries

**Data Stored:**
- Complete narrative descriptions of trials
- Hypothesis and design rationale
- Brief summary paragraphs

**Relationships:** One-to-one with ctgov_studies (or one per trial)

---

### RAG (Retrieval-Augmented Generation) Tables for Clinical Trials

#### `rag_study_corpus`
**Purpose:** Processed study documents for semantic search

**Data Stored:**
- Complete study JSON representation (study_json)
- Linked nct_id for reference

**Key Columns:** nct_id (PRIMARY), study_json (JSONB), created_at, updated_at

**Purpose:** Enables semantic retrieval of study information for AI/LLM queries

---

#### `rag_study_keys` (Materialized View)
**Purpose:** Indexed study metadata for fast semantic search

**Data Stored:**
- Study identifiers (nct_id)
- Normalized condition names
- Normalized intervention names
- Aliases and mesh term mappings

**Key Columns:** nct_id, condition_name, intervention_name, alias_name, mesh_condition_name, mesh_intervention_name

**Purpose:** Enables rapid matching of study attributes to normalized terms

---

#### `ctgov_enriched_search` (Denormalized Search Table)
**Purpose:** Optimized denormalized table for flexible clinical trial searching

**Data Stored:**
- Study identifiers (nct_id PRIMARY KEY)
- Aggregated arrays:
  - conditions[] - All condition names
  - interventions[] - All intervention names
  - sponsors[] - All sponsor names
  - countries[] - All country names
- Normalized text columns (for trigram similarity):
  - brief_title_norm, official_title_norm
  - conditions_text_norm, interventions_text_norm
  - sponsors_text_norm, keywords_norm
- Full-text search vectors:
  - title_tsv, conditions_tsv, interventions_tsv
- Filterable metadata:
  - phase, overall_status, study_type
  - enrollment, enrollment_type
  - start_date, completion_date, primary_completion_date
  - has_results (boolean)
  - is_fda_regulated (boolean)
  - lead_sponsor_name, lead_sponsor_class

**Indexes:**
- GIN indexes on array columns (conditions, interventions, sponsors, countries)
- GIN indexes on tsvector columns (title_tsv, conditions_tsv, interventions_tsv)
- GiST indexes for trigram similarity on _norm columns
- B-tree indexes on filterable columns (phase, overall_status, enrollment, dates)

**Purpose:** Enables fast, flexible searching with:
- Array containment queries (@> operator)
- Full-text search (@@)
- Trigram similarity (% operator, similarity())
- Combined filtering on multiple criteria

**Population:** Created by `generate_ctgov_enriched_search.py` after base CT.gov ingestion

---

## 2. MOLECULAR DATA LAYER

### Unified Molecular Integration Tables

#### `dm_molecule` (Integrated Data Model - Level 3: Specific Forms)
**Purpose:** Consolidated molecular structures from all sources (ChEMBL, DrugCentral, BindingDB). Represents specific molecular forms including salts.

**Data Stored:**
- Original structure identifiers (inchi_key, canonical_smiles, standard_inchi)
- Preferred names and variants (pref_name)
- RDKit-normalized properties (parent_smiles, parent_inchi_key_14, parent_stereo_inchi_key)
- Salt and form information (is_salt, salt_form)
- Stereochemistry details (stereo_type, n_chiral_centers, n_defined_centers)
- Cross-database identifiers (chembl_id, drugcentral_id, bindingdb_monomer_id, pubchem_cid)
- Fingerprints for similarity search (mfp2, ffp2) - Morgan fingerprints (radius 2)
- RDKit mol object for structure-based queries (mol)
- Source tracking (sources array: CHEMBL, DC, BDB)
- Hierarchy links (concept_id → dm_molecule_concept, stereo_id → dm_molecule_stereo)
- Timestamps (created_at, updated_at)

**Key Columns:** mol_id (PRIMARY), inchi_key (UNIQUE), concept_id (FOREIGN KEY), stereo_id (FOREIGN KEY)

**Indexes:**
- `idx_dm_molecule_inchi_key` - Unique structure lookup
- `idx_dm_molecule_concept_id` - Concept grouping
- `idx_dm_molecule_stereo_id` - Stereo form grouping
- `idx_dm_molecule_parent_inchi_key_14` - Parent structure lookup
- `idx_dm_molecule_mfp2` - Morgan fingerprint similarity (GiST)
- `idx_dm_molecule_ffp2` - Feature fingerprint similarity (GiST)

**Purpose:** Single source of truth for all molecular structures in the system

---

#### `dm_molecule_concept` (Level 1: Drug Concepts)
**Purpose:** Groups all forms of the same drug (therapeutic equivalence level). Represents the "drug" regardless of salt form or stereochemistry.

**Data Stored:**
- Connectivity-only InChI key (parent_inchi_key_14) - first 14 chars, no stereo, no salts
- Representative canonical parent structure (parent_smiles)
- Preferred drug name (preferred_name)
- External identifiers (rxnorm_cui, drugbank_id, unii)
- Metadata on variants (has_stereo_variants, has_salt_forms, n_forms)
- Source tracking (primary_source, sources array)
- Timestamps (created_at, updated_at)

**Key Columns:** concept_id (PRIMARY), parent_inchi_key_14 (UNIQUE)

**Indexes:**
- `idx_dm_molecule_concept_preferred_name` - Name lookup
- `idx_dm_molecule_concept_parent_inchi_key_14` - Structure lookup

**Purpose:** Represents "drug concepts" - groups all salts, stereoisomers, hydrates together

**Example:** All forms of atorvastatin (calcium salt, free acid, hydrate) share the same concept_id

---

#### `dm_molecule_stereo` (Level 2: Stereo Forms)
**Purpose:** Groups structures with same stereochemistry but different salt forms. Represents the API (Active Pharmaceutical Ingredient) level.

**Data Stored:**
- Concept reference (concept_id - FOREIGN KEY → dm_molecule_concept)
- Full parent InChI key with stereo (parent_stereo_inchi_key)
- Parent SMILES with stereochemistry
- Stereochemistry classification (stereo_type: ACHIRAL, DEFINED, RACEMIC, PARTIAL, UNKNOWN)
- Chiral center counts (n_chiral_centers, n_defined_centers)
- API flag (is_api) - marks Active Pharmaceutical Ingredient form
- Timestamps (created_at, updated_at)

**Key Columns:** stereo_id (PRIMARY), concept_id (FOREIGN KEY), parent_stereo_inchi_key (UNIQUE)

**Indexes:**
- `idx_dm_molecule_stereo_concept_id` - Concept grouping
- `idx_dm_molecule_stereo_parent_stereo_inchi_key` - Structure lookup

**Purpose:** API-level identity - specific stereoisomer

**Example:** (R)-atorvastatin and (S)-atorvastatin have different stereo_ids but same concept_id

---

#### `dm_molecule_synonyms`
**Purpose:** Comprehensive drug name dictionary for flexible matching

**Data Stored:**
- Molecule and concept references (mol_id, concept_id)
- Synonym text and normalized version (synonym, synonym_lower)
- Synonym type classification (syn_type: BRAND, GENERIC, RESEARCH_CODE, INN, USAN, IUPAC, CAS)
- Source of synonym (source: CHEMBL, DC, BDB)
- Primary flag (is_primary) - marks preferred name
- Timestamps (created_at)

**Key Columns:** id (PRIMARY), mol_id (FOREIGN KEY), concept_id (FOREIGN KEY)

**Indexes:**
- `idx_dm_molecule_synonyms_mol_id` - Molecule lookup
- `idx_dm_molecule_synonyms_concept_id` - Concept lookup
- `idx_dm_molecule_synonyms_synonym_lower` - Case-insensitive name search
- `idx_dm_molecule_synonyms_syn_type` - Type filtering
- `idx_dm_molecule_synonyms_trgm` - Trigram similarity (GiST)

**Relationships:** 
- Many synonyms per molecule (N:1)
- Many synonyms per concept (N:1)

**Purpose:** Enable flexible drug name matching and lookup (supports fuzzy matching via trigram)

---

### Source-Specific Molecular Tables

#### `chembl_molecule_dictionary`, `chembl_compound_structures`
**Purpose:** ChEMBL molecular data

**Data Stored:**
- ChEMBL identifiers (chembl_id)
- Canonical SMILES and InChI
- Molecular weights and properties
- Preferred names

**Relationship:** Linked to dm_molecule via chembl_id

---

#### `drugcentral_drugs`
**Purpose:** DrugCentral drug information

**Data Stored:**
- DrugCentral identifiers (dc_id)
- Drug names (name, name_norm)
- InChI keys and SMILES
- Molecular properties (formula, exact_mass)
- CAS registry numbers and synonyms
- URLs to sources

**Relationship:** Linked to dm_molecule via inchi_key

---

#### `bindingdb_molecules`, `bindingdb_targets`, `bindingdb_activities`
**Purpose:** BindingDB binding affinity data

**Data Stored:**
- Molecule information (ligand names, SMILES, InChI keys, CAS numbers)
- Target information (gene symbols, UniProt IDs, organism)
- Binding measurements (Ki, IC50, Kd, EC50 in nM)
- pChEMBL values (potency index)

**Relationships:**
- bindingdb_molecules → dm_molecule (via inchi_key)
- bindingdb_activities links molecules to targets (N:N)

---

## 3. TARGET/PROTEIN DATA LAYER

#### `dm_target` (Integrated Data Model)
**Purpose:** Unified biological targets/proteins from all sources

**Data Stored:**
- Target identifiers (target_id)
- Gene information (gene_symbol, uniprot_id)
- Target names and descriptions
- Cross-database identifiers (chembl_tid, bindingdb_target_id)
- Organism information
- Target classification (target_type)

**Key Columns:** target_id (PRIMARY), uniprot_id, gene_symbol

**Purpose:** Single source of truth for all biological targets

---

#### `chembl_target_dictionary`, `chembl_target_components`
**Purpose:** ChEMBL target information

**Data Stored:**
- ChEMBL target identifiers
- UniProt accession numbers
- Target classifications
- Component information (for multi-component targets)

**Relationship:** Linked to dm_target via chembl_tid

---

## 4. DRUG/PRODUCT DATA LAYER

#### `orange_book_products`
**Purpose:** FDA-approved drugs from the Orange Book

**Data Stored:**
- Application numbers (appl_no)
- Product numbers (product_no)
- Trade names and ingredients
- Dosage forms and routes
- Strengths and dosages
- Approval dates
- TE (therapeutic equivalence) codes
- Drug type (RX/OTC)
- Applicant information
- Reference Listed Drug (RLD) flags
- Reference Standard (RS) flags

**Key Columns:** id (PRIMARY), appl_no, product_no (UNIQUE combination)

**Purpose:** Regulatory approval status and therapeutic equivalence information

---

#### `dailymed_products`, `dailymed_sections`, `dailymed_section_names`
**Purpose:** FDA DailyMed drug label information

**Data Stored:**
- Product identifiers (set_id)
- Product names and brand names
- Generic and active substance names
- Section information (title_canonical, text)
- Section names with embeddings for semantic search
- Full-text searchable content

**Relationships:**
- dailymed_sections → dailymed_products (N:1, via set_id)
- dailymed_sections → dailymed_section_names (N:1, via section_name_id)

**Purpose:** Detailed drug label information for semantic search

---

## 5. REGULATORY/LABEL DATA LAYER

#### OpenFDA Label Tables

##### `set_ids`
**Purpose:** Master list of unique product identifiers

**Data Stored:**
- Unique set_id values from FDA records
- Creation timestamps

---

##### `labels_meta`
**Purpose:** Metadata about FDA drug labels

**Data Stored:**
- Document identifiers (doc_id)
- Product titles/names
- Update tracking

---

##### `sections` (Normalized)
**Purpose:** Full-text drug label content

**Data Stored:**
- Label text content
- Section-specific information
- Full-text search vectors (tsvector)

**Relationships:**
- Via set_id_id to set_ids (FOREIGN KEY)
- Via section_name_id to section_names (FOREIGN KEY)

---

##### `section_names`
**Purpose:** Canonical FDA label section classifications

**Data Stored:**
- Standard section names (indications_and_usage, adverse_reactions, etc.)
- Vector embeddings for semantic search (name_embedding)

**Purpose:** Standardize label structure across all products

---

##### Mapping Tables (`mapping_package_ndc`, `mapping_substance_name`, `mapping_rxcui`, etc.)
**Purpose:** Link products to various identifiers

**Data Stored:**
- Package NDC codes
- Substance names
- RxCui codes
- Brand names
- Generic names
- Active ingredients
- Applicant information

**Relationships:** All link to set_ids (via set_id_id FOREIGN KEY)

---

## 6. INTEGRATED MAPPING LAYER

### Clinical Trial to Molecular Mappings

#### `map_ctgov_molecules`
**Purpose:** Links clinical trials to specific molecules via intervention names

**Data Stored:**
- Trial reference (nct_id)
- Intervention reference (intervention_id → ctgov_interventions)
- Molecule reference (mol_id → dm_molecule)
- Concept reference (concept_id → dm_molecule_concept)
- Stereo reference (stereo_id → dm_molecule_stereo)
- Original intervention name (intervention_name)
- Matched synonym (matched_synonym) - the synonym that matched
- Match type (EXACT, FUZZY, SALT_STRIPPED, COMBO_PART, SYNONYM)
- Confidence score (0.0-1.0)
- Match method (EXACT_NAME, SYNONYM_MATCH, FUZZY_MATCH)
- Timestamps (created_at)

**Key Columns:** id (PRIMARY), intervention_id, mol_id (UNIQUE combination)

**Indexes:**
- `idx_map_ctgov_molecules_nct_id` - Trial lookup
- `idx_map_ctgov_molecules_mol_id` - Molecule lookup
- `idx_map_ctgov_molecules_concept_id` - Concept lookup
- `idx_map_ctgov_molecules_intervention_id` - Intervention lookup
- `idx_map_ctgov_molecules_confidence` - Confidence filtering

**Relationships:**
- Connects ctgov_interventions to dm_molecule
- Connects trials to drug concepts
- Links to stereo forms for precise matching

**Purpose:** Enable queries like "What trials use drug X?" or "What drugs are in trial Y?"

**Matching Strategy:**
1. Exact match on preferred name
2. Exact match on synonyms (dm_molecule_synonyms)
3. Fuzzy match with trigram similarity (threshold 0.6)
4. Salt-stripped matching (remove HCl, sodium, etc.)
5. Combination drug parsing (e.g., "Drug A + Drug B")

---

### Product to Molecular Mappings

#### `map_product_molecules`
**Purpose:** Links FDA products (OpenFDA, DailyMed, Orange Book) to molecular structures

**Data Stored:**
- Product identifier (set_id or product_id)
- Source table reference (source_table: OPENFDA, DAILYMED, ORANGE_BOOK)
- Molecule reference (mol_id → dm_molecule)
- Concept reference (concept_id → dm_molecule_concept)
- Original product/ingredient name (match_name)
- Match type (EXACT, FUZZY, INGREDIENT_MATCH)
- Confidence score (0.0-1.0)
- Timestamps (created_at)

**Key Columns:** id (PRIMARY), set_id, source_table, mol_id (UNIQUE combination)

**Indexes:**
- `idx_map_product_molecules_set_id` - Product lookup
- `idx_map_product_molecules_mol_id` - Molecule lookup
- `idx_map_product_molecules_concept_id` - Concept lookup

**Relationships:** Connects Orange Book, OpenFDA, and DailyMed products to molecules

**Purpose:** Bridge regulatory product databases to chemical structures for queries like "What FDA products contain molecule X?"

---

### Activity/Pharmacology Data

#### `dm_compound_target_activity` (Materialized View)
**Purpose:** Unified pharmacological activities across all sources for drug-target queries

**Data Stored:**
- Molecule information (mol_id, inchi_key, pref_name, canonical_smiles)
- Concept information (concept_id, preferred_name)
- Stereo information (stereo_id)
- Target information (target_id, gene_symbol, uniprot_id, target_name)
- Activity measurements:
  - assay_id, assay_type (BINDING, FUNCTIONAL, ADME)
  - activity_type (IC50, Ki, Kd, EC50)
  - activity_value_nm (normalized to nanomolar)
  - activity_units (original units)
- Potency values (pchembl_value) - standardized -log10(molar)
- Data quality (confidence_score, n_measurements)
- Source (CHEMBL, BINDINGDB, DRUGCENTRAL)

**Indexes:**
- `idx_dm_cta_mol_id` - Molecule queries
- `idx_dm_cta_concept_id` - Concept queries
- `idx_dm_cta_gene_symbol` - Target queries
- `idx_dm_cta_target_id` - Target ID lookup
- `idx_dm_cta_pchembl_value` - Potency filtering
- `idx_dm_cta_activity_type` - Activity type filtering

**Relationships:** Aggregates from:
- chembl.activities (via molecule_dictionary, target_dictionary)
- bindingdb_activities (via bindingdb_molecules, bindingdb_targets)
- dm_target (for unified target mapping)

**Purpose:** Single query source for all pharmacological data. Enables queries like:
- "What are the targets of drug X?"
- "What drugs hit target Y with pChEMBL > 7?"
- "Compare potency of drugs A, B, C on target Z"

**Refresh:** Run `REFRESH MATERIALIZED VIEW dm_compound_target_activity;` after data updates

---

## 7. SUMMARY OF KEY RELATIONSHIPS

### Hierarchical Relationships

```
Drug Concept (dm_molecule_concept)
├── parent_inchi_key_14 (connectivity only, no stereo, no salts)
├── preferred_name (e.g., "atorvastatin")
│
├── Stereo Forms (dm_molecule_stereo)
│   ├── parent_stereo_inchi_key (with stereo, no salts)
│   ├── stereo_type (ACHIRAL, DEFINED, RACEMIC, etc.)
│   │
│   └── Specific Molecules (dm_molecule)
│       ├── inchi_key (full, with salts)
│       ├── canonical_smiles
│       ├── is_salt, salt_form
│       ├── Synonyms (dm_molecule_synonyms)
│       ├── Activities (dm_compound_target_activity)
│       ├── Trial Mappings (map_ctgov_molecules)
│       └── Product Mappings (map_product_molecules)
│
└── Cross-db Links:
    ├── chembl_id → ChEMBL molecule_dictionary
    ├── drugcentral_id → drugcentral_drugs
    ├── bindingdb_monomer_id → bindingdb_molecules
    └── pubchem_cid → PubChem (external)
```

### 3-Level Molecular Hierarchy Example

```
Example: Atorvastatin
│
├── concept_id = 12345 (dm_molecule_concept)
│   └── parent_inchi_key_14 = "XUKUURHTRFLKSD" (connectivity only)
│
├── stereo_id = 67890 (dm_molecule_stereo)
│   └── parent_stereo_inchi_key = "XUKUURHTRFLKSD-UHFFFAOYSA" (with stereo)
│
└── mol_id = 111213 (dm_molecule)
    ├── inchi_key = "XUKUURHTRFLKSD-UHFFFAOYSA-M" (calcium salt)
    ├── is_salt = true
    └── salt_form = "CALCIUM"
```

### Trial-Related Relationships

```
ctgov_studies
├── ctgov_conditions (conditions studied)
├── ctgov_browse_conditions (standardized conditions)
├── ctgov_interventions (treatments used)
│   └── map_ctgov_molecules (linked to molecules)
├── ctgov_design_groups (study arms)
├── ctgov_design_outcomes (planned outcomes)
├── ctgov_outcome_measurements (measured outcomes)
├── ctgov_detailed_descriptions (full descriptions)
├── ctgov_brief_summaries (summaries)
├── rag_study_corpus (for semantic search)
└── rag_study_keys (indexed for fast retrieval)
```

### Molecular Structure Relationships

```
dm_molecule (specific form - Level 3)
│
├── HIERARCHY (upward links)
│   ├── stereo_id → dm_molecule_stereo (Level 2)
│   │   └── concept_id → dm_molecule_concept (Level 1)
│   └── concept_id → dm_molecule_concept (direct link)
│
├── NAMES
│   └── dm_molecule_synonyms (all names: brand, generic, research codes)
│
├── SOURCE DATA (where structure came from)
│   ├── chembl_id → chembl.compound_structures
│   ├── drugcentral_id → drugcentral_drugs
│   └── bindingdb_monomer_id → bindingdb_molecules
│
├── PHARMACOLOGY
│   ├── dm_compound_target_activity (activities and targets)
│   └── dm_target (via dm_compound_target_activity.target_id)
│
├── CLINICAL DATA
│   ├── map_ctgov_molecules → ctgov_interventions → ctgov_studies
│   └── map_product_molecules → set_ids/orange_book_products
│
└── STRUCTURE SEARCH (RDKit)
    ├── mol (RDKit mol object)
    ├── mfp2 (Morgan fingerprint, radius 2)
    └── ffp2 (Feature fingerprint)
```

### Product/Label Relationships

```
set_ids (product identifier)
├── labels_meta (product metadata)
├── sections (label text content)
│   └── section_names (standardized sections)
├── mapping_* (various identifiers)
├── orange_book_products (approval info)
└── dailymed_products (label info)
```

---

## 8. KEY DATA TYPES AND CONCEPTS

### RDKit-Based Normalization
- **parent_inchi_key_14**: Connectivity layer (first 14 chars) - groups all forms of same drug
- **parent_stereo_inchi_key**: Full InChI key with stereochemistry - groups same stereoisomer
- **parent_smiles**: Desalted, canonical SMILES of largest fragment

### RDKit PostgreSQL Extension (Optional)
When RDKit extension is installed, dm_molecule includes:
- **mol**: RDKit mol object for substructure/exact matching
- **mfp2**: Morgan fingerprint (radius 2) for similarity search
- **ffp2**: Feature fingerprint for similarity search

**Similarity Search:**
```sql
-- Tanimoto similarity (requires RDKit)
SELECT pref_name, tanimoto_sml(mfp2, morganbv_fp('CCO')) AS sim
FROM dm_molecule
WHERE mfp2 % morganbv_fp('CCO')  -- Uses GiST index
ORDER BY sim DESC LIMIT 10;

-- Substructure search
SELECT pref_name, canonical_smiles
FROM dm_molecule
WHERE mol @> 'c1ccccc1'::mol  -- Contains benzene ring
LIMIT 10;
```

### Stereochemistry Classification
- **ACHIRAL**: No chiral centers
- **DEFINED**: All chiral centers have defined stereochemistry
- **RACEMIC**: Chiral centers but not all defined
- **PARTIAL**: Some double bond stereo defined
- **UNKNOWN**: Stereo information unknown

### Salt Forms
- **FREE_BASE**: No salt
- **HYDROCHLORIDE**: HCl salt
- **PHOSPHATE**: Phosphate salt
- (And many others: SULFATE, ACETATE, TARTRATE, CITRATE, etc.)

### Activity Types
- **Ki**: Inhibitor constant (binding affinity)
- **IC50**: Half maximal inhibitory concentration
- **Kd**: Dissociation constant (binding affinity)
- **EC50**: Half maximal effective concentration

### pChEMBL Value
- Potency index: -log10(activity_value / 1e9)
- Higher values indicate higher potency
- Threshold >6 typically indicates significant activity

### Trial Phases
- **Phase 0**: Exploratory studies
- **Phase 1**: Safety and dosage
- **Phase 2**: Efficacy and side effects
- **Phase 3**: Efficacy and adverse reactions (large population)
- **Phase 4**: Post-marketing surveillance

---

## 9. FULL-TEXT SEARCH CAPABILITIES

### tsvector Columns
Multiple tables contain `tsvector` columns for full-text search:
- `ctgov_brief_summaries.description_vector`
- `ctgov_conditions.name_vector`
- `ctgov_interventions.description_vector`
- `dailymed_products.name_vector`
- `dailymed_sections.text_vector`
- `drugcentral_drugs.name_vector`, `synonyms_vector`, `cas_vector`
- `sections.text_vector` (OpenFDA labels)

### GIN Indexes
Generalized Inverted (GIN) indexes enable fast full-text search on all tsvector columns.

### Semantic Search
- `dailymed_section_names.name_embedding` - Sentence Transformer embeddings
- `rag_study_keys` - Indexed study metadata for semantic matching
- Vector similarity search via pgvector extension

---

## 10. QUERY PATTERNS

### Common Query Scenarios

**Find all trials for a drug:**
```sql
-- Via dm_molecule mapping
SELECT s.nct_id, s.brief_title, s.phase
FROM dm_molecule_concept c
JOIN map_ctgov_molecules map ON c.concept_id = map.concept_id
JOIN ctgov_studies s ON map.nct_id = s.nct_id
WHERE c.preferred_name ILIKE '%drug_name%';

-- Path: dm_molecule_concept → map_ctgov_molecules → ctgov_studies
```

**Get pharmacological targets for a drug:**
```sql
SELECT DISTINCT a.gene_symbol, a.pchembl_value, a.activity_type
FROM dm_molecule_concept c
JOIN dm_molecule m ON c.concept_id = m.concept_id
JOIN dm_compound_target_activity a ON m.mol_id = a.mol_id
WHERE c.preferred_name ILIKE '%drug_name%'
ORDER BY a.pchembl_value DESC;

-- Path: dm_molecule_concept → dm_molecule → dm_compound_target_activity
```

**Find structurally similar molecules (requires RDKit):**
```sql
SELECT m2.pref_name, m2.canonical_smiles,
       tanimoto_sml(m1.mfp2, m2.mfp2) AS similarity
FROM dm_molecule m1
JOIN dm_molecule m2 ON tanimoto_sml(m1.mfp2, m2.mfp2) > 0.7
WHERE m1.pref_name = 'query_molecule'
  AND m1.mol_id != m2.mol_id
ORDER BY similarity DESC;
```

**Get FDA approval status for a drug:**
```sql
SELECT c.preferred_name, ob.trade_name, ob.te_code, ob.approval_date
FROM dm_molecule_concept c
JOIN map_product_molecules pm ON c.concept_id = pm.concept_id
JOIN orange_book_products ob ON pm.set_id = ob.appl_no::text
WHERE c.preferred_name ILIKE '%drug_name%';
```

**Find label information:**
```sql
SELECT c.preferred_name, sec.section_name, sec.text
FROM dm_molecule_concept c
JOIN map_product_molecules pm ON c.concept_id = pm.concept_id
JOIN sections sec ON pm.set_id = sec.set_id_id::text
WHERE c.preferred_name ILIKE '%drug_name%'
  AND sec.section_name ILIKE '%adverse%';
```

**Match trials to conditions and treatments (via enriched search):**
```sql
-- Using enriched search table for fast flexible queries
SELECT nct_id, brief_title, phase, overall_status, enrollment
FROM ctgov_enriched_search
WHERE 'non-small cell lung cancer' = ANY(conditions)
  AND 'pembrolizumab' = ANY(interventions)
  AND phase = 'Phase 3'
  AND overall_status = 'Recruiting';

-- With trigram similarity
SELECT nct_id, brief_title, 
       similarity(conditions_text_norm, 'lung cancer') AS cond_sim
FROM ctgov_enriched_search
WHERE conditions_text_norm % 'lung cancer'
ORDER BY cond_sim DESC LIMIT 20;
```

---

## 11. INDEXING STRATEGY

### Primary Key Indexes
All tables have PRIMARY KEY indexes on id columns or composite keys.

### Foreign Key Indexes
All FOREIGN KEY relationships automatically have indexes for join performance.

### Custom Performance Indexes
- InChI keys (for structure deduplication)
- Gene symbols (for target queries)
- Trial status fields (for filtering)
- Normalized names (for fast lookups)
- Fingerprints (for similarity search with GiST)

---

## 12. DATA INTEGRITY CONSTRAINTS

### Uniqueness Constraints
- `dm_molecule(inchi_key)` - UNIQUE structures
- `dm_molecule_concept(parent_inchi_key_14)` - UNIQUE concepts
- `dm_molecule_stereo(parent_stereo_inchi_key)` - UNIQUE stereo forms
- `orange_book_products(appl_no, product_no)` - UNIQUE products
- `map_ctgov_molecules(intervention_id, mol_id)` - UNIQUE trial-molecule pairs

### Referential Integrity
- All FOREIGN KEY relationships have CASCADE or RESTRICT delete rules
- Most use ON DELETE CASCADE for data cleanup

### Generated Columns
- `synonym_lower` - Auto-generated lowercase for case-insensitive search
- `name_vector` - Auto-generated tsvector for full-text search
- `text_vector` - Auto-generated tsvector for label search

---

## 13. MATERIALIZED VIEWS

### `dm_compound_target_activity`
Pre-computed aggregation of all pharmacological activities.
Indexes on:
- gene_symbol (target queries)
- mol_id (molecule queries)
- concept_id (concept queries)
- pchembl_value (potency queries)

### `rag_study_keys`
Denormalized study metadata for fast semantic matching.
Indexes on normalized text fields with trigram search.

---

## 14. PERFORMANCE CONSIDERATIONS

### Tables > 1M Records
- ctgov_* tables (millions of trials, conditions, interventions)
- dm_molecule (hundreds of thousands of molecules)
- dm_molecule_synonyms (millions of synonyms)
- dm_compound_target_activity (millions of activities)

### Query Optimization Strategies
- Use concept_id for drug grouping (faster than joins)
- Leverage pChEMBL value indexes for potency filtering
- Use ngram/trigram indexes for flexible name matching
- Batch operations with execute_values for bulk loads
- Materialized views for pre-computed aggregations

---

## 15. METADATA TRACKING

### Timestamps
All main tables track:
- `created_at` - When record was first ingested
- `updated_at` - When record was last modified (via triggers)

### Source Tracking
- `dm_molecule.sources` - Array of sources (CHEMBL, DC, BDB)
- `map_ctgov_molecules.match_type` - How matching was done
- `map_product_molecules.match_type` - Mapping methodology

---

This schema enables comprehensive biomedical research by integrating:
- **Clinical trial design and outcomes**
- **Molecular structures and properties**
- **Pharmacological activity data**
- **Regulatory approval status**
- **Drug label information**
- **Semantic search across all data types**

