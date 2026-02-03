# Tools Documentation

This document provides a comprehensive overview of the tool architecture in the Advanced Biomedical Agent system. The tools are organized into two main categories:

1. **Retrieval-Based Tools** (`@search`) - Direct database access functions
2. **Agent-Based Tools** (`@tools`) - LLM-friendly wrappers with formatting and summarization

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Part 1: Retrieval-Based Tools](#part-1-retrieval-based-tools-search)
  - [1. Clinical Trials Search](#1-clinical-trials-search)
  - [2. Drug Labels Search](#2-drug-labels-search)
  - [3. Molecule-Trial Connectivity](#3-molecule-trial-connectivity)
  - [4. Adverse Events Search](#4-adverse-events-search)
  - [5. Outcomes Search](#5-outcomes-search)
  - [6. Orange Book Search](#6-orange-book-search)
  - [7. Cross-Database Lookup](#7-cross-database-lookup)
  - [8. Biotherapeutic Sequence Search](#8-biotherapeutic-sequence-search)
  - [9. Target/Drug Pharmacology Search](#9-targetdrug-pharmacology-search)
- [Part 2: Agent-Based Tools](#part-2-agent-based-tools-tools)
  - [Database Search Tools](#1-database-search-tools)
  - [Target/Drug Pharmacology Tools](#2-targetdrug-pharmacology-tools)
  - [Web Search Tools](#3-web-search-tools)
  - [Thinking Tool](#4-thinking-tool)
- [Tool Registration](#tool-registration)
- [Input Normalization](#input-normalization)
- [Output Summarization](#output-summarization)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Tool Collections](#tool-collections)
- [Quick reference (agent tools)](#quick-reference-agent-tools)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent-Based Tools (@tools)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • LLM-friendly formatting                            │  │
│  │  • Automatic summarization for long outputs           │  │
│  │  • Input normalization and validation                 │  │
│  │  • Error handling with helpful messages               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓ calls                            │
┌─────────────────────────────────────────────────────────────┐
│              Retrieval-Based Tools (@search)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Direct PostgreSQL database queries                 │  │
│  │  • Structured data models (Pydantic)                  │  │
│  │  • Async/await for performance                        │  │
│  │  • Raw search results                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Retrieval-Based Tools (`@search`)

**Location:** `src/bioagent/data/search/`

These are the **base search functions** that directly query the PostgreSQL database. They return structured data objects (Pydantic models) with raw search results.

### Characteristics

- **Direct database access**: Execute SQL queries against PostgreSQL
- **Structured outputs**: Return Pydantic models with typed fields
- **Async/await**: All functions are async for performance
- **No formatting**: Return raw data structures, not formatted strings
- **No summarization**: Return complete results without truncation
- **Input validation**: Use Pydantic for parameter validation

### Error handling (retrieval tools)

- **Invalid Literal/Enum values**: Passing an unsupported value for a Literal or Enum field (e.g. `mode`, `identifier_type`, `severity`, `outcome_type`, `biotherapeutic_type`) causes **ValidationError** at **input construction** when building the Pydantic model. Pydantic v2 error details typically include allowed values (e.g. "Input should be 'a' or 'b' or 'c'").
- **Invalid clinical trial enums**: For `ClinicalTrialsSearchInput`, custom validators for `status`, `phase`, `study_type`, and `intervention_type` raise **ValueError** (wrapped in **ValidationError**). The message includes **"Allowed: ..."** with the list of allowed values (e.g. "Invalid status 'x'. Allowed: Completed, Recruiting, ...").
- **Invalid search combination (DailyMed)**: `DailyMedAndOpenFDAInput` uses a model validator that raises **ValueError**; the message includes **"Valid patterns: ..."** with the four valid combinations (drug_names only; drug_names + fetch_all_sections=True; keyword_query only; drug_names and/or keyword_query with section_queries).
- **Unsupported mode**: If a mode-based tool returns `status="invalid_input"` for an unsupported mode, the `error` message includes **"Supported: ..."** with the list of modes (e.g. outcomes: outcomes_for_trial, trials_with_outcome, efficacy_comparison).
- **Missing required params for mode**: For tools with a `mode` and `validate_for_mode()` (molecule-trial, adverse events, outcomes, Orange Book, biotherapeutic, target search), if required parameters for the chosen mode are missing, the **async function does not raise**. It returns an output with `status="invalid_input"` and `error` describing what is missing (e.g. "'nct_id' is required for outcomes_for_trial").

### Available Retrieval Tools

#### 1. Clinical Trials Search
**File:** `clinical_trial_search.py`

**Function:** `clinical_trials_search_async(db_config: DatabaseConfig, search_input: ClinicalTrialsSearchInput) -> ClinicalTrialsSearchOutput`

**Purpose:** Search ClinicalTrials.gov data with flexible querying options.

**Key Features:**
- Multiple search strategies: combined, trigram, fulltext, exact (enum `SearchStrategy`: COMBINED, TRIGRAM, FULLTEXT, EXACT)
- Rich filtering: phase, status, dates, enrollment, eligibility, geography
- Supports pagination and sorting
- Returns trial metadata, eligibility criteria, outcomes, adverse events

**Input Model:** `ClinicalTrialsSearchInput`
- Search queries: `condition`, `intervention`, `keyword`, `nct_ids`, `sponsor`
- Filters: `phase`, `status`, `study_type`, `intervention_type`, `country`, etc.
- Date filters: `start_date_from`, `completion_date_to`, etc.
- Search options: `strategy`, `match_all`, `similarity_threshold`
- Output options: `sort_by`, `sort_order`, `limit`, `offset`
- Section controls: `output_eligibility`, `output_results`, `output_adverse_effects`, etc.

**Output Model:** `ClinicalTrialsSearchOutput`
- `status`: "success", "error", "not_found"
- `total_hits`: Total number of matching trials
- `hits`: List of `TrialSearchHit` objects
- `query_summary`: Human-readable summary of the query
- `filters_applied`: List of active filters
- `has_more`: Whether more results are available

**Data Sources:**
- `rag_study_search` - Trial metadata and search index
- `rag_study_corpus` - Full text content
- `ctgov_eligibilities` - Eligibility criteria
- `ctgov_outcomes` - Outcome data

---

#### 2. Drug Labels Search
**File:** `openfda_and_dailymed_search.py`

**Function:** `dailymed_and_openfda_search_async(db_config: DatabaseConfig, search_input: DailyMedAndOpenFDAInput) -> DailyMedAndOpenFDASearchOutput`

**Purpose:** Search FDA drug labels from OpenFDA and DailyMed databases.

**Key Features:**
- Search by drug name (brand or generic)
- Section-based retrieval (warnings, adverse reactions, drug interactions, etc.)
- Keyword search within labels
- Aggressive deduplication across sources

**Input Model:** `DailyMedAndOpenFDAInput`
- `drug_names`: List of drug names to search
- `section_queries`: Section names to match (list of strings)
- `keyword_query`: Keywords to search within labels (list of strings)
- `fetch_all_sections`: Boolean to retrieve all sections
- `result_limit`: Maximum number of results (default: 10, max: 50)
- Optional/tuning: `top_n_drugs`, `sections_per_query`, `aggressive_deduplication`
- Valid combinations (model validator): drug_names only; drug_names + fetch_all_sections=True; keyword_query only; drug_names and/or keyword_query with section_queries

**Output Model:** `DailyMedAndOpenFDASearchOutput`
- `status`: "success", "error", "not_found"
- `results`: List of `DailyMedAndOpenFDASearchResult` objects
- Each result contains:
  - `product_name`: Drug product name
  - `properties`: `DrugProperties` (smiles, formula, inchi_key, name)
  - `sections`: List of `SearchResultItem` (section_name, text, source)

**Data Sources:**
- `drugcentral_drugs` / `labels_meta` / section tables - DrugCentral labels
- `dailymed_products` / `dailymed_sections` / `dailymed_section_names` - DailyMed structured product labels
- DailyMed public SPL API for set IDs

---

#### 3. Molecule-Trial Connectivity
**File:** `molecule_trial_search.py`

**Function:** `molecule_trial_search_async(db_config: DatabaseConfig, search_input: MoleculeTrialSearchInput) -> MoleculeTrialSearchOutput`

**Purpose:** Search connections between molecules and clinical trials.

**Key Features:**
- Find trials linked to a molecule (by name or InChIKey)
- Find molecules associated with a condition
- Find trials linked to a target gene
- Structure-based: trials by SMILES/similarity or SMARTS/substructure

**Input Model:** `MoleculeTrialSearchInput`
- `mode`: "trials_by_molecule", "molecules_by_condition", "trials_by_target", "trials_by_structure", "trials_by_substructure"
- `molecule_name`: Molecule name for lookup
- `inchi_key`: InChIKey for precise molecule identification
- `condition`: Condition name for molecule discovery
- `target_gene`: Target gene symbol
- `smiles`, `smarts`: Structure input for trials_by_structure / trials_by_substructure
- `similarity_threshold`: For trials_by_structure (default: 0.7, 0.0–1.0)
- `min_pchembl`: Potency threshold (default: 6.0)
- `phase`, `status`: Optional trial filters
- `limit`, `offset`: Pagination (limit default: 50, max: 500)

**Output Model:** `MoleculeTrialSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found", "invalid_structure"
- `mode`: The search mode used
- `total_hits`: Total number of results
- `hits`: List of matching results (trials or molecules)
- `matched_molecules`, `structure_info`: For structure modes
- `query_summary`: Human-readable query description

**Data Sources:**
- `map_ctgov_molecules` - Molecule-trial mappings
- `dm_molecule_concept` - Molecule concepts and synonyms
- `rag_study_search` - Clinical trial data

---

#### 4. Adverse Events Search
**File:** `adverse_events_search.py`

**Function:** `adverse_events_search_async(db_config: DatabaseConfig, search_input: AdverseEventsSearchInput) -> AdverseEventsSearchOutput`

**Purpose:** Search adverse event data from ClinicalTrials.gov.

**Key Features:**
- Find top adverse events for a drug
- Find trials reporting a specific event
- Compare adverse events across multiple drugs

**Input Model:** `AdverseEventsSearchInput`
- `mode`: "events_for_drug", "drugs_with_event", "compare_safety"
- `drug_name`: Single drug name
- `event_term`: Adverse event term to search
- `drug_names`: List of drugs for comparison (required for compare_safety, at least 2)
- `severity`: Filter by severity. Allowed values: `"all"`, `"serious"`, `"other"` (non-serious)
- `min_subjects_affected`: Minimum number of affected subjects
- `limit`, `offset`: Pagination (limit default: 50, max: 500)

**Output Model:** `AdverseEventsSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found"
- `mode`: The search mode used
- `total_hits`: Total number of results
- `hits`: List of adverse event results (AdverseEventSummaryHit, AdverseEventTrialHit, or AdverseEventDrugComparison by mode)
- `query_summary`: Human-readable query description

**Data Sources:**
- `ctgov_reported_events` - Reported adverse event summaries
- `rag_study_search` - Trial metadata and intervention matching

---

#### 5. Outcomes Search
**File:** `outcomes_search.py`

**Function:** `outcomes_search_async(db_config: DatabaseConfig, search_input: OutcomesSearchInput) -> OutcomesSearchOutput`

**Purpose:** Search clinical trial outcomes, measurements, and statistical analyses.

**Key Features:**
- Get outcomes for a specific trial (NCT ID)
- Find trials with a specific outcome
- Compare efficacy across trials for a drug

**Input Model:** `OutcomesSearchInput`
- `mode`: "outcomes_for_trial", "trials_with_outcome", "efficacy_comparison"
- `nct_id`: Clinical trial identifier
- `outcome_keyword`: Keyword to search in outcome titles
- `drug_name`: Drug name for efficacy comparison
- `outcome_type`: Filter by type ("primary", "secondary", "all")
- `min_p_value`, `max_p_value`: Statistical significance filters
- `limit`, `offset`: Pagination (limit default: 50, max: 500)

**Output Model:** `OutcomesSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found"
- `mode`: The search mode used
- `total_hits`: Total number of results
- `hits`: List of outcome results (bundles for trial mode, individual outcomes otherwise)
- `query_summary`: Human-readable query description

**Data Sources:**
- `ctgov_outcomes` - Outcome definitions
- `ctgov_outcome_measurements` - Measurement data
- `ctgov_outcome_analyses` - Analysis and p-values
- `rag_study_search` - Trial metadata

---

#### 6. Orange Book Search
**File:** `orange_book_search.py`

**Function:** `orange_book_search_async(db_config: DatabaseConfig, search_input: OrangeBookSearchInput) -> OrangeBookSearchOutput`

**Purpose:** Search FDA Orange Book for products, TE codes, patents, and exclusivity.

**Key Features:**
- Search by drug name or ingredient
- Retrieve TE codes and product summaries
- Get patent and exclusivity information
- Lookup by NDA number

**Input Model:** `OrangeBookSearchInput`
- `mode`: "te_codes", "patents", "generics", "exclusivity" (at least one of drug_name, ingredient, or nda_number required)
- `drug_name`: Drug name to search
- `ingredient`: Active ingredient name
- `nda_number`: NDA application number
- `include_patents`: Include patent information (default: True)
- `include_exclusivity`: Include exclusivity data (default: True)
- `limit`, `offset`: Pagination (limit default: 50, max: 500)

**Output Model:** `OrangeBookSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found"
- `mode`: The search mode used
- `total_hits`: Total number of results
- `hits`: List of Orange Book entries (OrangeBookProductHit with patents/exclusivity)
- `query_summary`: Human-readable query description

**Data Sources:**
- `orange_book_products` - FDA Orange Book product data
- `orange_book_patents` - Patent information
- `orange_book_exclusivity` - Exclusivity data

---

#### 7. Cross-Database Lookup
**File:** `cross_db_lookup.py`

**Function:** `cross_database_lookup_async(db_config: DatabaseConfig, search_input: CrossDatabaseLookupInput) -> CrossDatabaseLookupOutput`

**Purpose:** Resolve drug identifiers across multiple databases.

**Key Features:**
- Lookup by any identifier type (ChEMBL ID, InChIKey, drug name, etc.)
- Auto-detect identifier type when `identifier_type="auto"`
- Return unified results from all databases

**Input Model:** `CrossDatabaseLookupInput`
- `identifier`: The identifier to resolve (required)
- `identifier_type`: "auto" (default) or specific type. Allowed values: `name`, `chembl`, `inchikey`, `ndc`, `rxcui`, `unii`, `auto`
- `include_labels`: Include drug label matches (default: True)
- `include_trials`: Include clinical trial matches (default: True)
- `include_targets`: Include target activity matches (default: True)
- `limit`: Maximum results per category (default: 20, max: 200)

**Output Model:** `CrossDatabaseLookupOutput`
- `status`: "success", "error", "invalid_input", "not_found"
- `identifier`: The identifier that was searched
- `identifier_type`: Detected or specified type
- `molecules`: List of `MoleculeIdentifierHit`
- `labels`: List of `LabelHit` (set_id, source, title)
- `trials`: List of `TrialHit`
- `targets`: List of `TargetHit`

**Data Sources:**
- `dm_molecule` / `dm_molecule_concept` / `dm_molecule_synonyms` - Molecule concepts
- `map_product_molecules` / `labels_meta` / `dailymed_products` - Drug labels
- `map_ctgov_molecules` / `rag_study_search` - Clinical trials
- `dm_compound_target_activity` - Target activities

---

#### 8. Biotherapeutic Sequence Search
**File:** `biotherapeutic_sequence_search.py`

**Function:** `biotherapeutic_sequence_search_async(db_config: DatabaseConfig, search_input: BiotherapeuticSearchInput) -> BiotherapeuticSearchOutput`

**Purpose:** Search biotherapeutics by sequence motif or target gene.

**Key Features:**
- Search by sequence motif
- Find similar biologics sharing a motif
- Find biotherapeutics linked to a target gene

**Input Model:** `BiotherapeuticSearchInput`
- `mode`: "by_sequence", "by_target", "similar_biologics"
- `sequence`: Amino acid sequence or motif (required for by_sequence and similar_biologics)
- `target_gene`: Target gene symbol (required for by_target)
- `biotherapeutic_type`: Filter by type. Allowed values: `"all"`, `"antibody"`, `"enzyme"`. Compared with `LOWER(biotherapeutic_type)` in the DB; stored values include mAb, Antibody, ADC, bispecific, Peptide, Oligonucleotide, Protein. So `"antibody"` matches only rows where the DB value lowercased is "antibody" (e.g. Antibody, not mAb/ADC/bispecific). `"enzyme"` may have few or no matches.
- `limit`, `offset`: Pagination (limit default: 50, max: 500)

**Output Model:** `BiotherapeuticSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found"
- `mode`: The search mode used
- `total_hits`: Total number of results
- `hits`: List of `BiotherapeuticHit` (with components)
- `query_summary`: Human-readable query description

**Data Sources:**
- `dm_biotherapeutic` - Biotherapeutic sequences
- `dm_biotherapeutic_component` - Component sequences (heavy/light chains, etc.)
- `dm_target` / `dm_target_uniprot_mappings` - Target gene mappings

---

#### 9. Target/Drug Pharmacology Search
**File:** `target_search.py`

**Function:** `target_search_async(db_config: DatabaseConfig, input: TargetSearchInput) -> TargetSearchOutput`  
(Internally uses `PharmacologySearch(db_config).search(input)`; `search` is **async**.)

**Purpose:** Unified pharmacology search for drug-target interactions, mechanisms, and molecular similarity.

**Key Features:**
- Multiple search modes (targets for drug, drugs for target, similarity, etc.)
- Integrates activity data (IC50, Ki, Kd) with curated mechanisms
- Structure-based searches (similarity, exact match, substructure)
- Drug comparison and selectivity analysis

**Input Model:** `TargetSearchInput`
- `mode`: Search mode enum (see SearchMode enum)
- `query`: Drug name or gene symbol (depending on mode)
- `smiles`: SMILES string for structure searches
- `smarts`: SMARTS pattern for substructure search
- `target`: Target gene symbol
- `off_targets`: List of off-target gene symbols
- `drug_names`: List of drug names for comparison
- `min_pchembl`: Minimum potency threshold
- `similarity_threshold`: Tanimoto similarity threshold
- `min_selectivity_fold`: Minimum fold-selectivity
- `data_source`: "both", "activity", or "mechanism"
- `activity_type`: Filter by activity type (IC50, Ki, etc.)
- `min_confidence`: Data quality filter
- `include_all_organisms`: Include non-human targets
- `include_trials`: Include clinical trial associations
- `include_forms`: Include salt/stereo forms
- `limit`: Maximum results

**Output Model:** `TargetSearchOutput`
- `status`: "success", "error", "invalid_input", "not_found", "invalid_structure"
- `mode`: SearchMode used
- `query_summary`: Human-readable query description
- `total_hits`: Total number of results
- `hits`: List of result objects (varies by mode):
  - `CompoundTargetProfile` - Complete target profile for a compound
  - `DrugForTargetHit` - Drug with activity/mechanism for a target
  - `DrugProfileResult` - Comprehensive drug profile
  - `ClinicalTrialHit` - Associated clinical trials
  - `MoleculeForm` - Salt/stereo forms
- `warnings`: List of warning messages
- `diagnostics`: Search diagnostics with suggestions
- `execution_time_ms`: Query execution time
- `structure_info`, `aggregated_summary`: For structure modes

**Search Modes (SearchMode enum):**
- `TARGETS_FOR_DRUG`, `TRIALS_FOR_DRUG`, `DRUG_PROFILE`, `DRUG_FORMS`
- `DRUGS_FOR_TARGET`, `SELECTIVE_DRUGS`
- `SIMILAR_MOLECULES`, `EXACT_STRUCTURE`, `SUBSTRUCTURE`
- `COMPARE_DRUGS`
- `ACTIVITIES_FOR_DRUG`, `ACTIVITIES_FOR_TARGET`
- `INDICATIONS_FOR_DRUG`, `DRUGS_FOR_INDICATION`
- `TARGET_PATHWAYS`, `DRUG_INTERACTIONS`

**Data Sources:**
- `dm_molecule` / `dm_molecule_concept` - Molecule data
- `dm_compound_target_activity` - Quantitative activity data (ChEMBL)
- `dm_mechanism_of_action` - Curated mechanism data (DrugCentral)
- `dm_target` / `dm_target_uniprot_mappings` - Target information
- `drugcentral_indications` - Indication data
- `drugcentral_drug_interactions` - Drug interaction data
- `map_ctgov_molecules` / `rag_study_search` - Clinical trial associations

---

## Part 2: Agent-Based Tools (`@tools`)

**Location:** `src/bioagent/agent/tools/`

These are **LLM-friendly wrappers** around the retrieval tools. They add formatting, summarization, input normalization, and error handling to make the tools easier for language models to use.

### Characteristics

- **LLM-friendly formatting**: Convert structured data to readable text
- **Automatic summarization**: Long outputs (>4000 chars) are summarized with full data stored
- **Input normalization**: Handle JSON strings, null values, wrapped dicts
- **Error handling**: Provide helpful error messages with suggestions
- **Output management**: Store full outputs and provide retrieval tools
- **Validation**: Additional input validation beyond Pydantic

### Tool Wrapper Architecture

#### Summarization Wrapper
**File:** `summarizing.py`

**Function:** `make_summarizing_tool(original_tool, summarizer_llm, output_dir) -> BaseTool`

**Purpose:** Wrap any tool to automatically summarize long outputs.

**How it works:**
1. Execute the original tool
2. If output > 4000 characters:
   - Generate a one-line summary
   - Store full output in a markdown file with metadata
   - Generate a detailed summary using LLM
   - Return summary with reference ID
3. If output ≤ 4000 characters:
   - Return output as-is

**Storage Format:**
- Files stored in: `research_outputs/{session_id}/{ref_id}.md` where `ref_id` has format `{tool_name}_{8-char-hex}` (e.g. `search_clinical_trials_a1b2c3d4`)
- Frontmatter includes: tool name, ref_id, timestamp, query params, size, one-line summary
- Body contains the full raw output

**Retrieval Tools:**
- `retrieve_full_output(reference_id, max_chars=None)` - Retrieve stored output. `reference_id` is the full string shown in the summary (e.g. after "ref: "), format `{tool_name}_{8-char-hex}` (e.g. `search_clinical_trials_a1b2c3d4`). `max_chars` is optional; when set, the returned string is truncated to that length.
- `list_research_outputs(tool_filter=None)` - List all stored outputs

---

### Defaults and value ranges

| Context | Parameter | Default | Range / max |
|---------|-----------|---------|------------|
| Clinical trials (agent) | `limit`, `offset` | 10, 0 | retrieval allows limit up to 500 |
| Clinical trials | `similarity_threshold` | 0.3 | 0.0–1.0 |
| Drug labels | `result_limit` | 10 | max 50 (agent caps at 50) |
| Molecule-trial | `min_pchembl` | 6.0 | — |
| Molecule-trial, adverse events, outcomes | `limit` | 20 | max 500 |
| Pharmacology (most tools) | `min_pchembl` | 5.0 | — |
| Pharmacology | `similarity_threshold` | 0.7 | 0.0–1.0 |
| Pharmacology | `limit` | 50 | max 500 |
| Cross-database lookup | `limit` | 20 | 1–200 |

---

### Agent vs retrieval parameter names

Some agent parameters use different names than the retrieval (Pydantic) input models; the wrapper maps them before calling the retrieval layer.

| Agent param | Retrieval field |
|-------------|-----------------|
| `query` | Mapped to `condition` / `intervention` / `keyword` depending on auto-detection (e.g. NCT ID → `nct_ids`) |
| `include_eligibility`, `include_results`, `include_adverse_events`, etc. | `output_eligibility`, `output_results`, `output_adverse_effects`, etc. |
| `molecule` | `molecule_name` |
| `outcome_term` | `outcome_keyword` |

Use agent parameter names when calling agent tools; use retrieval field names when calling retrieval functions directly.

---

### Available Agent Tools

#### 1. Database Search Tools
**File:** `dbsearch.py`

These tools wrap the retrieval-based search functions with LLM-friendly formatting.

##### `search_clinical_trials`
Wraps: `clinical_trials_search_async`

**Parameters (agent names; map to retrieval `output_*` / options):**
- Search: `query`, `condition`, `intervention`, `keyword`, `nct_ids`, `sponsor`
- Filters: `phase`, `status`, `study_type`, `intervention_type`, `country`, etc.
- Output sections (agent names): `include_eligibility`, `include_results`, `include_adverse_events`, `include_groups`, `include_baseline`, `include_sponsors`, `include_countries` — map to retrieval `output_eligibility`, `output_results`, `output_adverse_effects`, etc.
- Defaults: `limit=10`, `offset=0`; retrieval allows `limit` up to 500. `similarity_threshold` 0.0–1.0, default 0.3.

**Features:**
- Comprehensive parameter documentation in docstring
- Auto-detects NCT IDs from query string
- Parses comma-separated lists (phases, statuses, countries)
- Auto-enables `match_all=True` when multiple search criteria provided
- Formats results with brief or full details
- Handles date parsing and validation

**Output Format:**
- Brief mode: Summary lines with key info
- Full mode: Complete trial details with rendered summaries
- Includes match reasons, filters applied, pagination info

##### `search_drug_labels`
Wraps: `dailymed_and_openfda_search_async`

**Parameters:**
- `drug_names`, `section_queries`, `keyword_query`, `fetch_all_sections`, `result_limit` (default: 10, max: 50)
- `boxed_warnings_only`: If true, fetch only boxed warning sections
- `drug_interactions_only`: If true, fetch only drug interaction sections
- `adverse_reactions_only`: If true, fetch only adverse reaction sections

**Features:**
- Normalizes drug names to lists
- Handles section queries and keyword searches
- Formats results by drug and section
- Truncates very long sections (>1000 chars)

**Output Format:**
- Organized by drug product
- Shows properties (name, formula, SMILES)
- Lists matching sections with source

##### `search_molecule_trials`
Wraps: `molecule_trial_search_async`

**Parameters:** Agent uses `molecule` (maps to retrieval `molecule_name`), plus `mode`, `inchi_key`, `condition`, `target_gene`, `min_pchembl` (default: 6.0), `phase`, `limit` (default: 20, max: 500), `offset`.

**Required per mode:**
- `trials_by_molecule`: `molecule` or `inchi_key`
- `molecules_by_condition`: `condition`
- `trials_by_target`: `target_gene`

**Features:**
- Supports three search modes
- Parses phase lists from strings
- Formats results based on mode

**Output Format:**
- Mode-specific formatting
- Shows match type and confidence for molecule-trial links

##### `search_adverse_events`
Wraps: `adverse_events_search_async`

**Parameters:** `mode`, `drug_name`, `event_term`, `drug_names`, `severity` ("all" | "serious" | "other"), `min_subjects_affected`, `limit` (default: 20, max: 500), `offset`.

**Required per mode:**
- `events_for_drug`: `drug_name`
- `drugs_with_event`: `event_term`
- `compare_safety`: `drug_names` (at least 2)

**Features:**
- Three search modes with different output formats
- Normalizes drug names to lists

**Output Format:**
- Mode-specific: events for drug, drugs with event, or comparison

##### `search_trial_outcomes`
Wraps: `outcomes_search_async`

**Parameters:** Agent uses `outcome_term` (maps to retrieval `outcome_keyword`), plus `mode`, `nct_id`, `drug_name`, `outcome_type`, `min_p_value`, `max_p_value`, `limit`, `offset`.

**Required per mode:**
- `outcomes_for_trial`: `nct_id`
- `trials_with_outcome`: `outcome_term`
- `efficacy_comparison`: `drug_name`

**Features:**
- Three search modes
- Formats outcomes, measurements, and analyses

**Output Format:**
- Structured by trial or outcome depending on mode

##### `search_orange_book`
Wraps: `orange_book_search_async`

**Parameters:** `mode` ("te_codes" | "patents" | "generics" | "exclusivity"), `drug_name`, `ingredient`, `nda_number`, `include_patents`, `include_exclusivity`, `limit`, `offset`.

**Required:** At least one of `drug_name`, `ingredient`, or `nda_number`.

**Features:**
- Multiple search modes
- Formats patents and exclusivity data

**Output Format:**
- Product information with patent and exclusivity details

##### `lookup_drug_identifiers`
Wraps: `cross_database_lookup_async`

**Features:**
- Cross-database identifier resolution
- Auto-detects identifier type

**Output Format:**
- Unified results from all databases (molecules, labels, trials, targets)

##### `search_biotherapeutics`
Wraps: `biotherapeutic_sequence_search_async`

**Parameters:** `mode` ("by_sequence" | "by_target" | "similar_biologics"), `sequence`, `target_gene`, `biotherapeutic_type` ("all" | "antibody" | "enzyme"), `limit`, `offset`.

**Required per mode:**
- `by_sequence` or `similar_biologics`: `sequence`
- `by_target`: `target_gene`

**Features:**
- Sequence and target-based searches
- Formats component information

**Output Format:**
- Biotherapeutic details with component sequences

---

#### 2. Target/Drug Pharmacology Tools
**File:** `target_search.py`

These tools wrap the `PharmacologySearch` class with enhanced formatting and validation.

##### `search_drug_targets`
**Purpose:** Find all protein targets for a drug.

**Parameters:**
- `drug_name`: Drug name (brand, generic, or synonym)
- `min_pchembl`: Potency threshold (default: 5.0)
- `data_source`: "both", "activity", or "mechanism"
- `include_all_organisms`: Include non-human targets
- `limit`: Maximum results

**Output Format:**
- Compound profiles with activities and mechanisms
- Shows gene symbols, activity values, pChEMBL, confidence
- Includes SMILES and target names

##### `search_target_drugs`
**Purpose:** Find all drugs that hit a target.

**Parameters:**
- `gene_symbol`: Target gene symbol (HGNC)
- `min_pchembl`: Potency threshold
- `data_source`: "both", "activity", or "mechanism"
- `limit`: Maximum results

**Output Format:**
- Drugs with activity data and mechanisms
- Shows selectivity information when available

##### `search_similar_molecules`
**Purpose:** Find structurally similar molecules.

**Parameters:**
- `smiles`: SMILES string
- `similarity_threshold`: Tanimoto similarity (0.0-1.0)
- `min_pchembl`: Potency threshold
- `limit`: Maximum results

**Output Format:**
- Similar molecules with similarity scores and activity data

##### `search_exact_structure`
**Purpose:** Identify a molecule from SMILES.

**Parameters:**
- `smiles`: SMILES string
- `min_pchembl`: Potency threshold

**Output Format:**
- Compound identification with activity and mechanism data

##### `search_substructure`
**Purpose:** Find molecules containing a substructure.

**Parameters:**
- `pattern`: SMILES or SMARTS pattern
- `limit`: Maximum results

**Output Format:**
- Molecules containing the substructure

##### `get_drug_profile`
**Purpose:** Get comprehensive drug profile.

**Parameters:**
- `drug_name`: Drug name
- `include_trials`: Include clinical trials
- `include_forms`: Include salt/stereo forms
- `min_pchembl`: Potency threshold

**Output Format:**
- Complete drug profile with forms, targets, mechanisms, trials

##### `get_drug_forms`
**Purpose:** Get all molecular forms of a drug.

**Parameters:**
- `drug_name`: Drug name
- `limit`: Maximum results

**Output Format:**
- List of salt/stereo forms with identifiers

##### `search_drug_trials`
**Purpose:** Find clinical trials for a drug.

**Parameters:**
- `drug_name`: Drug name
- `limit`: Maximum results

**Output Format:**
- Clinical trials with NCT IDs, phases, status

##### `compare_drugs_on_target`
**Purpose:** Compare multiple drugs on a target.

**Parameters:**
- `target`: Gene symbol
- `drug_names`: List of drug names (at least 2)

**Output Format:**
- Comparison table with relative potency and fold-differences

##### `search_selective_drugs`
**Purpose:** Find selective drugs.

**Parameters:**
- `target`: Primary target gene symbol
- `off_targets`: List of off-target gene symbols
- `min_selectivity_fold`: Minimum fold-selectivity
- `min_pchembl`: Minimum potency
- `limit`: Maximum results

**Output Format:**
- Selective compounds with selectivity ratios

##### `pharmacology_search`
**Purpose:** Unified pharmacology search tool.

The unified `pharmacology_search` tool supports **only** the following `search_type` values: `drug_targets`, `target_drugs`, `drug_profile`, `drug_forms`, `drug_trials`, `similar_molecules`, `exact_structure`, `substructure`, `compare_drugs`, `selective_drugs`. Use the standalone tools below for the other capabilities.

**Parameters:**
- `search_type`: One of the values above (e.g. "drug_targets", "target_drugs")
- Additional parameters depend on search_type (e.g. `drug_name`, `gene_symbol`, `smiles`, `drug_names`, `off_targets`)
- `min_pchembl` (default: 5.0), `similarity_threshold` (default: 0.7), `data_source`, `limit` (default: 50, max: 500)

**Features:**
- Single entry point for the supported pharmacology searches
- Validates required parameters per search type
- Provides helpful error messages

##### Additional Pharmacology Tools (standalone only; not available as `search_type` in `pharmacology_search`)
- `search_drug_activities` - Get activity measurements for a drug
- `search_target_activities` - Get activity measurements for a target
- `search_drug_indications` - Find indications for a drug
- `search_indication_drugs` - Find drugs for an indication
- `search_target_pathways` - Find pathway annotations for a target
- `search_drug_interactions` - Find drug-drug interactions

---

#### 3. Web Search Tools
**File:** `web_search.py`

These tools provide web search capabilities for general information.

##### `web_search`
**Purpose:** General web search using DuckDuckGo. No API key required.

**Parameters:**
- `query`: Search query

**Output Format:**
- Formatted list of results with title, link, and snippet

##### `tavily_search`
**Purpose:** Web search using Tavily API. Requires `TAVILY_API_KEY` environment variable.

**Parameters:**
- `query`: Search query
- `search_depth`: "basic" (fast) or "advanced" (more thorough). Default: "basic"
- `include_answer`: Include direct answer (default: true)
- `max_results`: Maximum results (default: 3)
- `include_raw_content`: Include raw scraped content
- `include_images`: Include image URLs

**Output Format:**
- Query, answer, and formatted results

##### `scrape_url_content`
**Purpose:** Scrape content from URLs.

**Parameters:**
- `urls`: A single URL or a list of URLs (normalized to list internally)

**Output Format:**
- Scraped text content from pages

---

#### 4. Thinking Tool
**File:** `thinking.py`

##### `think`
**Purpose:** Record thinking and reasoning steps.

**Parameters:**
- `thought`: Thinking, reasoning, or plan

**Features:**
- Tracks thinking usage for analytics
- Returns confirmation message

---

## Tool Registration

### Retrieval Tools
Retrieval tools are **not directly registered** with agents. They are called by agent tools.

### Agent Tools
Agent tools are registered via the `get_summarized_tools()` function:

```python
from bioagent.agent.tools import get_summarized_tools

tools = get_summarized_tools(
    summarizer_llm=llm,
    session_id=session_id
)
```

This function:
1. Collects all base tools from:
   - `DBSEARCH_TOOLS` (clinical trials, drug labels, etc.)
   - `TARGET_SEARCH_TOOLS` (pharmacology tools)
   - `WEB_SEARCH_TOOLS` (web search tools)
2. Wraps each with `make_summarizing_tool()`
3. Adds retrieval tools: `retrieve_full_output` and `list_research_outputs`
4. Returns the complete list of tools for agent registration

---

## Input Normalization

All agent tools use the `@robust_unwrap_llm_inputs` decorator which handles:

1. **JSON strings**: `'["a", "b"]'` → `["a", "b"]`
2. **Null strings**: `"null"`, `"None"` → `None`
3. **Wrapped dicts**: `{"value": [...]}` → `[...]`
4. **Single values**: `"aspirin"` → `["aspirin"]` (when list expected)
5. **Boolean strings**: `"true"` → `True`, `"false"` → `False`

This makes tools more robust to LLM output variations.

---

## Output Summarization

When a tool output exceeds 4000 characters:

1. **Store**: Full output saved to `research_outputs/{session_id}/{tool_name}_{ref_id}.md`
2. **Summarize**: LLM generates a summary focused on the agent's query
3. **Return**: Summary with reference ID and instructions to retrieve full output

**Benefits:**
- Prevents token limit issues
- Keeps agent context focused
- Allows retrieval of full data when needed
- Tracks all research outputs for review

---

## Error Handling

Agent tools provide helpful error messages:

1. **Input validation errors**: Clear messages about required parameters
2. **Search errors**: Suggestions for fixing queries
3. **Not found errors**: Guidance on alternative search terms
4. **Diagnostics**: Search diagnostics with suggestions when available

Example error format:
```
✗ Error searching drug targets: ValueError: Drug not found

Input: drug_name='imatinib', min_pchembl=5.0

Suggestions:
  → Check drug name spelling
  → Try generic name instead of brand name
  → For biologics, try the INN name
```

---

## Best Practices

### For Retrieval Tools (`@search`)
- Keep focused on data retrieval
- Return structured Pydantic models
- Use async/await for performance
- Include comprehensive docstrings
- Validate inputs with Pydantic

### For Agent Tools (`@tools`)
- Add LLM-friendly formatting
- Provide comprehensive docstrings with examples
- Handle edge cases gracefully
- Normalize inputs robustly
- Return helpful error messages
- Use summarization for long outputs

### For Tool Usage
- Use agent tools in agent workflows
- Use retrieval tools for programmatic access
- Check tool docstrings for parameter details
- Use `retrieve_full_output()` for detailed analysis
- Use `list_research_outputs()` to review gathered data

---

## Tool Collections

### `DBSEARCH_TOOLS`
All database search tools:
- `search_clinical_trials`
- `search_drug_labels`
- `search_molecule_trials`
- `search_adverse_events`
- `search_trial_outcomes`
- `search_orange_book`
- `lookup_drug_identifiers`
- `search_biotherapeutics`

### `TARGET_SEARCH_TOOLS`
All pharmacology search tools:
- `search_drug_targets`
- `search_target_drugs`
- `search_similar_molecules`
- `search_exact_structure`
- `search_substructure`
- `get_drug_profile`
- `get_drug_forms`
- `search_drug_trials`
- `compare_drugs_on_target`
- `search_selective_drugs`
- Plus additional activity, indication, pathway, and interaction tools

### `WEB_SEARCH_TOOLS`
All web search tools:
- `web_search`
- `tavily_search`
- `scrape_url_content`

---

## Quick reference (agent tools)

| Tool | Main parameters | Required per mode |
|------|-----------------|-------------------|
| `search_clinical_trials` | query, condition, intervention, keyword, nct_ids, phase, status, limit, include_* | See [search_clinical_trials](#search_clinical_trials) |
| `search_drug_labels` | drug_names, section_queries, keyword_query, fetch_all_sections, boxed_warnings_only, drug_interactions_only, adverse_reactions_only, result_limit | — |
| `search_molecule_trials` | mode, molecule, inchi_key, condition, target_gene | trials_by_molecule: molecule or inchi_key; molecules_by_condition: condition; trials_by_target: target_gene |
| `search_adverse_events` | mode, drug_name, event_term, drug_names | events_for_drug: drug_name; drugs_with_event: event_term; compare_safety: drug_names (≥2) |
| `search_trial_outcomes` | mode, nct_id, outcome_term, drug_name | outcomes_for_trial: nct_id; trials_with_outcome: outcome_term; efficacy_comparison: drug_name |
| `search_orange_book` | mode, drug_name, ingredient, nda_number | At least one of drug_name, ingredient, nda_number |
| `lookup_drug_identifiers` | identifier, identifier_type (name/chembl/inchikey/ndc/rxcui/unii/auto) | identifier |
| `search_biotherapeutics` | mode, sequence, target_gene | by_sequence/similar_biologics: sequence; by_target: target_gene |
| `pharmacology_search` | search_type, drug_name, gene_symbol, smiles, drug_names, off_targets | See [pharmacology_search](#pharmacology_search) |
| `retrieve_full_output` | reference_id (e.g. search_clinical_trials_a1b2c3d4), max_chars | reference_id |
| `list_research_outputs` | tool_filter | — |
| `web_search` | query | query |
| `tavily_search` | query, search_depth, max_results (default 3) | query; TAVILY_API_KEY env |
| `scrape_url_content` | urls (single or list) | urls |

---

## Summary

- **Retrieval tools** (`@search`) provide direct database access with structured outputs
- **Agent tools** (`@tools`) wrap retrieval tools with formatting, summarization, and LLM-friendly interfaces
- **Summarization** automatically handles long outputs by storing full data and returning summaries
- **Input normalization** makes tools robust to LLM output variations
- **Error handling** provides helpful guidance for fixing issues
- **Tool registration** is handled via `get_summarized_tools()` which wraps all tools and adds retrieval utilities

This architecture separates concerns: retrieval tools focus on data access, while agent tools focus on LLM usability.
