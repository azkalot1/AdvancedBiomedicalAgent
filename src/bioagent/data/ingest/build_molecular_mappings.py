#!/usr/bin/env python3
"""
Molecular Mapping Pipeline with RDKit-based Normalization.

Version: 2.2
Features:
  - Idempotent: Can be re-run safely to update/replace all tables and views
  - 3-level molecule hierarchy (Concept → Stereo → Molecule)
  - RDKit-based structure normalization
  - Deduplicated activity summary table
  - Comprehensive search functions
  - Data validation and audit logging
  - Fuzzy matching for clinical trials
  - Progress tracking and reporting
  - Fast/debug mode for testing

Prerequisites:
  - PostgreSQL with RDKit extension installed
  - All source tables (chembl_*, bindingdb_*, drugcentral_*, ctgov_*) must exist
  - dm_target must exist (from your previous scripts)

Usage:
  python build_molecular_mappings.py                    # Full rebuild
  python build_molecular_mappings.py --fast             # Fast debug mode (10k molecules)
  python build_molecular_mappings.py --limit 50000      # Custom limit
  python build_molecular_mappings.py --incremental      # Incremental update
  python build_molecular_mappings.py --refresh-only     # Refresh views only
  python build_molecular_mappings.py --report           # Generate report
  python build_molecular_mappings.py --validate-only    # Run validation only
"""

from __future__ import annotations

import re
import sys
import time
import argparse
import psycopg2
import psycopg2.extras
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Handle imports
try:
    from .config import DatabaseConfig, get_connection, DEFAULT_CONFIG
except ImportError:
    from config import DatabaseConfig, get_connection, DEFAULT_CONFIG


# ============================================================================
# DEFAULT LIMITS FOR FAST MODE
# ============================================================================

FAST_MODE_LIMITS = {
    'chembl': 10000,
    'drugcentral': 2000,
    'bindingdb': 5000,
    'synonyms': 50000,
    'interventions': 10000,
    'fuzzy_interventions': 5000,
}


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class PrerequisiteError(PipelineError):
    """Raised when prerequisites are not met."""
    pass


# ============================================================================
# AUDIT LOGGING
# ============================================================================

def log_audit(cur, phase: str, operation: str, rows_affected: int, 
              duration_seconds: float, status: str, error_message: Optional[str] = None):
    """Log pipeline operations for audit trail."""
    try:
        cur.execute("""
            INSERT INTO dm_pipeline_audit (phase, operation, rows_affected, duration_seconds, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (phase, operation, rows_affected, round(duration_seconds, 2), status, error_message))
    except psycopg2.Error:
        pass  # Don't fail if audit logging fails


def safe_execute(cur, sql: str, params=None, description: str = "query") -> int:
    """Execute SQL with proper error handling."""
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur.rowcount
    except psycopg2.Error as e:
        raise PipelineError(f"Failed to execute {description}: {e.pgerror or str(e)}")


# ============================================================================
# PREREQUISITES CHECK
# ============================================================================

def check_rdkit_extension(config: DatabaseConfig) -> bool:
    """Verify RDKit extension is available and working."""
    with get_connection(config) as con:
        with con.cursor() as cur:
            try:
                cur.execute("SELECT mol_from_smiles('C'::cstring) IS NOT NULL")
                return cur.fetchone()[0]
            except Exception:
                return False


def check_prerequisites(config: DatabaseConfig) -> Tuple[bool, List[str]]:
    """Verify all required source tables exist before running pipeline."""
    required_tables = [
        ('compound_structures', 'ChEMBL'),
        ('molecule_dictionary', 'ChEMBL'),
        ('molecule_synonyms', 'ChEMBL'),
        ('activities', 'ChEMBL'),
        ('assays', 'ChEMBL'),
        ('drugcentral_drugs', 'DrugCentral'),
        ('bindingdb_molecules', 'BindingDB'),
        ('bindingdb_activities', 'BindingDB'),
        ('ctgov_interventions', 'ClinicalTrials.gov'),
        ('ctgov_studies', 'ClinicalTrials.gov'),
        ('dm_target', 'Target Integration'),
    ]
    
    missing = []
    with get_connection(config) as con:
        with con.cursor() as cur:
            for table, source in required_tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                if not cur.fetchone()[0]:
                    missing.append(f"{table} ({source})")
    
    return len(missing) == 0, missing


def get_source_counts(config: DatabaseConfig) -> Dict[str, int]:
    """Get counts from source tables for progress tracking."""
    counts = {}
    with get_connection(config) as con:
        with con.cursor() as cur:
            queries = {
                'chembl': "SELECT COUNT(*) FROM compound_structures WHERE standard_inchi_key IS NOT NULL AND canonical_smiles IS NOT NULL",
                'drugcentral': "SELECT COUNT(*) FROM drugcentral_drugs WHERE inchi_key IS NOT NULL AND smiles IS NOT NULL",
                'bindingdb': "SELECT COUNT(*) FROM bindingdb_molecules WHERE inchi_key IS NOT NULL AND smiles IS NOT NULL",
                'interventions': "SELECT COUNT(*) FROM ctgov_interventions WHERE intervention_type IN ('DRUG', 'BIOLOGICAL', 'DIETARY SUPPLEMENT')",
            }
            for key, query in queries.items():
                try:
                    cur.execute(query)
                    counts[key] = cur.fetchone()[0]
                except Exception:
                    counts[key] = 0
    return counts


# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

CREATE_EXTENSIONS_SQL = """
CREATE EXTENSION IF NOT EXISTS rdkit;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""

CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dm_pipeline_audit (
    id BIGSERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT NOW(),
    phase TEXT,
    operation TEXT,
    rows_affected BIGINT,
    duration_seconds NUMERIC,
    status TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON dm_pipeline_audit(run_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_phase ON dm_pipeline_audit(phase);
"""

DROP_SCHEMA_SQL = """
-- Drop in reverse dependency order
DROP TABLE IF EXISTS map_openfda_molecules CASCADE;
DROP TABLE IF EXISTS map_dailymed_molecules CASCADE;
DROP TABLE IF EXISTS dm_drug_indication CASCADE;
DROP TABLE IF EXISTS dm_indication CASCADE;
DROP TABLE IF EXISTS map_product_molecules CASCADE;
DROP TABLE IF EXISTS map_ctgov_molecules CASCADE;
DROP TABLE IF EXISTS dm_molecule_synonyms CASCADE;
DROP TABLE IF EXISTS dm_molecule_target_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS dm_compound_target_activity CASCADE;
DROP TABLE IF EXISTS dm_molecule CASCADE;
DROP TABLE IF EXISTS dm_molecule_stereo CASCADE;
DROP TABLE IF EXISTS dm_molecule_concept CASCADE;
"""

CREATE_SCHEMA_SQL = """
-- ============================================================================
-- LEVEL 1: Drug Concepts (Therapeutic equivalence - groups all forms)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_molecule_concept (
    concept_id BIGSERIAL PRIMARY KEY,
    
    -- The "flat" InChI key connectivity layer (first 14 chars, no stereo, no salts)
    parent_inchi_key_14 TEXT UNIQUE NOT NULL,
    
    -- Representative structure (any canonical parent)
    parent_smiles TEXT,
    
    -- Representative name (most common across forms)
    preferred_name TEXT,
    
    -- External drug concept identifiers
    rxnorm_cui TEXT,
    drugbank_id TEXT,
    unii TEXT,
    
    -- Metadata
    has_stereo_variants BOOLEAN DEFAULT FALSE,
    has_salt_forms BOOLEAN DEFAULT FALSE,
    n_forms INTEGER DEFAULT 1,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concept_inchi14 ON dm_molecule_concept(parent_inchi_key_14);
CREATE INDEX IF NOT EXISTS idx_concept_name ON dm_molecule_concept(preferred_name);
CREATE INDEX IF NOT EXISTS idx_concept_name_trgm ON dm_molecule_concept USING gin(preferred_name gin_trgm_ops);


-- ============================================================================
-- LEVEL 2: Stereo Forms (API-level identity - specific stereochemistry)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_molecule_stereo (
    stereo_id BIGSERIAL PRIMARY KEY,
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id) ON DELETE CASCADE,
    
    -- Full parent InChI key (with stereo, desalted)
    parent_stereo_inchi_key TEXT UNIQUE NOT NULL,
    parent_stereo_smiles TEXT,
    
    -- Stereochemistry classification
    stereo_type TEXT,  -- 'ACHIRAL', 'DEFINED', 'GEOMETRIC', 'UNKNOWN'
    n_chiral_centers INTEGER DEFAULT 0,
    n_defined_centers INTEGER DEFAULT 0,
    
    -- Is this the main Active Pharmaceutical Ingredient form?
    is_api BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stereo_concept ON dm_molecule_stereo(concept_id);
CREATE INDEX IF NOT EXISTS idx_stereo_inchi ON dm_molecule_stereo(parent_stereo_inchi_key);
CREATE INDEX IF NOT EXISTS idx_stereo_type ON dm_molecule_stereo(stereo_type);


-- ============================================================================
-- LEVEL 3: Specific Molecule Forms (Original structures - salts, hydrates, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_molecule (
    mol_id BIGSERIAL PRIMARY KEY,
    
    -- Original Structure Identity
    inchi_key TEXT UNIQUE NOT NULL,
    standard_inchi TEXT,
    canonical_smiles TEXT,
    pref_name TEXT,
    
    -- RDKit mol object (for searches)
    mol mol,
    
    -- Hierarchy links
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id),
    stereo_id BIGINT REFERENCES dm_molecule_stereo(stereo_id),
    
    -- Normalization results (computed by RDKit)
    parent_inchi_key_14 TEXT,
    parent_stereo_inchi_key TEXT,
    parent_smiles TEXT,
    
    -- Salt/Form information
    is_salt BOOLEAN DEFAULT FALSE,
    salt_form TEXT,
    n_components INTEGER DEFAULT 1,
    
    -- Stereochemistry info
    stereo_type TEXT,
    n_chiral_centers INTEGER DEFAULT 0,
    n_defined_centers INTEGER DEFAULT 0,
    
    -- Cross-database References
    chembl_id TEXT,
    drugcentral_id INTEGER,
    bindingdb_monomer_id INTEGER,
    pubchem_cid INTEGER,
    
    -- Fingerprints for similarity search
    mfp2 bfp,
    ffp2 bfp,
    
    -- Provenance
    sources TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dm_mol_inchikey ON dm_molecule(inchi_key);
CREATE INDEX IF NOT EXISTS idx_dm_mol_parent14 ON dm_molecule(parent_inchi_key_14);
CREATE INDEX IF NOT EXISTS idx_dm_mol_parent_stereo ON dm_molecule(parent_stereo_inchi_key);
CREATE INDEX IF NOT EXISTS idx_dm_mol_concept ON dm_molecule(concept_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_stereo ON dm_molecule(stereo_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_pref_name ON dm_molecule(pref_name);
CREATE INDEX IF NOT EXISTS idx_dm_mol_pref_name_trgm ON dm_molecule USING gin(pref_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dm_mol_chembl ON dm_molecule(chembl_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_drugcentral ON dm_molecule(drugcentral_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_bindingdb ON dm_molecule(bindingdb_monomer_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_salt ON dm_molecule(is_salt);
CREATE INDEX IF NOT EXISTS idx_dm_mol_mfp2 ON dm_molecule USING gist(mfp2);
CREATE INDEX IF NOT EXISTS idx_dm_mol_ffp2 ON dm_molecule USING gist(ffp2);
CREATE INDEX IF NOT EXISTS idx_dm_mol_mol ON dm_molecule USING gist(mol);


-- ============================================================================
-- Synonym Lookup Table (The Dictionary)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_molecule_synonyms (
    id BIGSERIAL PRIMARY KEY,
    mol_id BIGINT REFERENCES dm_molecule(mol_id) ON DELETE CASCADE,
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id) ON DELETE CASCADE,
    synonym TEXT NOT NULL,
    synonym_lower TEXT GENERATED ALWAYS AS (lower(synonym)) STORED,
    syn_type TEXT,
    source TEXT,
    
    UNIQUE(mol_id, synonym)
);

CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_lower ON dm_molecule_synonyms(synonym_lower);
CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_lower_trgm ON dm_molecule_synonyms USING gin(synonym_lower gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_concept ON dm_molecule_synonyms(concept_id);
CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_mol ON dm_molecule_synonyms(mol_id);


-- ============================================================================
-- Clinical Trial Mapping
-- ============================================================================
CREATE TABLE IF NOT EXISTS map_ctgov_molecules (
    id BIGSERIAL PRIMARY KEY,
    nct_id TEXT NOT NULL,
    intervention_id INTEGER NOT NULL,
    match_name TEXT,
    mol_id BIGINT REFERENCES dm_molecule(mol_id),
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id),
    match_type TEXT,  -- 'EXACT', 'SALT_STRIPPED', 'COMBO_PART', 'FUZZY'
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(intervention_id, mol_id)
);

CREATE INDEX IF NOT EXISTS idx_map_ctgov_mol ON map_ctgov_molecules(mol_id);
CREATE INDEX IF NOT EXISTS idx_map_ctgov_concept ON map_ctgov_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_map_ctgov_nct ON map_ctgov_molecules(nct_id);
CREATE INDEX IF NOT EXISTS idx_map_ctgov_type ON map_ctgov_molecules(match_type);


-- ============================================================================
-- Product/Label Mapping (OpenFDA/DailyMed)
-- ============================================================================
CREATE TABLE IF NOT EXISTS map_product_molecules (
    id BIGSERIAL PRIMARY KEY,
    set_id TEXT NOT NULL,
    source_table TEXT,
    match_name TEXT,
    mol_id BIGINT REFERENCES dm_molecule(mol_id),
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id),
    match_type TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(set_id, mol_id)
);

CREATE INDEX IF NOT EXISTS idx_map_prod_mol ON map_product_molecules(mol_id);
CREATE INDEX IF NOT EXISTS idx_map_prod_concept ON map_product_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_map_prod_set ON map_product_molecules(set_id);
"""

CREATE_INDICATION_SCHEMA_SQL = """
-- ============================================================================
-- INDICATION/DISEASE TABLE
-- Structured indication data from ChEMBL drug_indication + ontologies
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_indication (
    indication_id BIGSERIAL PRIMARY KEY,
    
    -- Ontology identifiers
    mesh_id TEXT UNIQUE,
    mesh_heading TEXT,
    efo_id TEXT,
    efo_term TEXT,
    mondo_id TEXT,
    snomed_id TEXT,
    icd10_codes TEXT[],
    
    -- Canonical name (best available)
    preferred_name TEXT NOT NULL,
    preferred_name_lower TEXT GENERATED ALWAYS AS (LOWER(preferred_name)) STORED,
    
    -- Classification
    therapeutic_area TEXT,  -- 'ONCOLOGY', 'IMMUNOLOGY', 'CARDIOLOGY', etc.
    indication_class TEXT,  -- 'DISEASE', 'SYMPTOM', 'SYNDROME', 'CONDITION'
    
    -- Hierarchy support
    parent_indication_id BIGINT REFERENCES dm_indication(indication_id),
    
    -- Synonyms for matching (denormalized for performance)
    synonyms TEXT[],
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_indication_mesh ON dm_indication(mesh_id);
CREATE INDEX IF NOT EXISTS idx_indication_efo ON dm_indication(efo_id);
CREATE INDEX IF NOT EXISTS idx_indication_name ON dm_indication(preferred_name);
CREATE INDEX IF NOT EXISTS idx_indication_name_lower ON dm_indication(preferred_name_lower);
CREATE INDEX IF NOT EXISTS idx_indication_name_trgm ON dm_indication USING gin(preferred_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_indication_area ON dm_indication(therapeutic_area);


-- ============================================================================
-- DRUG-INDICATION MAPPING TABLE
-- Links drug concepts to indications with approval status
-- ============================================================================
CREATE TABLE IF NOT EXISTS dm_drug_indication (
    id BIGSERIAL PRIMARY KEY,
    
    -- Drug reference (concept level - all forms of a drug)
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id) ON DELETE CASCADE,
    
    -- Indication reference
    indication_id BIGINT REFERENCES dm_indication(indication_id) ON DELETE CASCADE,
    
    -- Approval/Development status
    max_phase NUMERIC,  -- 0-4 (4 = approved)
    is_approved BOOLEAN GENERATED ALWAYS AS (max_phase >= 4) STORED,
    
    -- Source tracking
    sources TEXT[],
    chembl_drugind_id BIGINT,
    
    -- Reference information
    ref_type TEXT,      -- 'ClinicalTrials', 'FDA', 'EMA', 'DailyMed', etc.
    ref_id TEXT,        -- NCT ID, FDA application number, etc.
    ref_url TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(concept_id, indication_id)
);

CREATE INDEX IF NOT EXISTS idx_drug_ind_concept ON dm_drug_indication(concept_id);
CREATE INDEX IF NOT EXISTS idx_drug_ind_indication ON dm_drug_indication(indication_id);
CREATE INDEX IF NOT EXISTS idx_drug_ind_approved ON dm_drug_indication(is_approved);
CREATE INDEX IF NOT EXISTS idx_drug_ind_phase ON dm_drug_indication(max_phase DESC);


-- ============================================================================
-- DAILYMED PRODUCT LINKAGE
-- Links DailyMed products (FDA labels) to our molecule concepts
-- ============================================================================
CREATE TABLE IF NOT EXISTS map_dailymed_molecules (
    id BIGSERIAL PRIMARY KEY,
    
    -- DailyMed reference
    set_id TEXT NOT NULL,
    product_name TEXT,
    
    -- Our molecule layer
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id) ON DELETE CASCADE,
    mol_id BIGINT REFERENCES dm_molecule(mol_id) ON DELETE CASCADE,
    
    -- Match quality
    match_type TEXT,  -- 'SUBSTANCE', 'GENERIC_NAME', 'BRAND_NAME', 'FUZZY'
    matched_name TEXT,
    confidence NUMERIC,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(set_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_dailymed_mol_concept ON map_dailymed_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_dailymed_mol_setid ON map_dailymed_molecules(set_id);
CREATE INDEX IF NOT EXISTS idx_dailymed_mol_match ON map_dailymed_molecules(match_type);


-- ============================================================================
-- OPENFDA LABEL LINKAGE
-- Links OpenFDA labels to our molecule concepts
-- ============================================================================
CREATE TABLE IF NOT EXISTS map_openfda_molecules (
    id BIGSERIAL PRIMARY KEY,
    
    -- OpenFDA reference
    set_id TEXT NOT NULL,
    set_id_fk BIGINT,  -- FK to set_ids table if exists
    product_name TEXT,
    
    -- Our molecule layer
    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id) ON DELETE CASCADE,
    mol_id BIGINT REFERENCES dm_molecule(mol_id) ON DELETE CASCADE,
    
    -- Match quality
    match_type TEXT,  -- 'SUBSTANCE', 'GENERIC_NAME', 'BRAND_NAME', 'RXCUI', 'FUZZY'
    matched_name TEXT,
    confidence NUMERIC,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(set_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_openfda_mol_concept ON map_openfda_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_openfda_mol_setid ON map_openfda_molecules(set_id);
CREATE INDEX IF NOT EXISTS idx_openfda_mol_match ON map_openfda_molecules(match_type);
"""


# ============================================================================
# RDKIT HELPER FUNCTIONS IN POSTGRESQL
# ============================================================================

CREATE_RDKIT_FUNCTIONS_SQL = """
-- ============================================================================
-- Drop existing functions first (for idempotency)
-- ============================================================================
DROP FUNCTION IF EXISTS get_largest_fragment(mol) CASCADE;
DROP FUNCTION IF EXISTS get_largest_fragment_smiles(TEXT) CASCADE;
DROP FUNCTION IF EXISTS detect_salt_type(TEXT) CASCADE;
DROP FUNCTION IF EXISTS classify_stereo(mol) CASCADE;
DROP FUNCTION IF EXISTS classify_stereo_from_smiles(TEXT) CASCADE;
DROP FUNCTION IF EXISTS count_chiral_centers(mol) CASCADE;
DROP FUNCTION IF EXISTS count_chiral_centers_from_smiles(TEXT) CASCADE;
DROP FUNCTION IF EXISTS count_components(TEXT) CASCADE;
DROP FUNCTION IF EXISTS get_inchi_key_14(TEXT) CASCADE;

-- ============================================================================
-- Function: Extract first 14 chars from InChI key (connectivity layer)
-- ============================================================================
CREATE OR REPLACE FUNCTION get_inchi_key_14(full_inchi_key TEXT)
RETURNS TEXT AS $$
BEGIN
    IF full_inchi_key IS NULL OR length(full_inchi_key) < 14 THEN
        RETURN NULL;
    END IF;
    RETURN left(full_inchi_key, 14);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Get the largest fragment SMILES (salt stripping)
-- ============================================================================
CREATE OR REPLACE FUNCTION get_largest_fragment_smiles(input_smiles TEXT)
RETURNS TEXT AS $$
DECLARE
    frags TEXT[];
    largest_frag TEXT;
    largest_frag_mol mol;
    current_mol mol;
    max_atoms INT := 0;
    current_atoms INT;
BEGIN
    IF input_smiles IS NULL OR input_smiles = '' THEN
        RETURN NULL;
    END IF;
    
    frags := string_to_array(input_smiles, '.');
    
    IF frags IS NULL OR array_length(frags, 1) IS NULL OR array_length(frags, 1) = 1 THEN
        current_mol := mol_from_smiles(input_smiles::cstring);
        IF current_mol IS NOT NULL THEN
            RETURN mol_to_smiles(current_mol)::TEXT;
        END IF;
        RETURN input_smiles;
    END IF;
    
    FOR i IN 1..array_length(frags, 1) LOOP
        BEGIN
            current_mol := mol_from_smiles(frags[i]::cstring);
            IF current_mol IS NOT NULL THEN
                current_atoms := mol_numheavyatoms(current_mol);
                IF current_atoms > max_atoms THEN
                    max_atoms := current_atoms;
                    largest_frag := frags[i];
                    largest_frag_mol := current_mol;
                END IF;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            CONTINUE;
        END;
    END LOOP;
    
    IF largest_frag_mol IS NOT NULL THEN
        RETURN mol_to_smiles(largest_frag_mol)::TEXT;
    END IF;
    
    RETURN largest_frag;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Get the largest fragment as mol object
-- ============================================================================
CREATE OR REPLACE FUNCTION get_largest_fragment(input_mol mol)
RETURNS mol AS $$
DECLARE
    parent_smiles TEXT;
BEGIN
    IF input_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    parent_smiles := get_largest_fragment_smiles(mol_to_smiles(input_mol)::TEXT);
    
    IF parent_smiles IS NULL THEN
        RETURN input_mol;
    END IF;
    
    RETURN mol_from_smiles(parent_smiles::cstring);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Detect salt type from SMILES
-- ============================================================================
CREATE OR REPLACE FUNCTION detect_salt_type(input_smiles TEXT)
RETURNS TEXT AS $$
DECLARE
    frags TEXT[];
    n_frags INTEGER;
    salt_frag TEXT;
    smallest_len INTEGER := 999999;
BEGIN
    IF input_smiles IS NULL OR input_smiles = '' THEN
        RETURN 'UNKNOWN';
    END IF;
    
    frags := string_to_array(input_smiles, '.');
    n_frags := array_length(frags, 1);
    
    IF n_frags IS NULL OR n_frags = 1 THEN 
        RETURN 'FREE_BASE'; 
    END IF;
    
    FOR i IN 1..n_frags LOOP
        IF length(frags[i]) < smallest_len THEN
            smallest_len := length(frags[i]);
            salt_frag := frags[i];
        END IF;
    END LOOP;
    
    IF salt_frag IN ('[Cl-]', '[Cl]', 'Cl') THEN RETURN 'HYDROCHLORIDE';
    ELSIF salt_frag IN ('[Br-]', '[Br]', 'Br') THEN RETURN 'HYDROBROMIDE';
    ELSIF salt_frag IN ('[I-]', '[I]', 'I') THEN RETURN 'HYDROIODIDE';
    ELSIF salt_frag IN ('[Na+]', '[Na]') THEN RETURN 'SODIUM';
    ELSIF salt_frag IN ('[K+]', '[K]') THEN RETURN 'POTASSIUM';
    ELSIF salt_frag IN ('[Li+]', '[Li]') THEN RETURN 'LITHIUM';
    ELSIF salt_frag LIKE '[Ca%' THEN RETURN 'CALCIUM';
    ELSIF salt_frag LIKE '[Mg%' THEN RETURN 'MAGNESIUM';
    ELSIF salt_frag LIKE '[Zn%' THEN RETURN 'ZINC';
    ELSIF salt_frag IN ('O', '[OH2]') THEN RETURN 'HYDRATE';
    ELSE RETURN 'SALT_OTHER';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Classify stereochemistry from SMILES
-- ============================================================================
CREATE OR REPLACE FUNCTION classify_stereo_from_smiles(input_smiles TEXT)
RETURNS TEXT AS $$
DECLARE
    n_at_symbols INTEGER;
BEGIN
    IF input_smiles IS NULL OR input_smiles = '' THEN
        RETURN 'UNKNOWN';
    END IF;
    
    n_at_symbols := length(input_smiles) - length(replace(input_smiles, '@', ''));
    
    IF n_at_symbols = 0 THEN
        IF position('/' in input_smiles) > 0 OR position('\' in input_smiles) > 0 THEN
            RETURN 'GEOMETRIC';
        END IF;
        RETURN 'ACHIRAL';
    END IF;
    
    RETURN 'DEFINED';
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Classify stereochemistry from mol
-- ============================================================================
CREATE OR REPLACE FUNCTION classify_stereo(input_mol mol)
RETURNS TEXT AS $$
BEGIN
    IF input_mol IS NULL THEN
        RETURN 'UNKNOWN';
    END IF;
    RETURN classify_stereo_from_smiles(mol_to_smiles(input_mol)::TEXT);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Count chiral centers from SMILES
-- ============================================================================
CREATE OR REPLACE FUNCTION count_chiral_centers_from_smiles(input_smiles TEXT)
RETURNS INTEGER AS $$
DECLARE
    n_at_symbols INTEGER;
    n_double_at INTEGER;
BEGIN
    IF input_smiles IS NULL THEN
        RETURN 0;
    END IF;
    
    n_at_symbols := length(input_smiles) - length(replace(input_smiles, '@', ''));
    n_double_at := (length(input_smiles) - length(replace(input_smiles, '@@', ''))) / 2;
    
    RETURN GREATEST(0, n_at_symbols - n_double_at);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Count chiral centers from mol
-- ============================================================================
CREATE OR REPLACE FUNCTION count_chiral_centers(input_mol mol)
RETURNS INTEGER AS $$
BEGIN
    IF input_mol IS NULL THEN
        RETURN 0;
    END IF;
    RETURN count_chiral_centers_from_smiles(mol_to_smiles(input_mol)::TEXT);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- Function: Count components in SMILES
-- ============================================================================
CREATE OR REPLACE FUNCTION count_components(input_smiles TEXT)
RETURNS INTEGER AS $$
BEGIN
    IF input_smiles IS NULL OR input_smiles = '' THEN
        RETURN 0;
    END IF;
    RETURN COALESCE(array_length(string_to_array(input_smiles, '.'), 1), 1);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


THERAPEUTIC_AREA_KEYWORDS = {
    'ONCOLOGY': [
        'neoplasm', 'cancer', 'carcinoma', 'tumor', 'tumour', 'lymphoma', 
        'leukemia', 'leukaemia', 'melanoma', 'sarcoma', 'myeloma', 'glioma',
        'adenocarcinoma', 'metasta', 'malignant', 'oncolog'
    ],
    'IMMUNOLOGY': [
        'autoimmune', 'immune', 'rheumatoid', 'lupus', 'psoriasis', 'crohn',
        'colitis', 'inflammatory bowel', 'multiple sclerosis', 'immunodeficiency',
        'allergy', 'allergic', 'asthma', 'atopic', 'graft versus host'
    ],
    'CARDIOLOGY': [
        'heart', 'cardiac', 'cardiovascular', 'coronary', 'hypertension',
        'arrhythmia', 'atrial fibrillation', 'myocardial', 'angina', 
        'thrombosis', 'embolism', 'stroke', 'atherosclerosis'
    ],
    'NEUROLOGY': [
        'neurolog', 'alzheimer', 'parkinson', 'epilepsy', 'seizure', 'migraine',
        'neuropath', 'dementia', 'huntington', 'amyotrophic', 'multiple sclerosis',
        'brain', 'central nervous', 'neurodegenerative'
    ],
    'INFECTIOUS_DISEASE': [
        'infection', 'infectious', 'bacterial', 'viral', 'fungal', 'hiv', 'aids',
        'hepatitis', 'tuberculosis', 'malaria', 'sepsis', 'pneumonia', 
        'influenza', 'covid', 'coronavirus', 'antibiotic', 'antiviral'
    ],
    'ENDOCRINOLOGY': [
        'diabetes', 'diabetic', 'thyroid', 'hormone', 'endocrine', 'obesity',
        'metabolic', 'insulin', 'glucose', 'adrenal', 'pituitary', 'osteoporosis'
    ],
    'GASTROENTEROLOGY': [
        'gastro', 'intestin', 'hepat', 'liver', 'pancrea', 'bowel', 'colon',
        'stomach', 'ulcer', 'reflux', 'cirrhosis', 'cholesterol', 'dyslipidemia'
    ],
    'RESPIRATORY': [
        'pulmonary', 'lung', 'respiratory', 'copd', 'asthma', 'bronch',
        'pneumonia', 'fibrosis', 'emphysema', 'airway'
    ],
    'DERMATOLOGY': [
        'skin', 'dermat', 'psoriasis', 'eczema', 'acne', 'melanoma',
        'alopecia', 'vitiligo', 'urticaria'
    ],
    'HEMATOLOGY': [
        'blood', 'hematolog', 'anemia', 'hemophilia', 'thrombocytopenia',
        'neutropenia', 'coagulation', 'bleeding', 'platelet'
    ],
    'OPHTHALMOLOGY': [
        'eye', 'ocular', 'ophthalm', 'retina', 'macular', 'glaucoma',
        'cataract', 'uveitis', 'vision'
    ],
    'NEPHROLOGY': [
        'kidney', 'renal', 'nephro', 'dialysis', 'glomerulo', 'proteinuria'
    ],
    'PSYCHIATRY': [
        'depression', 'anxiety', 'schizophrenia', 'bipolar', 'psycho',
        'psychiatric', 'mental', 'adhd', 'ocd', 'ptsd'
    ],
    'RARE_DISEASE': [
        'orphan', 'rare', 'genetic', 'hereditary', 'congenital', 'syndrome'
    ],
}


def fast_drop_schema(config: DatabaseConfig):
    """Fast schema drop using CASCADE - let PostgreSQL handle dependencies."""
    print("\n🧹 Dropping schema...")
    
    with get_connection(config) as con:
        con.autocommit = True
        with con.cursor() as cur:
            start_time = time.time()
            
            # Just drop everything with CASCADE - PostgreSQL handles dependencies
            print("  ⚡ Dropping all tables and views...")
            
            drop_statements = [
                "DROP MATERIALIZED VIEW IF EXISTS dm_compound_target_activity CASCADE",
                "DROP TABLE IF EXISTS map_product_molecules CASCADE",
                "DROP TABLE IF EXISTS map_ctgov_molecules CASCADE", 
                "DROP TABLE IF EXISTS dm_molecule_synonyms CASCADE",
                "DROP TABLE IF EXISTS dm_molecule_target_summary CASCADE",
                "DROP TABLE IF EXISTS dm_molecule CASCADE",
                "DROP TABLE IF EXISTS dm_molecule_stereo CASCADE",
                "DROP TABLE IF EXISTS dm_molecule_concept CASCADE",
            ]
            
            for stmt in drop_statements:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    print(f"     Warning: {e}")
                    pass  # Continue even if one fails
            
            elapsed = time.time() - start_time
            print(f"  ✅ Schema dropped in {elapsed:.1f}s")


def classify_therapeutic_area(mesh_heading: str, efo_term: str = None) -> str:
    """Classify an indication into a therapeutic area based on keywords."""
    if not mesh_heading and not efo_term:
        return 'OTHER'
    
    text = f"{mesh_heading or ''} {efo_term or ''}".lower()
    
    # Check each therapeutic area
    scores = {}
    for area, keywords in THERAPEUTIC_AREA_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[area] = score
    
    if scores:
        # Return the area with highest score
        return max(scores, key=scores.get)
    
    return 'OTHER'


# ============================================================================
# PHASE 1: POPULATE MOLECULES WITH NORMALIZATION
# ============================================================================

def populate_molecules(config: DatabaseConfig, source_counts: Optional[Dict[str, int]] = None,
                       limit: Optional[int] = None):
    """Load molecules from all sources with RDKit normalization."""
    print("\n🧪 PHASE 1: Consolidating Molecules with RDKit Normalization...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting to ~{limit:,} molecules per source")
    
    if source_counts:
        chembl_expected = min(source_counts.get('chembl', 0), limit or float('inf'))
        dc_expected = min(source_counts.get('drugcentral', 0), limit or float('inf'))
        bdb_expected = min(source_counts.get('bindingdb', 0), limit or float('inf'))
        print(f"  📊 Expected: ChEMBL≈{int(chembl_expected):,}, "
              f"DrugCentral≈{int(dc_expected):,}, "
              f"BindingDB≈{int(bdb_expected):,}")
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # --- 1.1 Load ChEMBL ---
            print("  📥 Loading ChEMBL molecules...")
            cur.execute(f"""
                INSERT INTO dm_molecule (
                    inchi_key, standard_inchi, canonical_smiles, pref_name, chembl_id, sources,
                    mol, parent_smiles, parent_inchi_key_14, parent_stereo_inchi_key,
                    is_salt, salt_form, n_components, stereo_type, n_chiral_centers,
                    mfp2, ffp2
                )
                SELECT DISTINCT ON (s.standard_inchi_key)
                    s.standard_inchi_key,
                    s.standard_inchi,
                    s.canonical_smiles,
                    md.pref_name,
                    md.chembl_id,
                    ARRAY['CHEMBL'],
                    mol_from_smiles(s.canonical_smiles::cstring),
                    get_largest_fragment_smiles(s.canonical_smiles),
                    get_inchi_key_14(s.standard_inchi_key),
                    s.standard_inchi_key,
                    count_components(s.canonical_smiles) > 1,
                    detect_salt_type(s.canonical_smiles),
                    count_components(s.canonical_smiles),
                    classify_stereo_from_smiles(get_largest_fragment_smiles(s.canonical_smiles)),
                    count_chiral_centers_from_smiles(get_largest_fragment_smiles(s.canonical_smiles)),
                    morganbv_fp(mol_from_smiles(get_largest_fragment_smiles(s.canonical_smiles)::cstring)),
                    featmorganbv_fp(mol_from_smiles(get_largest_fragment_smiles(s.canonical_smiles)::cstring))
                FROM compound_structures s
                JOIN molecule_dictionary md ON s.molregno = md.molregno
                WHERE s.standard_inchi_key IS NOT NULL
                  AND s.canonical_smiles IS NOT NULL
                  AND length(s.standard_inchi_key) >= 14
                  AND mol_from_smiles(s.canonical_smiles::cstring) IS NOT NULL
                ORDER BY s.standard_inchi_key, s.molregno
                {limit_clause}
                ON CONFLICT (inchi_key) DO UPDATE SET
                    chembl_id = EXCLUDED.chembl_id,
                    pref_name = COALESCE(dm_molecule.pref_name, EXCLUDED.pref_name),
                    sources = CASE 
                        WHEN 'CHEMBL' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'CHEMBL')
                    END;
            """)
            chembl_count = cur.rowcount
            print(f"     ✓ ChEMBL molecules loaded: {chembl_count:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_1', 'load_chembl', chembl_count, time.time() - start_time, 'SUCCESS')
            con.commit()

            # --- 1.2 Load DrugCentral ---
            start_time = time.time()
            print("  📥 Merging DrugCentral molecules...")
            cur.execute(f"""
                INSERT INTO dm_molecule (
                    inchi_key, canonical_smiles, pref_name, drugcentral_id, sources,
                    mol, parent_smiles, parent_inchi_key_14, parent_stereo_inchi_key,
                    is_salt, salt_form, n_components, stereo_type, n_chiral_centers,
                    mfp2, ffp2
                )
                SELECT DISTINCT ON (inchi_key)
                    inchi_key,
                    smiles,
                    name,
                    id,
                    ARRAY['DC'],
                    mol_from_smiles(smiles::cstring),
                    get_largest_fragment_smiles(smiles),
                    get_inchi_key_14(inchi_key),
                    inchi_key,
                    count_components(smiles) > 1,
                    detect_salt_type(smiles),
                    count_components(smiles),
                    classify_stereo_from_smiles(get_largest_fragment_smiles(smiles)),
                    count_chiral_centers_from_smiles(get_largest_fragment_smiles(smiles)),
                    morganbv_fp(mol_from_smiles(get_largest_fragment_smiles(smiles)::cstring)),
                    featmorganbv_fp(mol_from_smiles(get_largest_fragment_smiles(smiles)::cstring))
                FROM drugcentral_drugs
                WHERE inchi_key IS NOT NULL
                  AND smiles IS NOT NULL
                  AND length(inchi_key) >= 14
                  AND mol_from_smiles(smiles::cstring) IS NOT NULL
                ORDER BY inchi_key, id DESC
                {limit_clause}
                ON CONFLICT (inchi_key) DO UPDATE SET
                    drugcentral_id = EXCLUDED.drugcentral_id,
                    pref_name = COALESCE(dm_molecule.pref_name, EXCLUDED.pref_name),
                    sources = CASE 
                        WHEN 'DC' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'DC')
                    END;
            """)
            dc_count = cur.rowcount
            print(f"     ✓ DrugCentral molecules merged: {dc_count:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_1', 'load_drugcentral', dc_count, time.time() - start_time, 'SUCCESS')
            con.commit()

            # --- 1.3 Load BindingDB ---
            start_time = time.time()
            print("  📥 Merging BindingDB molecules...")
            cur.execute(f"""
                INSERT INTO dm_molecule (
                    inchi_key, standard_inchi, canonical_smiles, pref_name, 
                    bindingdb_monomer_id, pubchem_cid, sources,
                    mol, parent_smiles, parent_inchi_key_14, parent_stereo_inchi_key,
                    is_salt, salt_form, n_components, stereo_type, n_chiral_centers,
                    mfp2, ffp2
                )
                SELECT DISTINCT ON (inchi_key)
                    inchi_key,
                    inchi,
                    smiles,
                    ligand_name,
                    id,
                    pubchem_cid,
                    ARRAY['BDB'],
                    mol_from_smiles(smiles::cstring),
                    get_largest_fragment_smiles(smiles),
                    get_inchi_key_14(inchi_key),
                    inchi_key,
                    count_components(smiles) > 1,
                    detect_salt_type(smiles),
                    count_components(smiles),
                    classify_stereo_from_smiles(get_largest_fragment_smiles(smiles)),
                    count_chiral_centers_from_smiles(get_largest_fragment_smiles(smiles)),
                    morganbv_fp(mol_from_smiles(get_largest_fragment_smiles(smiles)::cstring)),
                    featmorganbv_fp(mol_from_smiles(get_largest_fragment_smiles(smiles)::cstring))
                FROM bindingdb_molecules
                WHERE inchi_key IS NOT NULL
                  AND smiles IS NOT NULL
                  AND length(inchi_key) >= 14
                  AND mol_from_smiles(smiles::cstring) IS NOT NULL
                ORDER BY inchi_key, id DESC
                {limit_clause}
                ON CONFLICT (inchi_key) DO UPDATE SET
                    bindingdb_monomer_id = EXCLUDED.bindingdb_monomer_id,
                    pubchem_cid = COALESCE(dm_molecule.pubchem_cid, EXCLUDED.pubchem_cid),
                    pref_name = COALESCE(dm_molecule.pref_name, EXCLUDED.pref_name),
                    sources = CASE 
                        WHEN 'BDB' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'BDB')
                    END;
            """)
            bdb_count = cur.rowcount
            print(f"     ✓ BindingDB molecules merged: {bdb_count:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_1', 'load_bindingdb', bdb_count, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            # --- 1.4 Fix parent keys for salts ---
            print("  🧂 Fixing parent keys for salt forms...")
            cur.execute("""
                WITH parent_lookup AS (
                    SELECT DISTINCT ON (parent_smiles)
                        parent_smiles,
                        parent_inchi_key_14,
                        inchi_key as parent_full_key
                    FROM dm_molecule
                    WHERE is_salt = FALSE
                      AND parent_smiles IS NOT NULL
                      AND parent_inchi_key_14 IS NOT NULL
                    ORDER BY parent_smiles, mol_id
                )
                UPDATE dm_molecule dm
                SET 
                    parent_inchi_key_14 = pl.parent_inchi_key_14,
                    parent_stereo_inchi_key = pl.parent_full_key
                FROM parent_lookup pl
                WHERE dm.parent_smiles = pl.parent_smiles
                  AND dm.is_salt = TRUE
                  AND dm.parent_inchi_key_14 != pl.parent_inchi_key_14;
            """)
            salt_fixed = cur.rowcount
            print(f"     ✓ Salt parent keys fixed: {salt_fixed:,}")
            con.commit()
            
            # Get total count and stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT parent_inchi_key_14) as unique_concepts,
                    SUM(CASE WHEN is_salt THEN 1 ELSE 0 END) as salts
                FROM dm_molecule
            """)
            stats = cur.fetchone()
            print(f"\n  📊 Phase 1 Summary:")
            print(f"     Total molecules: {stats[0]:,}")
            print(f"     Unique concepts (parent_inchi_key_14): {stats[1]:,}")
            print(f"     Salt forms: {stats[2]:,}")
            
            return stats[0]


# ============================================================================
# PHASE 2: BUILD MOLECULE HIERARCHY
# ============================================================================

def build_molecule_hierarchy(config: DatabaseConfig):
    """Build the concept and stereo hierarchy from normalized molecules."""
    print("\n🏗️  PHASE 2: Building Molecule Hierarchy...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # --- 2.1 Create Drug Concepts (Level 1) ---
            print("  📊 Creating drug concepts (Level 1 - connectivity grouping)...")
            
            # Use raw string to avoid Python escape sequence warnings
            cur.execute(r"""
                INSERT INTO dm_molecule_concept (
                    parent_inchi_key_14,
                    parent_smiles,
                    preferred_name,
                    has_stereo_variants,
                    has_salt_forms,
                    n_forms
                )
                SELECT 
                    parent_inchi_key_14,
                    MODE() WITHIN GROUP (ORDER BY parent_smiles) as parent_smiles,
                    -- Priority: ChEMBL single-word name > any ChEMBL name > shortest name
                    COALESCE(
                        (SELECT pref_name FROM dm_molecule m2 
                         WHERE m2.parent_inchi_key_14 = m.parent_inchi_key_14 
                           AND 'CHEMBL' = ANY(m2.sources)
                           AND m2.pref_name IS NOT NULL
                           AND m2.pref_name !~ '\s'
                           AND length(m2.pref_name) > 2
                         ORDER BY length(m2.pref_name) LIMIT 1),
                        (SELECT pref_name FROM dm_molecule m2 
                         WHERE m2.parent_inchi_key_14 = m.parent_inchi_key_14 
                           AND 'CHEMBL' = ANY(m2.sources)
                           AND m2.pref_name IS NOT NULL
                           AND length(m2.pref_name) > 2
                         ORDER BY length(m2.pref_name) LIMIT 1),
                        (SELECT pref_name FROM dm_molecule m2 
                         WHERE m2.parent_inchi_key_14 = m.parent_inchi_key_14 
                           AND m2.pref_name IS NOT NULL
                           AND length(m2.pref_name) > 2
                         ORDER BY length(m2.pref_name) LIMIT 1)
                    ) as preferred_name,
                    COUNT(DISTINCT parent_stereo_inchi_key) > 1 as has_stereo_variants,
                    bool_or(is_salt) as has_salt_forms,
                    COUNT(*) as n_forms
                FROM dm_molecule m
                WHERE parent_inchi_key_14 IS NOT NULL
                GROUP BY parent_inchi_key_14
                ON CONFLICT (parent_inchi_key_14) DO UPDATE SET
                    preferred_name = COALESCE(EXCLUDED.preferred_name, dm_molecule_concept.preferred_name),
                    has_stereo_variants = EXCLUDED.has_stereo_variants,
                    has_salt_forms = EXCLUDED.has_salt_forms,
                    n_forms = EXCLUDED.n_forms,
                    updated_at = NOW();
            """)
            concept_count = cur.rowcount
            print(f"     ✓ Drug concepts created/updated: {concept_count:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_2', 'create_concepts', concept_count, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            # --- 2.2 Create Stereo Forms (Level 2) ---
            start_time = time.time()
            print("  📊 Creating stereo forms (Level 2 - stereoisomer grouping)...")
            cur.execute("""
                INSERT INTO dm_molecule_stereo (
                    concept_id,
                    parent_stereo_inchi_key,
                    parent_stereo_smiles,
                    stereo_type,
                    n_chiral_centers,
                    n_defined_centers,
                    is_api
                )
                SELECT DISTINCT ON (dm.parent_stereo_inchi_key)
                    mc.concept_id,
                    dm.parent_stereo_inchi_key,
                    dm.parent_smiles,
                    dm.stereo_type,
                    dm.n_chiral_centers,
                    CASE 
                        WHEN dm.stereo_type = 'DEFINED' THEN dm.n_chiral_centers
                        ELSE 0
                    END as n_defined_centers,
                    (dm.stereo_type = 'DEFINED' AND NOT dm.is_salt) as is_api
                FROM dm_molecule dm
                JOIN dm_molecule_concept mc ON dm.parent_inchi_key_14 = mc.parent_inchi_key_14
                WHERE dm.parent_stereo_inchi_key IS NOT NULL
                ORDER BY dm.parent_stereo_inchi_key, dm.is_salt ASC, dm.mol_id
                ON CONFLICT (parent_stereo_inchi_key) DO UPDATE SET
                    stereo_type = EXCLUDED.stereo_type,
                    n_chiral_centers = EXCLUDED.n_chiral_centers,
                    n_defined_centers = EXCLUDED.n_defined_centers;
            """)
            stereo_count = cur.rowcount
            print(f"     ✓ Stereo forms created/updated: {stereo_count:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_2', 'create_stereo_forms', stereo_count, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            # --- 2.3 Link Molecules to Hierarchy ---
            start_time = time.time()
            print("  🔗 Linking molecules to hierarchy...")
            
            # Link to concepts first
            cur.execute("""
                UPDATE dm_molecule dm
                SET concept_id = mc.concept_id
                FROM dm_molecule_concept mc
                WHERE dm.parent_inchi_key_14 = mc.parent_inchi_key_14
                  AND dm.parent_inchi_key_14 IS NOT NULL;
            """)
            concept_linked = cur.rowcount
            print(f"     ✓ Molecules linked to concepts: {concept_linked:,}")
            con.commit()
            
            # Link to stereo forms
            cur.execute("""
                UPDATE dm_molecule dm
                SET stereo_id = ms.stereo_id
                FROM dm_molecule_stereo ms
                WHERE dm.parent_stereo_inchi_key = ms.parent_stereo_inchi_key
                  AND dm.parent_stereo_inchi_key IS NOT NULL;
            """)
            stereo_linked = cur.rowcount
            print(f"     ✓ Molecules linked to stereo forms: {stereo_linked:,}")
            con.commit()
            
            log_audit(cur, 'PHASE_2', 'link_hierarchy', concept_linked + stereo_linked, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            # Print summary statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_molecules,
                    COUNT(DISTINCT concept_id) as unique_concepts,
                    COUNT(DISTINCT stereo_id) as unique_stereo_forms,
                    SUM(CASE WHEN is_salt THEN 1 ELSE 0 END) as salt_forms,
                    SUM(CASE WHEN stereo_type = 'DEFINED' THEN 1 ELSE 0 END) as defined_stereo,
                    SUM(CASE WHEN stereo_type = 'ACHIRAL' THEN 1 ELSE 0 END) as achiral,
                    SUM(CASE WHEN stereo_type = 'GEOMETRIC' THEN 1 ELSE 0 END) as geometric,
                    SUM(CASE WHEN concept_id IS NULL THEN 1 ELSE 0 END) as orphaned
                FROM dm_molecule
            """)
            stats = cur.fetchone()
            print(f"\n  📊 Hierarchy Statistics:")
            print(f"     Total molecules:     {stats[0]:,}")
            print(f"     Unique concepts:     {stats[1]:,}")
            print(f"     Unique stereo forms: {stats[2]:,}")
            print(f"     Salt forms:          {stats[3]:,}")
            print(f"     Defined stereo:      {stats[4]:,}")
            print(f"     Achiral:             {stats[5]:,}")
            print(f"     Geometric stereo:    {stats[6]:,}")
            if stats[7] and stats[7] > 0:
                print(f"     ⚠️  Orphaned (no concept): {stats[7]:,}")


# ============================================================================
# PHASE 3: POPULATE SYNONYMS
# ============================================================================

def populate_synonyms(config: DatabaseConfig, limit: Optional[int] = None):
    """
    Build synonym dictionary linked to both molecules and concepts.
    
    Args:
        config: Database configuration
        limit: Maximum synonyms to load per source (for fast/debug mode)
    """
    print("\n📚 PHASE 3: Building Synonym Dictionary...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting synonyms per source to ~{limit:,}")
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # Clear existing synonyms for clean rebuild
            print("  🧹 Clearing existing synonyms...")
            cur.execute("TRUNCATE dm_molecule_synonyms RESTART IDENTITY CASCADE;")
            con.commit()
            
            # --- 3.1 ChEMBL Synonyms ---
            print("  📥 Indexing ChEMBL synonyms...")
            cur.execute(f"""
                INSERT INTO dm_molecule_synonyms (mol_id, concept_id, synonym, syn_type, source)
                SELECT DISTINCT
                    dm.mol_id,
                    dm.concept_id,
                    ms.synonyms,
                    ms.syn_type,
                    'CHEMBL'
                FROM dm_molecule dm
                JOIN molecule_dictionary md ON dm.chembl_id = md.chembl_id
                JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                WHERE ms.synonyms IS NOT NULL
                  AND length(ms.synonyms) >= 2
                  AND length(ms.synonyms) <= 500
                  AND dm.concept_id IS NOT NULL
                {limit_clause}
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            chembl_syn_count = cur.rowcount
            print(f"     ✓ ChEMBL synonyms added: {chembl_syn_count:,}")
            con.commit()
            
            # --- 3.2 Add pref_names as synonyms ---
            print("  📥 Indexing preferred names as synonyms...")
            cur.execute(f"""
                INSERT INTO dm_molecule_synonyms (mol_id, concept_id, synonym, syn_type, source)
                SELECT mol_id, concept_id, pref_name, 'PREF_NAME', 
                       CASE 
                           WHEN 'CHEMBL' = ANY(sources) THEN 'CHEMBL'
                           WHEN 'DC' = ANY(sources) THEN 'DC'
                           ELSE 'BDB'
                       END
                FROM dm_molecule 
                WHERE pref_name IS NOT NULL
                  AND length(pref_name) >= 2
                  AND length(pref_name) <= 500
                  AND concept_id IS NOT NULL
                {limit_clause}
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            pref_count = cur.rowcount
            print(f"     ✓ Preferred names added: {pref_count:,}")
            con.commit()

            # --- 3.3 DrugCentral Synonyms ---
            print("  📥 Indexing DrugCentral synonyms...")
            cur.execute(f"""
                INSERT INTO dm_molecule_synonyms (mol_id, concept_id, synonym, syn_type, source)
                SELECT DISTINCT
                    dm.mol_id,
                    dm.concept_id,
                    trim(syn),
                    'GENERIC',
                    'DC'
                FROM dm_molecule dm
                JOIN drugcentral_drugs dd ON dm.drugcentral_id = dd.id
                CROSS JOIN LATERAL unnest(string_to_array(dd.synonyms, ';')) AS syn
                WHERE dd.synonyms IS NOT NULL
                  AND length(trim(syn)) >= 2
                  AND length(trim(syn)) <= 500
                  AND dm.concept_id IS NOT NULL
                {limit_clause}
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            dc_syn_count = cur.rowcount
            print(f"     ✓ DrugCentral synonyms added: {dc_syn_count:,}")
            con.commit()
            
            # --- 3.4 BindingDB names ---
            print("  📥 Indexing BindingDB names...")
            cur.execute(f"""
                INSERT INTO dm_molecule_synonyms (mol_id, concept_id, synonym, syn_type, source)
                SELECT DISTINCT
                    dm.mol_id,
                    dm.concept_id,
                    bm.ligand_name,
                    'LIGAND_NAME',
                    'BDB'
                FROM dm_molecule dm
                JOIN bindingdb_molecules bm ON dm.bindingdb_monomer_id = bm.id
                WHERE bm.ligand_name IS NOT NULL
                  AND length(bm.ligand_name) >= 2
                  AND length(bm.ligand_name) <= 500
                  AND bm.ligand_name NOT LIKE '%::%'
                  AND dm.concept_id IS NOT NULL
                {limit_clause}
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            bdb_syn_count = cur.rowcount
            print(f"     ✓ BindingDB names added: {bdb_syn_count:,}")
            con.commit()
            
            total_syns = chembl_syn_count + pref_count + dc_syn_count + bdb_syn_count
            log_audit(cur, 'PHASE_3', 'populate_synonyms', total_syns, time.time() - start_time, 'SUCCESS')
            con.commit()

            # Summary
            cur.execute("""
                SELECT 
                    COUNT(*) as total_synonyms,
                    COUNT(DISTINCT mol_id) as molecules_with_synonyms,
                    COUNT(DISTINCT concept_id) as concepts_with_synonyms
                FROM dm_molecule_synonyms
            """)
            stats = cur.fetchone()
            print(f"\n  📊 Synonym Statistics:")
            print(f"     Total synonyms:      {stats[0]:,}")
            print(f"     Molecules covered:   {stats[1]:,}")
            print(f"     Concepts covered:    {stats[2]:,}")


# ============================================================================
# PHASE 4: MATCH CLINICAL TRIALS
# ============================================================================

def clean_name(name: str) -> str:
    """Basic text normalization for matching."""
    if not name: 
        return ""
    name = name.lower().strip()
    # Remove common dosage info
    name = re.sub(r'\s*\d+(\.\d+)?\s*(mg|g|ml|mcg|ug|µg|%|iu|units?)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(injection|tablet|capsule|solution|cream|gel|ointment|patch|spray|powder|suspension|syrup|drops?)s?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(extended|immediate|sustained|delayed|modified)[\s-]?release\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(oral|topical|intravenous|subcutaneous|intramuscular|iv|im|sc)\b', '', name, flags=re.IGNORECASE)
    # Remove parenthetical content
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    # Remove registered/trademark symbols
    name = re.sub(r'[®™©]', '', name)
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


# Salt suffixes for stripping
SALT_SUFFIXES = [
    ' hcl', ' hydrochloride', ' dihydrochloride', ' trihydrochloride',
    ' sodium', ' potassium', ' calcium', ' magnesium',
    ' phosphate', ' diphosphate',
    ' sulfate', ' sulphate', ' bisulfate',
    ' acetate', ' diacetate',
    ' maleate', ' fumarate', ' succinate', ' tartrate', ' citrate',
    ' mesylate', ' mesilate', ' besylate', ' besilate', ' tosylate',
    ' lactate', ' gluconate', ' benzoate',
    ' bromide', ' chloride', ' iodide',
    ' nitrate', ' carbonate', ' bicarbonate',
    ' monohydrate', ' dihydrate', ' trihydrate', ' hemihydrate',
    ' hydrochloride monohydrate', ' hydrochloride dihydrate',
]


def map_clinical_trials(config: DatabaseConfig, enable_fuzzy: bool = True, 
                        fuzzy_threshold: float = 0.8, limit: Optional[int] = None):
    """
    Map CT.gov interventions to molecules AND concepts.
    
    Args:
        config: Database configuration
        enable_fuzzy: Enable fuzzy matching for unmatched interventions
        fuzzy_threshold: Minimum similarity for fuzzy matches
        limit: Maximum interventions to process (for fast/debug mode)
    """
    print("\n🏥 PHASE 4: Mapping CT.gov Interventions...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting to {limit:,} interventions")
    
    with get_connection(config) as con:
        start_time = time.time()
        
        # Clear existing mappings for clean rebuild
        print("  🧹 Clearing existing trial mappings...")
        with con.cursor() as cur:
            cur.execute("TRUNCATE map_ctgov_molecules RESTART IDENTITY;")
            con.commit()
        
        # Load synonyms into memory
        print("  🧠 Loading synonyms into memory...")
        syn_map = {}  # cleaned_name -> (mol_id, concept_id)
        
        with con.cursor(name='syn_cursor') as cur:
            cur.itersize = 10000
            cur.execute("""
                SELECT mol_id, concept_id, synonym_lower 
                FROM dm_molecule_synonyms
                WHERE length(synonym_lower) >= 3
                  AND concept_id IS NOT NULL
            """)
            for mol_id, concept_id, syn in tqdm(cur, desc="     Loading synonyms", leave=False):
                if syn and len(syn) >= 3:
                    syn_map[syn] = (mol_id, concept_id)
        
        print(f"     ✓ Loaded {len(syn_map):,} unique synonyms")

        # Fetch interventions
        print("  🔎 Matching interventions...")
        exact_matches = 0
        salt_stripped_matches = 0
        combo_matches = 0
        batch_inserts = []
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        with con.cursor() as cur:
            cur.execute(f"""
                SELECT id, nct_id, name 
                FROM ctgov_interventions 
                WHERE intervention_type IN ('DRUG', 'BIOLOGICAL', 'DIETARY SUPPLEMENT')
                  AND name IS NOT NULL
                  AND length(name) >= 2
                ORDER BY id
                {limit_clause}
            """)
            interventions = cur.fetchall()
            
            for int_id, nct_id, raw_name in tqdm(interventions, desc="     Matching", leave=False):
                if not raw_name: 
                    continue
                
                cleaned = clean_name(raw_name)
                
                if not cleaned or len(cleaned) < 3:
                    continue
                
                # 1. Exact Match
                if cleaned in syn_map:
                    mol_id, concept_id = syn_map[cleaned]
                    batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'EXACT', 1.0))
                    exact_matches += 1
                    continue
                
                # 2. Try without common salt suffixes
                matched = False
                for suffix in SALT_SUFFIXES:
                    if cleaned.endswith(suffix):
                        base_name = cleaned[:-len(suffix)].strip()
                        if base_name and base_name in syn_map:
                            mol_id, concept_id = syn_map[base_name]
                            batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'SALT_STRIPPED', 0.95))
                            salt_stripped_matches += 1
                            matched = True
                            break
                
                if matched:
                    continue
                
                # 3. Split combo drugs
                if " and " in cleaned or " + " in cleaned or "/" in cleaned or " with " in cleaned:
                    parts = re.split(r'\s+and\s+|\s*\+\s*|/|\s+with\s+', cleaned)
                    for part in parts:
                        p_clean = part.strip()
                        if p_clean and p_clean in syn_map:
                            mol_id, concept_id = syn_map[p_clean]
                            batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'COMBO_PART', 0.9))
                            combo_matches += 1
            
            # Bulk Insert exact matches
            if batch_inserts:
                print(f"  💾 Inserting {len(batch_inserts):,} matches...")
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO map_ctgov_molecules 
                       (nct_id, intervention_id, match_name, mol_id, concept_id, match_type, confidence) 
                       VALUES %s ON CONFLICT (intervention_id, mol_id) DO NOTHING""",
                    batch_inserts,
                    page_size=5000
                )
                con.commit()
        
        print(f"     ✓ Exact matches: {exact_matches:,}")
        print(f"     ✓ Salt-stripped matches: {salt_stripped_matches:,}")
        print(f"     ✓ Combo part matches: {combo_matches:,}")
        
        # Fuzzy matching for unmatched interventions
        fuzzy_matches = 0
        if enable_fuzzy:
            print("  🔍 Running fuzzy matching for unmatched interventions...")
            
            fuzzy_limit = min(limit or 50000, 50000)  # Cap fuzzy matching
            
            with con.cursor() as cur:
                # Get unmatched interventions
                cur.execute(f"""
                    SELECT DISTINCT i.id, i.nct_id, i.name
                    FROM ctgov_interventions i
                    LEFT JOIN map_ctgov_molecules m ON i.id = m.intervention_id
                    WHERE i.intervention_type IN ('DRUG', 'BIOLOGICAL')
                      AND i.name IS NOT NULL
                      AND length(i.name) >= 4
                      AND m.id IS NULL
                    LIMIT {fuzzy_limit}
                """)
                unmatched = cur.fetchall()
                
                if unmatched:
                    print(f"     Processing {len(unmatched):,} unmatched interventions...")
                    fuzzy_batch = []
                    
                    for int_id, nct_id, raw_name in tqdm(unmatched, desc="     Fuzzy matching", leave=False):
                        cleaned = clean_name(raw_name)
                        if len(cleaned) < 4:
                            continue
                        
                        # Use PostgreSQL trigram similarity
                        cur.execute("""
                            SELECT mol_id, concept_id, synonym_lower, 
                                   similarity(synonym_lower, %s) as sim
                            FROM dm_molecule_synonyms
                            WHERE synonym_lower %% %s
                              AND length(synonym_lower) >= 4
                              AND similarity(synonym_lower, %s) >= %s
                            ORDER BY sim DESC
                            LIMIT 1
                        """, (cleaned, cleaned, cleaned, fuzzy_threshold))
                        
                        result = cur.fetchone()
                        if result:
                            mol_id, concept_id, matched_syn, similarity = result
                            fuzzy_batch.append((
                                nct_id, int_id, raw_name, mol_id, concept_id, 
                                'FUZZY', float(similarity)
                            ))
                            fuzzy_matches += 1
                    
                    if fuzzy_batch:
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO map_ctgov_molecules 
                               (nct_id, intervention_id, match_name, mol_id, concept_id, match_type, confidence) 
                               VALUES %s ON CONFLICT (intervention_id, mol_id) DO NOTHING""",
                            fuzzy_batch,
                            page_size=5000
                        )
                        con.commit()
                    
                    print(f"     ✓ Fuzzy matches: {fuzzy_matches:,}")
        
        total_matches = exact_matches + salt_stripped_matches + combo_matches + fuzzy_matches
        
        with con.cursor() as cur:
            log_audit(cur, 'PHASE_4', 'map_clinical_trials', total_matches, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            # Get summary stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT nct_id) as trials_mapped,
                    COUNT(DISTINCT concept_id) as concepts_mapped
                FROM map_ctgov_molecules
            """)
            stats = cur.fetchone()
        
        print(f"\n  📊 Trial Mapping Summary:")
        print(f"     Total mappings:      {stats[0]:,}")
        print(f"     Trials mapped:       {stats[1]:,}")
        print(f"     Concepts in trials:  {stats[2]:,}")


# ============================================================================
# PHASE 5: UNIFIED ANALYTICS VIEW (RAW ACTIVITIES)
# ============================================================================

def create_materialized_analytics_view(config: DatabaseConfig, limit: Optional[int] = None):
    """
    Create materialized view for compound-target activities.
    
    Args:
        config: Database configuration
        limit: Maximum activities to include (for fast/debug mode)
    """
    print("\n🔗 PHASE 5: Creating Materialized Analytics View...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Activities may be limited based on available molecules")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            print("  🧹 Dropping existing view if exists...")
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS dm_compound_target_activity CASCADE;")
            con.commit()
            
            print("  🏗️  Creating materialized view (this may take several minutes)...")
            cur.execute("""
                CREATE MATERIALIZED VIEW dm_compound_target_activity AS
                
                -- ChEMBL Activities
                SELECT 
                    m.mol_id,
                    m.concept_id,
                    m.pref_name as molecule_name,
                    mc.preferred_name as concept_name,
                    t.target_id,
                    t.gene_symbol,
                    'CHEMBL' as source,
                    ass.assay_id,
                    ass.assay_type,
                    act.standard_type as activity_type,
                    act.standard_value as activity_value,
                    act.standard_units as activity_units,
                    act.pchembl_value
                FROM dm_molecule m
                LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                JOIN molecule_dictionary md ON m.chembl_id = md.chembl_id
                JOIN activities act ON md.molregno = act.molregno
                JOIN assays ass ON act.assay_id = ass.assay_id
                JOIN dm_target t ON ass.tid = t.chembl_tid
                WHERE act.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50', 'IC90', 'GI50')
                  AND act.standard_value IS NOT NULL
                  AND act.standard_value > 0
                  AND act.standard_value < 1000000000
                  AND t.gene_symbol IS NOT NULL
                  AND m.concept_id IS NOT NULL
                
                UNION ALL
                
                -- BindingDB Ki
                SELECT
                    m.mol_id, m.concept_id, m.pref_name, mc.preferred_name,
                    t.target_id, t.gene_symbol, 'BINDINGDB', 
                    NULL::bigint, 'B', 'Ki', ba.ki_nm, 'nM', ba.pchembl_value
                FROM dm_molecule m
                LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                JOIN bindingdb_activities ba ON m.bindingdb_monomer_id = ba.molecule_id
                JOIN dm_target t ON ba.target_id = t.bindingdb_target_id
                WHERE ba.ki_nm IS NOT NULL
                  AND ba.ki_nm > 0
                  AND ba.ki_nm < 1000000000
                  AND t.gene_symbol IS NOT NULL
                  AND m.concept_id IS NOT NULL
                
                UNION ALL
                
                -- BindingDB IC50
                SELECT
                    m.mol_id, m.concept_id, m.pref_name, mc.preferred_name,
                    t.target_id, t.gene_symbol, 'BINDINGDB', 
                    NULL::bigint, 'F', 'IC50', ba.ic50_nm, 'nM', ba.pchembl_value
                FROM dm_molecule m
                LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                JOIN bindingdb_activities ba ON m.bindingdb_monomer_id = ba.molecule_id
                JOIN dm_target t ON ba.target_id = t.bindingdb_target_id
                WHERE ba.ic50_nm IS NOT NULL
                  AND ba.ic50_nm > 0
                  AND ba.ic50_nm < 1000000000
                  AND t.gene_symbol IS NOT NULL
                  AND m.concept_id IS NOT NULL

                UNION ALL
                
                -- BindingDB Kd
                SELECT
                    m.mol_id, m.concept_id, m.pref_name, mc.preferred_name,
                    t.target_id, t.gene_symbol, 'BINDINGDB', 
                    NULL::bigint, 'B', 'Kd', ba.kd_nm, 'nM', ba.pchembl_value
                FROM dm_molecule m
                LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                JOIN bindingdb_activities ba ON m.bindingdb_monomer_id = ba.molecule_id
                JOIN dm_target t ON ba.target_id = t.bindingdb_target_id
                WHERE ba.kd_nm IS NOT NULL
                  AND ba.kd_nm > 0
                  AND ba.kd_nm < 1000000000
                  AND t.gene_symbol IS NOT NULL
                  AND m.concept_id IS NOT NULL;
            """)
            con.commit()
            
            print("  ⚡ Building indexes...")
            cur.execute("""
                CREATE INDEX idx_cta_gene_symbol ON dm_compound_target_activity(gene_symbol);
                CREATE INDEX idx_cta_mol_id ON dm_compound_target_activity(mol_id);
                CREATE INDEX idx_cta_concept_id ON dm_compound_target_activity(concept_id);
                CREATE INDEX idx_cta_potency ON dm_compound_target_activity(pchembl_value DESC NULLS LAST);
                CREATE INDEX idx_cta_activity_type ON dm_compound_target_activity(activity_type);
                CREATE INDEX idx_cta_gene_potency ON dm_compound_target_activity(gene_symbol, pchembl_value DESC);
                CREATE INDEX idx_cta_source ON dm_compound_target_activity(source);
                CREATE INDEX idx_cta_concept_gene ON dm_compound_target_activity(concept_id, gene_symbol);
            """)
            con.commit()
            
            # Get row count
            cur.execute("SELECT COUNT(*) FROM dm_compound_target_activity")
            row_count = cur.fetchone()[0]
            
            log_audit(cur, 'PHASE_5', 'create_activity_view', row_count, time.time() - start_time, 'SUCCESS')
            con.commit()
    
    print(f"  ✅ Materialized view created with {row_count:,} rows")


# ============================================================================
# PHASE 5B: UNIFIED ANALYTICS SUMMARY TABLE (DEDUPLICATED)
# ============================================================================

def create_materialized_analytics_view_summary(config: DatabaseConfig):
    """Create a deduplicated summary table for compound-target activities."""
    print("\n🔗 PHASE 5B: Creating Analytics Summary Table...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            print("  🧹 Dropping existing summary table...")
            cur.execute("DROP TABLE IF EXISTS dm_molecule_target_summary CASCADE;")
            con.commit()
            
            print("  📋 Creating summary table schema...")
            cur.execute("""
                CREATE TABLE dm_molecule_target_summary (
                    id BIGSERIAL PRIMARY KEY,
                    concept_id BIGINT REFERENCES dm_molecule_concept(concept_id),
                    concept_name TEXT,
                    representative_mol_id BIGINT,
                    representative_smiles TEXT,
                    target_id BIGINT REFERENCES dm_target(target_id),
                    gene_symbol TEXT NOT NULL,
                    target_name TEXT,
                    target_organism TEXT DEFAULT 'Homo sapiens',
                    best_ic50_nm NUMERIC,
                    best_ki_nm NUMERIC,
                    best_kd_nm NUMERIC,
                    best_ec50_nm NUMERIC,
                    best_pchembl NUMERIC,
                    median_ic50_nm NUMERIC,
                    median_ki_nm NUMERIC,
                    median_kd_nm NUMERIC,
                    median_ec50_nm NUMERIC,
                    ic50_min_nm NUMERIC,
                    ic50_max_nm NUMERIC,
                    ki_min_nm NUMERIC,
                    ki_max_nm NUMERIC,
                    kd_min_nm NUMERIC,
                    kd_max_nm NUMERIC,
                    ec50_min_nm NUMERIC,
                    ec50_max_nm NUMERIC,
                    n_ic50_measurements INTEGER DEFAULT 0,
                    n_ki_measurements INTEGER DEFAULT 0,
                    n_kd_measurements INTEGER DEFAULT 0,
                    n_ec50_measurements INTEGER DEFAULT 0,
                    n_total_measurements INTEGER DEFAULT 0,
                    sources TEXT[],
                    n_sources INTEGER DEFAULT 1,
                    measurement_consistency TEXT,
                    data_confidence TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(concept_id, target_id)
                );
            """)
            con.commit()
            
            print("  🔄 Aggregating activities...")
            cur.execute("""
                WITH representative AS (
                    SELECT DISTINCT ON (concept_id)
                        concept_id,
                        mol_id as representative_mol_id,
                        canonical_smiles as representative_smiles
                    FROM dm_molecule
                    WHERE is_salt = FALSE
                      AND canonical_smiles IS NOT NULL
                    ORDER BY concept_id, mol_id
                ),
                
                aggregated AS (
                    SELECT
                        concept_id,
                        target_id,
                        gene_symbol,
                        
                        MIN(CASE WHEN activity_type = 'IC50' THEN activity_value END) as best_ic50_nm,
                        MIN(CASE WHEN activity_type = 'Ki' THEN activity_value END) as best_ki_nm,
                        MIN(CASE WHEN activity_type = 'Kd' THEN activity_value END) as best_kd_nm,
                        MIN(CASE WHEN activity_type = 'EC50' THEN activity_value END) as best_ec50_nm,
                        MAX(pchembl_value) as best_pchembl,
                        
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE WHEN activity_type = 'IC50' THEN activity_value END) as median_ic50_nm,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE WHEN activity_type = 'Ki' THEN activity_value END) as median_ki_nm,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE WHEN activity_type = 'Kd' THEN activity_value END) as median_kd_nm,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE WHEN activity_type = 'EC50' THEN activity_value END) as median_ec50_nm,
                        
                        MIN(CASE WHEN activity_type = 'IC50' THEN activity_value END) as ic50_min_nm,
                        MAX(CASE WHEN activity_type = 'IC50' THEN activity_value END) as ic50_max_nm,
                        MIN(CASE WHEN activity_type = 'Ki' THEN activity_value END) as ki_min_nm,
                        MAX(CASE WHEN activity_type = 'Ki' THEN activity_value END) as ki_max_nm,
                        MIN(CASE WHEN activity_type = 'Kd' THEN activity_value END) as kd_min_nm,
                        MAX(CASE WHEN activity_type = 'Kd' THEN activity_value END) as kd_max_nm,
                        MIN(CASE WHEN activity_type = 'EC50' THEN activity_value END) as ec50_min_nm,
                        MAX(CASE WHEN activity_type = 'EC50' THEN activity_value END) as ec50_max_nm,
                        
                        COUNT(CASE WHEN activity_type = 'IC50' THEN 1 END) as n_ic50_measurements,
                        COUNT(CASE WHEN activity_type = 'Ki' THEN 1 END) as n_ki_measurements,
                        COUNT(CASE WHEN activity_type = 'Kd' THEN 1 END) as n_kd_measurements,
                        COUNT(CASE WHEN activity_type = 'EC50' THEN 1 END) as n_ec50_measurements,
                        COUNT(*) as n_total_measurements,
                        
                        ARRAY_AGG(DISTINCT source) as sources,
                        COUNT(DISTINCT source) as n_sources
                        
                    FROM dm_compound_target_activity
                    WHERE concept_id IS NOT NULL
                      AND gene_symbol IS NOT NULL
                      AND activity_value > 0
                    GROUP BY concept_id, target_id, gene_symbol
                )
                
                INSERT INTO dm_molecule_target_summary (
                    concept_id, concept_name, representative_mol_id, representative_smiles,
                    target_id, gene_symbol, target_name, target_organism,
                    best_ic50_nm, best_ki_nm, best_kd_nm, best_ec50_nm, best_pchembl,
                    median_ic50_nm, median_ki_nm, median_kd_nm, median_ec50_nm,
                    ic50_min_nm, ic50_max_nm, ki_min_nm, ki_max_nm,
                    kd_min_nm, kd_max_nm, ec50_min_nm, ec50_max_nm,
                    n_ic50_measurements, n_ki_measurements, n_kd_measurements, n_ec50_measurements, n_total_measurements,
                    sources, n_sources,
                    measurement_consistency, data_confidence
                )
                SELECT
                    a.concept_id,
                    mc.preferred_name,
                    r.representative_mol_id,
                    r.representative_smiles,
                    a.target_id,
                    a.gene_symbol,
                    t.protein_name,
                    t.organism,
                    a.best_ic50_nm, a.best_ki_nm, a.best_kd_nm, a.best_ec50_nm, a.best_pchembl,
                    a.median_ic50_nm, a.median_ki_nm, a.median_kd_nm, a.median_ec50_nm,
                    a.ic50_min_nm, a.ic50_max_nm, a.ki_min_nm, a.ki_max_nm,
                    a.kd_min_nm, a.kd_max_nm, a.ec50_min_nm, a.ec50_max_nm,
                    a.n_ic50_measurements, a.n_ki_measurements, a.n_kd_measurements, a.n_ec50_measurements, a.n_total_measurements,
                    a.sources, a.n_sources,
                    CASE 
                        WHEN COALESCE(a.ic50_max_nm / NULLIF(a.ic50_min_nm, 0), a.ki_max_nm / NULLIF(a.ki_min_nm, 0), 1) <= 3 THEN 'HIGH'
                        WHEN COALESCE(a.ic50_max_nm / NULLIF(a.ic50_min_nm, 0), a.ki_max_nm / NULLIF(a.ki_min_nm, 0), 1) <= 10 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END,
                    CASE 
                        WHEN a.n_sources >= 2 AND a.n_total_measurements >= 3 THEN 'HIGH'
                        WHEN a.n_sources >= 1 AND a.n_total_measurements >= 2 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END
                FROM aggregated a
                JOIN dm_molecule_concept mc ON a.concept_id = mc.concept_id
                JOIN dm_target t ON a.target_id = t.target_id
                LEFT JOIN representative r ON a.concept_id = r.concept_id;
            """)
            rows_inserted = cur.rowcount
            con.commit()
            
            print("  ⚡ Creating indexes...")
            cur.execute("""
                CREATE INDEX idx_mts_concept_id ON dm_molecule_target_summary(concept_id);
                CREATE INDEX idx_mts_target_id ON dm_molecule_target_summary(target_id);
                CREATE INDEX idx_mts_gene_symbol ON dm_molecule_target_summary(gene_symbol);
                CREATE INDEX idx_mts_concept_name ON dm_molecule_target_summary(concept_name);
                CREATE INDEX idx_mts_concept_name_trgm ON dm_molecule_target_summary USING gin(concept_name gin_trgm_ops);
                CREATE INDEX idx_mts_best_pchembl ON dm_molecule_target_summary(best_pchembl DESC NULLS LAST);
                CREATE INDEX idx_mts_best_ic50 ON dm_molecule_target_summary(best_ic50_nm ASC NULLS LAST);
                CREATE INDEX idx_mts_gene_pchembl ON dm_molecule_target_summary(gene_symbol, best_pchembl DESC NULLS LAST);
            """)
            con.commit()
            
            log_audit(cur, 'PHASE_5B', 'create_summary_table', rows_inserted, time.time() - start_time, 'SUCCESS')
            con.commit()
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT concept_id) as unique_concepts,
                    COUNT(DISTINCT gene_symbol) as unique_targets,
                    ROUND(AVG(n_total_measurements)::NUMERIC, 2) as avg_measurements,
                    SUM(CASE WHEN data_confidence = 'HIGH' THEN 1 ELSE 0 END) as high_confidence
                FROM dm_molecule_target_summary
            """)
            stats = cur.fetchone()
            
    print(f"  ✅ Summary table created with {rows_inserted:,} rows")
    print(f"     Unique concepts: {stats[1]:,}")
    print(f"     Unique targets: {stats[2]:,}")
    print(f"     High confidence: {stats[4]:,}")


# ============================================================================
# PHASE 6: CREATE SEARCH FUNCTIONS
# ============================================================================

def create_search_functions(config: DatabaseConfig):
    """Create SQL functions for similarity search and concept-based queries."""
    print("\n🔍 PHASE 6: Creating Search Functions...")
    
    sql_functions = r"""
    -- ========================================================================
    -- Drop existing functions for idempotency
    -- ========================================================================
    DROP FUNCTION IF EXISTS find_similar_molecules(TEXT, FLOAT, INT) CASCADE;
    DROP FUNCTION IF EXISTS get_drug_forms(TEXT) CASCADE;
    DROP FUNCTION IF EXISTS find_trials_for_drug_concept(TEXT, INT) CASCADE;
    DROP FUNCTION IF EXISTS find_drug_targets(TEXT, FLOAT, TEXT) CASCADE;
    DROP FUNCTION IF EXISTS find_target_drugs(TEXT, FLOAT, INT) CASCADE;
    DROP FUNCTION IF EXISTS get_drug_activity_profile(TEXT) CASCADE;
    DROP FUNCTION IF EXISTS compare_drugs_on_target(TEXT, TEXT[], TEXT) CASCADE;
    DROP FUNCTION IF EXISTS find_trials_by_similar_molecules(TEXT, FLOAT, INT) CASCADE;
    DROP FUNCTION IF EXISTS find_molecules_by_substructure(TEXT, INT) CASCADE;
    DROP FUNCTION IF EXISTS get_concept_summary(TEXT) CASCADE;
    DROP FUNCTION IF EXISTS validate_molecule_hierarchy() CASCADE;

    -- ========================================================================
    -- Find similar molecules by SMILES
    -- Uses % operator (not %%) for fingerprint similarity
    -- ========================================================================
    CREATE FUNCTION find_similar_molecules(
        query_smiles TEXT,
        similarity_threshold FLOAT DEFAULT 0.5,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        mol_id BIGINT,
        concept_id BIGINT,
        pref_name TEXT,
        concept_name TEXT,
        chembl_id TEXT,
        canonical_smiles TEXT,
        tanimoto_similarity FLOAT,
        is_salt BOOLEAN,
        salt_form TEXT,
        stereo_type TEXT
    ) AS $$
    DECLARE
        query_fp bfp;
        query_mol mol;
    BEGIN
        query_mol := mol_from_smiles(query_smiles::cstring);
        IF query_mol IS NULL THEN
            RAISE EXCEPTION 'Invalid SMILES: %', query_smiles;
        END IF;
        
        query_fp := morganbv_fp(query_mol);
        
        -- Set similarity threshold
        EXECUTE format('SET rdkit.tanimoto_threshold = %s', similarity_threshold);
        
        RETURN QUERY
        SELECT 
            dm.mol_id,
            dm.concept_id,
            dm.pref_name::TEXT,
            mc.preferred_name::TEXT as concept_name,
            dm.chembl_id::TEXT,
            dm.canonical_smiles::TEXT,
            tanimoto_sml(query_fp, dm.mfp2)::FLOAT as tanimoto_similarity,
            dm.is_salt,
            dm.salt_form::TEXT,
            dm.stereo_type::TEXT
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        WHERE dm.mfp2 % query_fp  -- Single % for similarity search
        ORDER BY tanimoto_sml(query_fp, dm.mfp2) DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find all forms of a drug concept
    -- ========================================================================
    CREATE FUNCTION get_drug_forms(drug_name TEXT)
    RETURNS TABLE (
        concept_id BIGINT,
        concept_name TEXT,
        mol_id BIGINT,
        form_name TEXT,
        inchi_key TEXT,
        is_salt BOOLEAN,
        salt_form TEXT,
        stereo_type TEXT,
        chembl_id TEXT,
        drugcentral_id INTEGER
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mc.concept_id,
            mc.preferred_name::TEXT,
            dm.mol_id,
            dm.pref_name::TEXT,
            dm.inchi_key::TEXT,
            dm.is_salt,
            dm.salt_form::TEXT,
            dm.stereo_type::TEXT,
            dm.chembl_id::TEXT,
            dm.drugcentral_id
        FROM dm_molecule_concept mc
        JOIN dm_molecule dm ON mc.concept_id = dm.concept_id
        WHERE mc.preferred_name ILIKE '%' || drug_name || '%'
           OR mc.concept_id IN (
               SELECT DISTINCT s.concept_id 
               FROM dm_molecule_synonyms s
               WHERE s.synonym_lower ILIKE '%' || lower(drug_name) || '%'
           )
        ORDER BY mc.preferred_name, dm.is_salt, dm.stereo_type;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find trials for a drug concept (all forms)
    -- ========================================================================
    CREATE FUNCTION find_trials_for_drug_concept(
        drug_name TEXT,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        concept_name TEXT,
        mol_id BIGINT,
        molecule_form TEXT,
        salt_form TEXT,
        nct_id TEXT,
        trial_title TEXT,
        trial_status TEXT,
        phase TEXT
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT DISTINCT
            mc.preferred_name::TEXT as concept_name,
            map.mol_id,
            dm.pref_name::TEXT as molecule_form,
            dm.salt_form::TEXT,
            s.nct_id::TEXT,
            s.brief_title::TEXT as trial_title,
            s.overall_status::TEXT as trial_status,
            s.phase::TEXT
        FROM dm_molecule_concept mc
        JOIN map_ctgov_molecules map ON map.concept_id = mc.concept_id
        LEFT JOIN dm_molecule dm ON map.mol_id = dm.mol_id
        JOIN ctgov_studies s ON map.nct_id = s.nct_id
        WHERE mc.preferred_name ILIKE '%' || drug_name || '%'
           OR mc.concept_id IN (
               SELECT DISTINCT syn.concept_id 
               FROM dm_molecule_synonyms syn
               WHERE syn.synonym_lower ILIKE '%' || lower(drug_name) || '%'
           )
        ORDER BY s.nct_id::TEXT DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find targets for a drug (SUMMARY TABLE - deduplicated)
    -- ========================================================================
    CREATE FUNCTION find_drug_targets(
        drug_name TEXT,
        min_pchembl FLOAT DEFAULT 5.0,
        min_confidence TEXT DEFAULT 'LOW'
    )
    RETURNS TABLE (
        concept_name TEXT,
        gene_symbol TEXT,
        best_ic50_nm NUMERIC,
        best_ki_nm NUMERIC,
        best_pchembl NUMERIC,
        n_measurements INTEGER,
        measurement_consistency TEXT,
        data_confidence TEXT,
        sources TEXT[]
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mts.concept_name::TEXT,
            mts.gene_symbol::TEXT,
            mts.best_ic50_nm,
            mts.best_ki_nm,
            mts.best_pchembl,
            mts.n_total_measurements,
            mts.measurement_consistency::TEXT,
            mts.data_confidence::TEXT,
            mts.sources
        FROM dm_molecule_target_summary mts
        WHERE (mts.concept_name ILIKE '%' || drug_name || '%'
               OR mts.concept_id IN (
                   SELECT DISTINCT syn.concept_id 
                   FROM dm_molecule_synonyms syn
                   WHERE syn.synonym_lower ILIKE '%' || lower(drug_name) || '%'
               ))
          AND (mts.best_pchembl >= min_pchembl OR mts.best_pchembl IS NULL)
          AND (
              (min_confidence = 'LOW') OR
              (min_confidence = 'MEDIUM' AND mts.data_confidence IN ('MEDIUM', 'HIGH')) OR
              (min_confidence = 'HIGH' AND mts.data_confidence = 'HIGH')
          )
        ORDER BY mts.best_pchembl DESC NULLS LAST;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find drugs for a target (e.g., all JAK2 inhibitors)
    -- ========================================================================
    CREATE FUNCTION find_target_drugs(
        target_gene TEXT,
        min_pchembl FLOAT DEFAULT 6.0,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        concept_name TEXT,
        representative_smiles TEXT,
        best_ic50_nm NUMERIC,
        best_ki_nm NUMERIC,
        best_pchembl NUMERIC,
        n_measurements INTEGER,
        data_confidence TEXT,
        sources TEXT[]
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mts.concept_name::TEXT,
            mts.representative_smiles::TEXT,
            mts.best_ic50_nm,
            mts.best_ki_nm,
            mts.best_pchembl,
            mts.n_total_measurements,
            mts.data_confidence::TEXT,
            mts.sources
        FROM dm_molecule_target_summary mts
        WHERE mts.gene_symbol = upper(target_gene)
          AND mts.best_pchembl >= min_pchembl
        ORDER BY mts.best_pchembl DESC NULLS LAST
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Get activity profile for a drug (simplified - no array columns)
    -- ========================================================================
    CREATE FUNCTION get_drug_activity_profile(drug_name TEXT)
    RETURNS TABLE (
        concept_name TEXT,
        gene_symbol TEXT,
        target_name TEXT,
        best_ic50_nm NUMERIC,
        best_ki_nm NUMERIC,
        best_kd_nm NUMERIC,
        best_ec50_nm NUMERIC,
        best_pchembl NUMERIC,
        n_measurements INTEGER,
        data_confidence TEXT,
        sources TEXT[]
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mts.concept_name::TEXT,
            mts.gene_symbol::TEXT,
            mts.target_name::TEXT,
            mts.best_ic50_nm,
            mts.best_ki_nm,
            mts.best_kd_nm,
            mts.best_ec50_nm,
            mts.best_pchembl,
            mts.n_total_measurements,
            mts.data_confidence::TEXT,
            mts.sources
        FROM dm_molecule_target_summary mts
        WHERE mts.concept_name ILIKE '%' || drug_name || '%'
           OR mts.concept_id IN (
               SELECT DISTINCT syn.concept_id 
               FROM dm_molecule_synonyms syn
               WHERE syn.synonym_lower ILIKE '%' || lower(drug_name) || '%'
           )
        ORDER BY mts.best_pchembl DESC NULLS LAST;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Compare drugs on a specific target
    -- ========================================================================
    CREATE FUNCTION compare_drugs_on_target(
        target_gene TEXT,
        drug_names TEXT[],
        activity_type TEXT DEFAULT 'IC50'
    )
    RETURNS TABLE (
        concept_name TEXT,
        best_value_nm NUMERIC,
        median_value_nm NUMERIC,
        n_measurements INTEGER,
        fold_vs_best NUMERIC,
        data_confidence TEXT
    ) AS $$
    DECLARE
        best_overall NUMERIC;
    BEGIN
        -- Find the best (lowest) value among the drugs
        SELECT MIN(
            CASE activity_type
                WHEN 'IC50' THEN mts.best_ic50_nm
                WHEN 'Ki' THEN mts.best_ki_nm
                WHEN 'Kd' THEN mts.best_kd_nm
                ELSE mts.best_ic50_nm
            END
        ) INTO best_overall
        FROM dm_molecule_target_summary mts
        WHERE mts.gene_symbol = upper(target_gene)
          AND mts.concept_name ILIKE ANY(SELECT '%' || unnest(drug_names) || '%');
        
        RETURN QUERY
        SELECT 
            mts.concept_name::TEXT,
            CASE activity_type
                WHEN 'IC50' THEN mts.best_ic50_nm
                WHEN 'Ki' THEN mts.best_ki_nm
                WHEN 'Kd' THEN mts.best_kd_nm
                ELSE mts.best_ic50_nm
            END as best_value,
            CASE activity_type
                WHEN 'IC50' THEN mts.median_ic50_nm
                WHEN 'Ki' THEN mts.median_ki_nm
                WHEN 'Kd' THEN mts.median_kd_nm
                ELSE mts.median_ic50_nm
            END as median_value,
            CASE activity_type
                WHEN 'IC50' THEN mts.n_ic50_measurements
                WHEN 'Ki' THEN mts.n_ki_measurements
                WHEN 'Kd' THEN mts.n_kd_measurements
                ELSE mts.n_ic50_measurements
            END as n_meas,
            ROUND(
                CASE activity_type
                    WHEN 'IC50' THEN mts.best_ic50_nm
                    WHEN 'Ki' THEN mts.best_ki_nm
                    WHEN 'Kd' THEN mts.best_kd_nm
                    ELSE mts.best_ic50_nm
                END / NULLIF(best_overall, 0), 
            2) as fold_diff,
            mts.data_confidence::TEXT
        FROM dm_molecule_target_summary mts
        WHERE mts.gene_symbol = upper(target_gene)
          AND mts.concept_name ILIKE ANY(SELECT '%' || unnest(drug_names) || '%')
        ORDER BY best_value ASC NULLS LAST;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find trials by structure similarity
    -- ========================================================================
    CREATE FUNCTION find_trials_by_similar_molecules(
        query_smiles TEXT,
        similarity_threshold FLOAT DEFAULT 0.7,
        max_results INT DEFAULT 50
    )
    RETURNS TABLE (
        mol_id BIGINT,
        molecule_name TEXT,
        concept_name TEXT,
        tanimoto_similarity FLOAT,
        nct_id TEXT,
        trial_title TEXT,
        trial_status TEXT,
        phase TEXT
    ) AS $$
    DECLARE
        query_fp bfp;
        query_mol mol;
    BEGIN
        query_mol := mol_from_smiles(query_smiles::cstring);
        IF query_mol IS NULL THEN
            RAISE EXCEPTION 'Invalid SMILES: %', query_smiles;
        END IF;
        
        query_fp := morganbv_fp(query_mol);
        EXECUTE format('SET rdkit.tanimoto_threshold = %s', similarity_threshold);
        
        RETURN QUERY
        SELECT DISTINCT
            dm.mol_id,
            dm.pref_name::TEXT as molecule_name,
            mc.preferred_name::TEXT as concept_name,
            tanimoto_sml(query_fp, dm.mfp2)::FLOAT as tanimoto_similarity,
            s.nct_id::TEXT,
            s.brief_title::TEXT as trial_title,
            s.overall_status::TEXT as trial_status,
            s.phase::TEXT
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        JOIN map_ctgov_molecules map ON dm.concept_id = map.concept_id
        JOIN ctgov_studies s ON map.nct_id = s.nct_id
        WHERE dm.mfp2 % query_fp  -- Single % for similarity
        ORDER BY tanimoto_sml(query_fp, dm.mfp2) DESC, s.nct_id::TEXT DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Substructure search
    -- ========================================================================
    CREATE FUNCTION find_molecules_by_substructure(
        query_smarts TEXT,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        mol_id BIGINT,
        concept_id BIGINT,
        pref_name TEXT,
        concept_name TEXT,
        chembl_id TEXT,
        canonical_smiles TEXT,
        is_salt BOOLEAN,
        salt_form TEXT
    ) AS $$
    DECLARE
        query_mol qmol;
    BEGIN
        query_mol := qmol_from_smarts(query_smarts::cstring);
        IF query_mol IS NULL THEN
            RAISE EXCEPTION 'Invalid SMARTS: %', query_smarts;
        END IF;
        
        RETURN QUERY
        SELECT 
            dm.mol_id,
            dm.concept_id,
            dm.pref_name::TEXT,
            mc.preferred_name::TEXT as concept_name,
            dm.chembl_id::TEXT,
            dm.canonical_smiles::TEXT,
            dm.is_salt,
            dm.salt_form::TEXT
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        WHERE dm.mol @> query_mol
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Get concept summary
    -- ========================================================================
    CREATE FUNCTION get_concept_summary(drug_name TEXT)
    RETURNS TABLE (
        concept_id BIGINT,
        concept_name TEXT,
        parent_inchi_key_14 TEXT,
        n_forms INTEGER,
        n_stereo_variants BIGINT,
        n_salt_forms BIGINT,
        n_trials BIGINT,
        n_targets BIGINT,
        all_forms JSONB
    ) AS $$
    BEGIN
        RETURN QUERY
        WITH concept_match AS (
            SELECT DISTINCT mc.concept_id
            FROM dm_molecule_concept mc
            LEFT JOIN dm_molecule_synonyms dms ON mc.concept_id = dms.concept_id
            WHERE mc.preferred_name ILIKE '%' || drug_name || '%'
               OR dms.synonym_lower ILIKE '%' || lower(drug_name) || '%'
        )
        SELECT 
            mc.concept_id,
            mc.preferred_name::TEXT,
            mc.parent_inchi_key_14::TEXT,
            mc.n_forms,
            (SELECT COUNT(DISTINCT stereo_id) FROM dm_molecule dm2 WHERE dm2.concept_id = mc.concept_id)::BIGINT,
            (SELECT COUNT(*) FROM dm_molecule dm2 WHERE dm2.concept_id = mc.concept_id AND dm2.is_salt = TRUE)::BIGINT,
            (SELECT COUNT(DISTINCT m.nct_id) FROM map_ctgov_molecules m WHERE m.concept_id = mc.concept_id)::BIGINT,
            (SELECT COUNT(DISTINCT mts.gene_symbol) FROM dm_molecule_target_summary mts WHERE mts.concept_id = mc.concept_id)::BIGINT,
            (SELECT jsonb_agg(jsonb_build_object(
                'mol_id', dm2.mol_id,
                'name', dm2.pref_name,
                'inchi_key', dm2.inchi_key,
                'salt_form', dm2.salt_form,
                'stereo_type', dm2.stereo_type,
                'chembl_id', dm2.chembl_id
            )) FROM dm_molecule dm2 WHERE dm2.concept_id = mc.concept_id)
        FROM dm_molecule_concept mc
        JOIN concept_match cm ON mc.concept_id = cm.concept_id;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Validate molecule hierarchy
    -- ========================================================================
    CREATE FUNCTION validate_molecule_hierarchy()
    RETURNS TABLE (
        check_name TEXT,
        status TEXT,
        count BIGINT,
        details TEXT
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 'orphaned_molecules'::TEXT, 
               CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
               COUNT(*)::BIGINT,
               'Molecules without concept_id'::TEXT
        FROM dm_molecule WHERE concept_id IS NULL AND parent_inchi_key_14 IS NOT NULL;
        
        RETURN QUERY
        SELECT 'empty_concepts'::TEXT,
               CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
               COUNT(*)::BIGINT,
               'Concepts with no molecules'::TEXT
        FROM dm_molecule_concept mc
        WHERE NOT EXISTS (SELECT 1 FROM dm_molecule dm WHERE dm.concept_id = mc.concept_id);
        
        RETURN QUERY
        SELECT 'ambiguous_synonyms'::TEXT,
               CASE WHEN COUNT(*) < 1000 THEN 'PASS' ELSE 'WARN' END::TEXT,
               COUNT(*)::BIGINT,
               'Synonyms pointing to multiple concepts'::TEXT
        FROM (
            SELECT synonym_lower, COUNT(DISTINCT dms.concept_id) as n
            FROM dm_molecule_synonyms dms
            WHERE dms.concept_id IS NOT NULL
            GROUP BY synonym_lower
            HAVING COUNT(DISTINCT dms.concept_id) > 1
        ) x;
        
        RETURN QUERY
        SELECT 'summary_table_populated'::TEXT,
               CASE WHEN (SELECT COUNT(*) FROM dm_molecule_target_summary) > 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
               (SELECT COUNT(*) FROM dm_molecule_target_summary)::BIGINT,
               'Rows in summary table'::TEXT;
        
        RETURN QUERY
        SELECT 'trial_mappings'::TEXT,
               CASE WHEN (SELECT COUNT(*) FROM map_ctgov_molecules) > 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
               (SELECT COUNT(*) FROM map_ctgov_molecules)::BIGINT,
               'Trial-molecule mappings'::TEXT;
        
        RETURN QUERY
        SELECT 'activity_view'::TEXT,
               CASE WHEN (SELECT COUNT(*) FROM dm_compound_target_activity) > 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
               (SELECT COUNT(*) FROM dm_compound_target_activity)::BIGINT,
               'Rows in activity view'::TEXT;
               
        RETURN QUERY
        SELECT 'molecules_with_fps'::TEXT,
               CASE WHEN (SELECT COUNT(*) FROM dm_molecule dm WHERE dm.mfp2 IS NOT NULL) > 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
               (SELECT COUNT(*) FROM dm_molecule dm WHERE dm.mfp2 IS NOT NULL)::BIGINT,
               'Molecules with Morgan fingerprints'::TEXT;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(sql_functions)
            con.commit()
    
    print("  ✅ Search functions created")


# ============================================================================
# PHASE 9: POPULATE INDICATIONS FROM CHEMBL
# ============================================================================

def populate_indications(config: DatabaseConfig, limit: Optional[int] = None):
    """
    Load indication data from ChEMBL drug_indication table.
    
    Args:
        config: Database configuration
        limit: Maximum indications to load (for fast/debug mode)
    """
    print("\n💊 PHASE 9: Populating Indications from ChEMBL...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting to {limit:,} indications")
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # Check if drug_indication table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'drug_indication'
                )
            """)
            if not cur.fetchone()[0]:
                print("  ⚠️  ChEMBL drug_indication table not found - skipping")
                return
            
            # Create schema if not exists
            print("  📋 Ensuring indication schema exists...")
            cur.execute(CREATE_INDICATION_SCHEMA_SQL)
            con.commit()
            
            # Clear existing data for clean rebuild
            print("  🧹 Clearing existing indication data...")
            cur.execute("TRUNCATE dm_drug_indication RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE dm_indication RESTART IDENTITY CASCADE;")
            con.commit()
            
            # --- 9.1 Load unique indications (MeSH + EFO) ---
            print("  📥 Loading unique indications...")
            cur.execute(f"""
                INSERT INTO dm_indication (
                    mesh_id,
                    mesh_heading,
                    efo_id,
                    efo_term,
                    preferred_name,
                    indication_class
                )
                SELECT DISTINCT ON (COALESCE(mesh_id, efo_id))
                    mesh_id,
                    mesh_heading,
                    efo_id,
                    efo_term,
                    COALESCE(mesh_heading, efo_term) as preferred_name,
                    'DISEASE' as indication_class
                FROM drug_indication
                WHERE (mesh_id IS NOT NULL OR efo_id IS NOT NULL)
                  AND (mesh_heading IS NOT NULL OR efo_term IS NOT NULL)
                ORDER BY COALESCE(mesh_id, efo_id), mesh_id NULLS LAST
                {limit_clause}
                ON CONFLICT (mesh_id) DO UPDATE SET
                    mesh_heading = COALESCE(EXCLUDED.mesh_heading, dm_indication.mesh_heading),
                    efo_id = COALESCE(EXCLUDED.efo_id, dm_indication.efo_id),
                    efo_term = COALESCE(EXCLUDED.efo_term, dm_indication.efo_term),
                    updated_at = NOW();
            """)
            indication_count = cur.rowcount
            con.commit()
            print(f"     ✓ Unique indications loaded: {indication_count:,}")
            
            # --- 9.2 Classify therapeutic areas ---
            print("  🏷️  Classifying therapeutic areas...")
            cur.execute("SELECT indication_id, mesh_heading, efo_term FROM dm_indication")
            indications = cur.fetchall()
            
            updates = []
            for ind_id, mesh_heading, efo_term in indications:
                area = classify_therapeutic_area(mesh_heading, efo_term)
                updates.append((area, ind_id))
            
            if updates:
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE dm_indication SET therapeutic_area = %s WHERE indication_id = %s",
                    updates,
                    page_size=1000
                )
                con.commit()
            
            # Count by area
            cur.execute("""
                SELECT therapeutic_area, COUNT(*) 
                FROM dm_indication 
                GROUP BY therapeutic_area 
                ORDER BY COUNT(*) DESC
            """)
            area_counts = cur.fetchall()
            print("     Therapeutic areas:")
            for area, count in area_counts[:10]:
                print(f"       {area}: {count:,}")
            
            # --- 9.3 Load drug-indication mappings ---
            print("\n  📥 Loading drug-indication mappings...")
            cur.execute(f"""
                INSERT INTO dm_drug_indication (
                    concept_id,
                    indication_id,
                    max_phase,
                    sources,
                    chembl_drugind_id
                )
                SELECT DISTINCT ON (dm.concept_id, di2.indication_id)
                    dm.concept_id,
                    di2.indication_id,
                    MAX(di.max_phase_for_ind) as max_phase,
                    ARRAY['CHEMBL'],
                    MIN(di.drugind_id)
                FROM drug_indication di
                JOIN molecule_dictionary md ON di.molregno = md.molregno
                JOIN dm_molecule dm ON md.chembl_id = dm.chembl_id
                JOIN dm_indication di2 ON (
                    di2.mesh_id = di.mesh_id 
                    OR (di2.mesh_id IS NULL AND di2.efo_id = di.efo_id)
                )
                WHERE dm.concept_id IS NOT NULL
                  AND di2.indication_id IS NOT NULL
                GROUP BY dm.concept_id, di2.indication_id
                {limit_clause}
                ON CONFLICT (concept_id, indication_id) DO UPDATE SET
                    max_phase = GREATEST(dm_drug_indication.max_phase, EXCLUDED.max_phase),
                    sources = CASE 
                        WHEN 'CHEMBL' = ANY(dm_drug_indication.sources) 
                        THEN dm_drug_indication.sources
                        ELSE array_append(dm_drug_indication.sources, 'CHEMBL')
                    END;
            """)
            mapping_count = cur.rowcount
            con.commit()
            print(f"     ✓ Drug-indication mappings: {mapping_count:,}")
            
            # --- 9.4 Summary statistics ---
            cur.execute("""
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT concept_id) as unique_drugs,
                    COUNT(DISTINCT indication_id) as unique_indications,
                    COUNT(*) FILTER (WHERE is_approved) as approved_mappings,
                    COUNT(DISTINCT concept_id) FILTER (WHERE is_approved) as approved_drugs
                FROM dm_drug_indication
            """)
            stats = cur.fetchone()
            
            log_audit(cur, 'PHASE_9', 'populate_indications', mapping_count, 
                     time.time() - start_time, 'SUCCESS')
            con.commit()
            
            print(f"\n  📊 Indication Summary:")
            print(f"     Total mappings:       {stats[0]:,}")
            print(f"     Drugs with indications: {stats[1]:,}")
            print(f"     Unique indications:   {stats[2]:,}")
            print(f"     Approved mappings:    {stats[3]:,}")
            print(f"     Approved drugs:       {stats[4]:,}")


# ============================================================================
# PHASE 10: LINK DAILYMED PRODUCTS
# ============================================================================

# ============================================================================
# PHASE 10: LINK DAILYMED PRODUCTS (FIXED - prioritize generic_names)
# ============================================================================

def link_dailymed_products(config: DatabaseConfig, limit: Optional[int] = None):
    """
    Link DailyMed products to our molecule concepts using name matching.
    
    Matching priority:
      1. generic_names (most reliable - actual drug names)
      2. substances (active ingredients - may have salt forms)
      3. brand_names (trade names)
      4. product_name (fallback - extract drug name)
    
    Args:
        config: Database configuration
        limit: Maximum products to process (for fast/debug mode)
    """
    print("\n📋 PHASE 10: Linking DailyMed Products...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting to {limit:,} products")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # Check if dailymed tables exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dailymed_products'
                )
            """)
            if not cur.fetchone()[0]:
                print("  ⚠️  DailyMed tables not found - skipping")
                return
            
            # Check dailymed_products columns
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'dailymed_products'
            """)
            dm_columns = {row[0] for row in cur.fetchall()}
            print(f"  📋 DailyMed columns: {dm_columns}")
            
            # Clear existing mappings
            print("  🧹 Clearing existing DailyMed mappings...")
            cur.execute("TRUNCATE map_dailymed_molecules RESTART IDENTITY;")
            con.commit()
            
            # Load synonyms into memory for matching
            print("  🧠 Loading synonyms into memory...")
            cur.execute("""
                SELECT DISTINCT synonym_lower, mol_id, concept_id
                FROM dm_molecule_synonyms
                WHERE concept_id IS NOT NULL
                  AND LENGTH(synonym_lower) >= 3
            """)
            synonym_map = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            print(f"     ✓ Loaded {len(synonym_map):,} synonyms")
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            # Load DailyMed products
            print("  📥 Loading DailyMed products...")
            cur.execute(f"""
                SELECT 
                    id,
                    set_id,
                    product_name,
                    generic_names,
                    brand_names,
                    substances
                FROM dailymed_products
                WHERE set_id IS NOT NULL
                {limit_clause}
            """)
            products = cur.fetchall()
            print(f"     ✓ Found {len(products):,} products")
            
            # Sample a few products to understand the data
            print("  📋 Sample products (first 3):")
            for i, (pid, sid, pname, gnames, bnames, subs) in enumerate(products[:3]):
                print(f"     {i+1}. product_name: {pname}")
                print(f"        generic_names: {gnames}")
                print(f"        brand_names: {bnames}")
                print(f"        substances: {subs}")
            
            # Common salt suffixes to strip
            SALT_SUFFIXES_TO_STRIP = [
                ' hydrochloride', ' hcl', ' sodium', ' potassium', ' calcium',
                ' acetate', ' maleate', ' fumarate', ' succinate', ' tartrate',
                ' citrate', ' phosphate', ' sulfate', ' mesylate', ' besylate',
                ' bromide', ' chloride', ' nitrate', ' lactate', ' gluconate',
                ' monohydrate', ' dihydrate', ' trihydrate',
            ]
            
            def clean_and_match(text: str, synonym_map: dict) -> tuple:
                """Try to match text against synonym map with various normalizations."""
                if not text:
                    return None
                
                cleaned = text.lower().strip()
                
                # Direct match
                if cleaned in synonym_map:
                    return synonym_map[cleaned], cleaned
                
                # Try stripping salt suffixes
                for suffix in SALT_SUFFIXES_TO_STRIP:
                    if cleaned.endswith(suffix):
                        base = cleaned[:-len(suffix)].strip()
                        if base and base in synonym_map:
                            return synonym_map[base], base
                
                # Try removing parenthetical content
                if '(' in cleaned:
                    without_paren = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned).strip()
                    if without_paren and without_paren in synonym_map:
                        return synonym_map[without_paren], without_paren
                
                return None
            
            def parse_field(field_value: str) -> list:
                """Parse a field that might be semicolon or comma separated."""
                if not field_value:
                    return []
                
                field_str = str(field_value).strip()
                if not field_str:
                    return []
                
                # Try semicolon first, then comma
                if ';' in field_str:
                    return [s.strip() for s in field_str.split(';') if s.strip()]
                elif ',' in field_str:
                    return [s.strip() for s in field_str.split(',') if s.strip()]
                else:
                    return [field_str]
            
            # Match products to molecules
            print("  🔍 Matching products to molecules...")
            matches = []
            match_stats = {'GENERIC_NAME': 0, 'SUBSTANCE': 0, 'BRAND_NAME': 0, 'PRODUCT_NAME': 0}
            
            for prod_id, set_id, product_name, generic_names, brand_names, substances in tqdm(
                products, desc="     Matching", leave=False
            ):
                matched = False
                
                # Priority 1: Generic names (most reliable)
                if not matched:
                    for generic in parse_field(generic_names):
                        result = clean_and_match(generic, synonym_map)
                        if result:
                            (mol_id, concept_id), matched_name = result
                            matches.append((
                                set_id, product_name, concept_id, mol_id,
                                'GENERIC_NAME', generic, 1.0
                            ))
                            match_stats['GENERIC_NAME'] += 1
                            matched = True
                            break
                
                # Priority 2: Substances (active ingredients)
                if not matched:
                    for substance in parse_field(substances):
                        result = clean_and_match(substance, synonym_map)
                        if result:
                            (mol_id, concept_id), matched_name = result
                            matches.append((
                                set_id, product_name, concept_id, mol_id,
                                'SUBSTANCE', substance, 0.95
                            ))
                            match_stats['SUBSTANCE'] += 1
                            matched = True
                            break
                
                # Priority 3: Brand names
                if not matched:
                    for brand in parse_field(brand_names):
                        result = clean_and_match(brand, synonym_map)
                        if result:
                            (mol_id, concept_id), matched_name = result
                            matches.append((
                                set_id, product_name, concept_id, mol_id,
                                'BRAND_NAME', brand, 0.9
                            ))
                            match_stats['BRAND_NAME'] += 1
                            matched = True
                            break
                
                # Priority 4: Product name (fallback)
                if not matched and product_name:
                    name_str = str(product_name).strip()
                    
                    # Try the full product name first
                    result = clean_and_match(name_str, synonym_map)
                    if result:
                        (mol_id, concept_id), matched_name = result
                        matches.append((
                            set_id, product_name, concept_id, mol_id,
                            'PRODUCT_NAME', matched_name, 0.85
                        ))
                        match_stats['PRODUCT_NAME'] += 1
                        matched = True
                    
                    # Try first word (often the drug name)
                    if not matched:
                        first_word = name_str.lower().split()[0] if name_str.split() else ''
                        if first_word and len(first_word) >= 3 and first_word in synonym_map:
                            mol_id, concept_id = synonym_map[first_word]
                            matches.append((
                                set_id, product_name, concept_id, mol_id,
                                'PRODUCT_NAME', first_word, 0.8
                            ))
                            match_stats['PRODUCT_NAME'] += 1
                            matched = True
                    
                    # Try content in parentheses (often contains generic name)
                    if not matched:
                        paren_match = re.search(r'\(([^)]+)\)', name_str.lower())
                        if paren_match:
                            paren_content = paren_match.group(1).strip()
                            result = clean_and_match(paren_content, synonym_map)
                            if result:
                                (mol_id, concept_id), matched_name = result
                                matches.append((
                                    set_id, product_name, concept_id, mol_id,
                                    'PRODUCT_NAME', paren_content, 0.8
                                ))
                                match_stats['PRODUCT_NAME'] += 1
            
            print(f"     Match breakdown during processing:")
            for match_type, count in match_stats.items():
                print(f"       {match_type}: {count:,}")
            
            # Bulk insert matches
            if matches:
                print(f"  💾 Inserting {len(matches):,} matches...")
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO map_dailymed_molecules 
                       (set_id, product_name, concept_id, mol_id, match_type, matched_name, confidence)
                       VALUES %s
                       ON CONFLICT (set_id, concept_id) DO UPDATE SET
                         confidence = GREATEST(map_dailymed_molecules.confidence, EXCLUDED.confidence),
                         match_type = CASE 
                             WHEN EXCLUDED.confidence > map_dailymed_molecules.confidence 
                             THEN EXCLUDED.match_type 
                             ELSE map_dailymed_molecules.match_type 
                         END
                    """,
                    matches,
                    page_size=5000
                )
                con.commit()
            
            # Summary from database
            cur.execute("""
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT set_id) as products_mapped,
                    COUNT(DISTINCT concept_id) as concepts_linked,
                    COUNT(*) FILTER (WHERE match_type = 'GENERIC_NAME') as generic_matches,
                    COUNT(*) FILTER (WHERE match_type = 'SUBSTANCE') as substance_matches,
                    COUNT(*) FILTER (WHERE match_type = 'BRAND_NAME') as brand_matches,
                    COUNT(*) FILTER (WHERE match_type = 'PRODUCT_NAME') as product_matches
                FROM map_dailymed_molecules
            """)
            stats = cur.fetchone()
            
            log_audit(cur, 'PHASE_10', 'link_dailymed', stats[0], 
                     time.time() - start_time, 'SUCCESS')
            con.commit()
            
            print(f"\n  📊 DailyMed Linkage Summary:")
            print(f"     Total mappings:      {stats[0]:,}")
            print(f"     Products mapped:     {stats[1]:,}")
            print(f"     Concepts linked:     {stats[2]:,}")
            print(f"     By generic name:     {stats[3]:,}")
            print(f"     By substance:        {stats[4]:,}")
            print(f"     By brand name:       {stats[5]:,}")
            print(f"     By product name:     {stats[6]:,}")


# ============================================================================
# PHASE 11: LINK OPENFDA LABELS
# ============================================================================

# ============================================================================
# PHASE 11: LINK OPENFDA LABELS (FIXED)
# ============================================================================

def link_openfda_labels(config: DatabaseConfig, limit: Optional[int] = None):
    """
    Link OpenFDA labels to our molecule concepts using mapping tables.
    
    Args:
        config: Database configuration  
        limit: Maximum labels to process (for fast/debug mode)
    """
    print("\n📋 PHASE 11: Linking OpenFDA Labels...")
    
    if limit:
        print(f"  ⚡ FAST MODE: Limiting to {limit:,} labels")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            # Check if OpenFDA tables exist
            openfda_tables = ['set_ids', 'mapping_substance_name', 'mapping_generic_name', 
                            'mapping_brand_name', 'labels_meta']
            
            existing_tables = []
            for table in openfda_tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                if cur.fetchone()[0]:
                    existing_tables.append(table)
            
            if 'set_ids' not in existing_tables:
                print("  ⚠️  OpenFDA tables not found - skipping")
                return
            
            print(f"  📋 Found OpenFDA tables: {existing_tables}")
            
            # Check labels_meta columns if it exists
            if 'labels_meta' in existing_tables:
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'labels_meta'
                """)
                labels_meta_cols = {row[0] for row in cur.fetchall()}
                print(f"  📋 labels_meta columns: {labels_meta_cols}")
            else:
                labels_meta_cols = set()
            
            # Clear existing mappings
            print("  🧹 Clearing existing OpenFDA mappings...")
            cur.execute("TRUNCATE map_openfda_molecules RESTART IDENTITY;")
            con.commit()
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            # --- Match via substance_name (highest confidence) ---
            if 'mapping_substance_name' in existing_tables:
                print("  🔍 Matching via substance names...")
                cur.execute(f"""
                    INSERT INTO map_openfda_molecules 
                        (set_id, set_id_fk, product_name, concept_id, mol_id, 
                         match_type, matched_name, confidence)
                    SELECT DISTINCT ON (si.set_id, dms.concept_id)
                        si.set_id,
                        msn.set_id_id,
                        msn.substance_name,  -- Use substance name as product identifier
                        dms.concept_id,
                        dms.mol_id,
                        'SUBSTANCE',
                        msn.substance_name,
                        1.0
                    FROM mapping_substance_name msn
                    JOIN set_ids si ON msn.set_id_id = si.id
                    JOIN dm_molecule_synonyms dms ON LOWER(msn.substance_name) = dms.synonym_lower
                    WHERE dms.concept_id IS NOT NULL
                    ORDER BY si.set_id, dms.concept_id, msn.set_id_id
                    {limit_clause}
                    ON CONFLICT (set_id, concept_id) DO NOTHING
                """)
                substance_matches = cur.rowcount
                con.commit()
                print(f"     ✓ Substance matches: {substance_matches:,}")
            else:
                substance_matches = 0
            
            # --- Match via generic_name ---
            if 'mapping_generic_name' in existing_tables:
                print("  🔍 Matching via generic names...")
                cur.execute(f"""
                    INSERT INTO map_openfda_molecules 
                        (set_id, set_id_fk, product_name, concept_id, mol_id,
                         match_type, matched_name, confidence)
                    SELECT DISTINCT ON (si.set_id, dms.concept_id)
                        si.set_id,
                        mgn.set_id_id,
                        mgn.generic_name,  -- Use generic name as product identifier
                        dms.concept_id,
                        dms.mol_id,
                        'GENERIC_NAME',
                        mgn.generic_name,
                        0.95
                    FROM mapping_generic_name mgn
                    JOIN set_ids si ON mgn.set_id_id = si.id
                    JOIN dm_molecule_synonyms dms ON LOWER(mgn.generic_name) = dms.synonym_lower
                    WHERE dms.concept_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM map_openfda_molecules m 
                          WHERE m.set_id = si.set_id AND m.concept_id = dms.concept_id
                      )
                    ORDER BY si.set_id, dms.concept_id, mgn.set_id_id
                    {limit_clause}
                    ON CONFLICT (set_id, concept_id) DO NOTHING
                """)
                generic_matches = cur.rowcount
                con.commit()
                print(f"     ✓ Generic name matches: {generic_matches:,}")
            else:
                generic_matches = 0
            
            # --- Match via brand_name (lower confidence) ---
            if 'mapping_brand_name' in existing_tables:
                print("  🔍 Matching via brand names...")
                cur.execute(f"""
                    INSERT INTO map_openfda_molecules 
                        (set_id, set_id_fk, product_name, concept_id, mol_id,
                         match_type, matched_name, confidence)
                    SELECT DISTINCT ON (si.set_id, dms.concept_id)
                        si.set_id,
                        mbn.set_id_id,
                        mbn.brand_name,  -- Use brand name as product identifier
                        dms.concept_id,
                        dms.mol_id,
                        'BRAND_NAME',
                        mbn.brand_name,
                        0.85
                    FROM mapping_brand_name mbn
                    JOIN set_ids si ON mbn.set_id_id = si.id
                    JOIN dm_molecule_synonyms dms ON LOWER(mbn.brand_name) = dms.synonym_lower
                    WHERE dms.concept_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM map_openfda_molecules m 
                          WHERE m.set_id = si.set_id AND m.concept_id = dms.concept_id
                      )
                    ORDER BY si.set_id, dms.concept_id, mbn.set_id_id
                    {limit_clause}
                    ON CONFLICT (set_id, concept_id) DO NOTHING
                """)
                brand_matches = cur.rowcount
                con.commit()
                print(f"     ✓ Brand name matches: {brand_matches:,}")
            else:
                brand_matches = 0
            
            # Summary
            cur.execute("""
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT set_id) as labels_mapped,
                    COUNT(DISTINCT concept_id) as concepts_linked,
                    COUNT(*) FILTER (WHERE match_type = 'SUBSTANCE') as substance_ct,
                    COUNT(*) FILTER (WHERE match_type = 'GENERIC_NAME') as generic_ct,
                    COUNT(*) FILTER (WHERE match_type = 'BRAND_NAME') as brand_ct
                FROM map_openfda_molecules
            """)
            stats = cur.fetchone()
            
            log_audit(cur, 'PHASE_11', 'link_openfda', stats[0], 
                     time.time() - start_time, 'SUCCESS')
            con.commit()
            
            print(f"\n  📊 OpenFDA Linkage Summary:")
            print(f"     Total mappings:      {stats[0]:,}")
            print(f"     Labels mapped:       {stats[1]:,}")
            print(f"     Concepts linked:     {stats[2]:,}")
            print(f"     By substance:        {stats[3]:,}")
            print(f"     By generic name:     {stats[4]:,}")
            print(f"     By brand name:       {stats[5]:,}")

# ============================================================================
# PHASE 12: CREATE INDICATION SEARCH FUNCTIONS
# ============================================================================

def create_indication_search_functions(config: DatabaseConfig):
    """Create SQL functions for indication-based queries."""
    print("\n🔍 PHASE 12: Creating Indication Search Functions...")
    
    sql_functions = r"""
    -- ========================================================================
    -- Drop existing functions for idempotency
    -- ========================================================================
    DROP FUNCTION IF EXISTS find_drugs_for_indication(TEXT, BOOLEAN, INT) CASCADE;
    DROP FUNCTION IF EXISTS find_indications_for_drug(TEXT, BOOLEAN) CASCADE;
    DROP FUNCTION IF EXISTS find_drugs_by_therapeutic_area(TEXT, BOOLEAN, INT) CASCADE;
    DROP FUNCTION IF EXISTS get_drug_label_sections(TEXT, TEXT[]) CASCADE;
    DROP FUNCTION IF EXISTS search_label_text(TEXT, INT) CASCADE;

    -- ========================================================================
    -- Find drugs approved/in development for an indication
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_drugs_for_indication(
        indication_query TEXT,
        approved_only BOOLEAN DEFAULT FALSE,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        concept_id BIGINT,
        concept_name TEXT,
        chembl_id TEXT,
        canonical_smiles TEXT,
        indication_name TEXT,
        therapeutic_area TEXT,
        max_phase NUMERIC,
        is_approved BOOLEAN,
        has_fda_label BOOLEAN
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT DISTINCT ON (mc.concept_id, di2.indication_id)
            mc.concept_id,
            mc.preferred_name::TEXT as concept_name,
            dm.chembl_id::TEXT,
            dm.canonical_smiles::TEXT,
            di2.preferred_name::TEXT as indication_name,
            di2.therapeutic_area::TEXT,
            ddi.max_phase,
            ddi.is_approved,
            EXISTS(SELECT 1 FROM map_openfda_molecules mof WHERE mof.concept_id = mc.concept_id)
                OR EXISTS(SELECT 1 FROM map_dailymed_molecules mdm WHERE mdm.concept_id = mc.concept_id)
                as has_fda_label
        FROM dm_indication di2
        JOIN dm_drug_indication ddi ON di2.indication_id = ddi.indication_id
        JOIN dm_molecule_concept mc ON ddi.concept_id = mc.concept_id
        LEFT JOIN dm_molecule dm ON mc.concept_id = dm.concept_id AND dm.is_salt = FALSE
        WHERE (
            di2.preferred_name ILIKE '%' || indication_query || '%'
            OR di2.mesh_heading ILIKE '%' || indication_query || '%'
            OR di2.efo_term ILIKE '%' || indication_query || '%'
        )
        AND (NOT approved_only OR ddi.is_approved)
        ORDER BY mc.concept_id, di2.indication_id, ddi.max_phase DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find all indications for a drug
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_indications_for_drug(
        drug_query TEXT,
        approved_only BOOLEAN DEFAULT FALSE
    )
    RETURNS TABLE (
        indication_name TEXT,
        mesh_id TEXT,
        efo_id TEXT,
        therapeutic_area TEXT,
        max_phase NUMERIC,
        is_approved BOOLEAN
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT DISTINCT
            di.preferred_name::TEXT as indication_name,
            di.mesh_id::TEXT,
            di.efo_id::TEXT,
            di.therapeutic_area::TEXT,
            ddi.max_phase,
            ddi.is_approved
        FROM dm_drug_indication ddi
        JOIN dm_indication di ON ddi.indication_id = di.indication_id
        JOIN dm_molecule_concept mc ON ddi.concept_id = mc.concept_id
        WHERE (
            mc.preferred_name ILIKE '%' || drug_query || '%'
            OR mc.concept_id IN (
                SELECT DISTINCT dms.concept_id 
                FROM dm_molecule_synonyms dms
                WHERE dms.synonym_lower ILIKE '%' || LOWER(drug_query) || '%'
            )
        )
        AND (NOT approved_only OR ddi.is_approved)
        ORDER BY ddi.max_phase DESC NULLS LAST, di.preferred_name;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find drugs by therapeutic area
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_drugs_by_therapeutic_area(
        area TEXT,
        approved_only BOOLEAN DEFAULT TRUE,
        max_results INT DEFAULT 100
    )
    RETURNS TABLE (
        concept_name TEXT,
        chembl_id TEXT,
        indication_names TEXT,
        max_phase NUMERIC,
        n_indications BIGINT
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mc.preferred_name::TEXT as concept_name,
            dm.chembl_id::TEXT,
            STRING_AGG(DISTINCT di.preferred_name, '; ' ORDER BY di.preferred_name)::TEXT as indication_names,
            MAX(ddi.max_phase) as max_phase,
            COUNT(DISTINCT di.indication_id) as n_indications
        FROM dm_drug_indication ddi
        JOIN dm_indication di ON ddi.indication_id = di.indication_id
        JOIN dm_molecule_concept mc ON ddi.concept_id = mc.concept_id
        LEFT JOIN dm_molecule dm ON mc.concept_id = dm.concept_id AND dm.is_salt = FALSE
        WHERE di.therapeutic_area ILIKE '%' || area || '%'
          AND (NOT approved_only OR ddi.is_approved)
        GROUP BY mc.concept_id, mc.preferred_name, dm.chembl_id
        ORDER BY MAX(ddi.max_phase) DESC, COUNT(DISTINCT di.indication_id) DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Get FDA label sections for a drug
    -- ========================================================================
    CREATE OR REPLACE FUNCTION get_drug_label_sections(
        drug_query TEXT,
        section_names TEXT[] DEFAULT NULL
    )
    RETURNS TABLE (
        concept_name TEXT,
        set_id TEXT,
        source TEXT,
        section_name TEXT,
        section_text TEXT
    ) AS $$
    BEGIN
        -- Try OpenFDA first
        RETURN QUERY
        SELECT 
            mc.preferred_name::TEXT as concept_name,
            mof.set_id::TEXT,
            'OPENFDA'::TEXT as source,
            sec.section_name::TEXT,
            sec.text::TEXT as section_text
        FROM dm_molecule_concept mc
        JOIN map_openfda_molecules mof ON mc.concept_id = mof.concept_id
        JOIN set_ids si ON mof.set_id = si.set_id
        JOIN sections sec ON si.id = sec.set_id_id
        WHERE (
            mc.preferred_name ILIKE '%' || drug_query || '%'
            OR mc.concept_id IN (
                SELECT DISTINCT dms.concept_id 
                FROM dm_molecule_synonyms dms
                WHERE dms.synonym_lower ILIKE '%' || LOWER(drug_query) || '%'
            )
        )
        AND (section_names IS NULL OR sec.section_name = ANY(section_names))
        
        UNION ALL
        
        -- DailyMed sections
        SELECT 
            mc.preferred_name::TEXT,
            mdm.set_id::TEXT,
            'DAILYMED'::TEXT,
            ds.section_name::TEXT,
            ds.text::TEXT
        FROM dm_molecule_concept mc
        JOIN map_dailymed_molecules mdm ON mc.concept_id = mdm.concept_id
        JOIN dailymed_sections ds ON mdm.set_id = ds.set_id
        WHERE (
            mc.preferred_name ILIKE '%' || drug_query || '%'
            OR mc.concept_id IN (
                SELECT DISTINCT dms.concept_id 
                FROM dm_molecule_synonyms dms
                WHERE dms.synonym_lower ILIKE '%' || LOWER(drug_query) || '%'
            )
        )
        AND (section_names IS NULL OR ds.section_name = ANY(section_names))
        
        ORDER BY concept_name, source, section_name;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Search label text for keywords
    -- ========================================================================
    CREATE OR REPLACE FUNCTION search_label_text(
        keyword_query TEXT,
        max_results INT DEFAULT 50
    )
    RETURNS TABLE (
        concept_name TEXT,
        set_id TEXT,
        source TEXT,
        section_name TEXT,
        snippet TEXT
    ) AS $$
    DECLARE
        tsquery_text TEXT;
    BEGIN
        -- Convert to tsquery format
        tsquery_text := plainto_tsquery('english', keyword_query)::TEXT;
        
        RETURN QUERY
        -- OpenFDA
        SELECT 
            mc.preferred_name::TEXT as concept_name,
            si.set_id::TEXT,
            'OPENFDA'::TEXT as source,
            sec.section_name::TEXT,
            ts_headline('english', sec.text, plainto_tsquery('english', keyword_query),
                       'MaxWords=50, MinWords=20, StartSel=**, StopSel=**')::TEXT as snippet
        FROM map_openfda_molecules mof
        JOIN dm_molecule_concept mc ON mof.concept_id = mc.concept_id
        JOIN set_ids si ON mof.set_id = si.set_id
        JOIN sections sec ON si.id = sec.set_id_id
        WHERE sec.text_vector @@ plainto_tsquery('english', keyword_query)
        
        UNION ALL
        
        -- DailyMed
        SELECT 
            mc.preferred_name::TEXT,
            ds.set_id::TEXT,
            'DAILYMED'::TEXT,
            ds.section_name::TEXT,
            ts_headline('english', ds.text, plainto_tsquery('english', keyword_query),
                       'MaxWords=50, MinWords=20, StartSel=**, StopSel=**')::TEXT
        FROM map_dailymed_molecules mdm
        JOIN dm_molecule_concept mc ON mdm.concept_id = mc.concept_id
        JOIN dailymed_sections ds ON mdm.set_id = ds.set_id
        WHERE ds.text_vector @@ plainto_tsquery('english', keyword_query)
        
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            try:
                cur.execute(sql_functions)
                con.commit()
                print("  ✅ Indication search functions created")
            except psycopg2.Error as e:
                print(f"  ⚠️  Some functions may have failed (missing tables): {e}")
                con.rollback()




# ============================================================================
# Fix NAMES
# ============================================================================
def fix_null_concept_names(config: DatabaseConfig):
    """Fix NULL concept names using chembl_id or other identifiers."""
    print("\n🏷️  Fixing NULL concept names...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            # Check how many NULLs we have
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM dm_molecule_concept WHERE preferred_name IS NULL) as null_concepts,
                    (SELECT COUNT(*) FROM dm_molecule_target_summary WHERE concept_name IS NULL) as null_summary
            """)
            before = cur.fetchone()
            print(f"  Before: {before[0]:,} NULL concepts, {before[1]:,} NULL summary rows")
            
            # Fix dm_molecule_concept
            cur.execute("""
                UPDATE dm_molecule_concept mc
                SET preferred_name = sub.best_name
                FROM (
                    SELECT DISTINCT ON (dm.concept_id)
                        dm.concept_id,
                        COALESCE(
                            CASE WHEN dm.pref_name IS NOT NULL 
                                  AND dm.pref_name NOT LIKE 'CHEMBL%' 
                                 THEN dm.pref_name END,
                            dm.chembl_id,
                            CASE WHEN dm.drugcentral_id IS NOT NULL 
                                 THEN 'DC_' || dm.drugcentral_id::TEXT END,
                            CASE WHEN dm.bindingdb_monomer_id IS NOT NULL 
                                 THEN 'BDB_' || dm.bindingdb_monomer_id::TEXT END,
                            LEFT(dm.inchi_key, 14)
                        ) as best_name
                    FROM dm_molecule dm
                    WHERE dm.concept_id IS NOT NULL
                    ORDER BY dm.concept_id,
                             CASE WHEN dm.pref_name IS NOT NULL AND dm.pref_name NOT LIKE 'CHEMBL%' THEN 0 ELSE 1 END,
                             CASE WHEN dm.chembl_id IS NOT NULL THEN 0 ELSE 1 END,
                             dm.mol_id
                ) sub
                WHERE mc.concept_id = sub.concept_id
                  AND mc.preferred_name IS NULL
                  AND sub.best_name IS NOT NULL;
            """)
            fixed_concepts = cur.rowcount
            con.commit()
            
            # Fix dm_molecule_target_summary
            cur.execute("""
                UPDATE dm_molecule_target_summary mts
                SET concept_name = mc.preferred_name
                FROM dm_molecule_concept mc
                WHERE mts.concept_id = mc.concept_id
                  AND mts.concept_name IS NULL
                  AND mc.preferred_name IS NOT NULL;
            """)
            fixed_summary = cur.rowcount
            con.commit()
            
            # For any remaining NULLs, use CONCEPT_ID
            cur.execute("""
                UPDATE dm_molecule_concept
                SET preferred_name = 'CONCEPT_' || concept_id::TEXT
                WHERE preferred_name IS NULL;
            """)
            remaining1 = cur.rowcount
            
            cur.execute("""
                UPDATE dm_molecule_target_summary
                SET concept_name = 'CONCEPT_' || concept_id::TEXT
                WHERE concept_name IS NULL;
            """)
            remaining2 = cur.rowcount
            con.commit()
            
            # Verify
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM dm_molecule_concept WHERE preferred_name IS NULL) as null_concepts,
                    (SELECT COUNT(*) FROM dm_molecule_target_summary WHERE concept_name IS NULL) as null_summary
            """)
            after = cur.fetchone()
            
            print(f"  Fixed: {fixed_concepts:,} concepts, {fixed_summary:,} summary rows")
            print(f"  Fallback (CONCEPT_ID): {remaining1:,} concepts, {remaining2:,} summary rows")
            print(f"  After: {after[0]:,} NULL concepts, {after[1]:,} NULL summary rows")
            
    return fixed_concepts + fixed_summary

def fix_placeholder_values(config: DatabaseConfig):
    """Replace 999999 placeholder values with NULL."""
    print("\n🧹 Fixing placeholder values (999999)...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            # Check how many we have
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN best_ic50_nm = 999999 THEN 1 ELSE 0 END) as ic50_999999,
                    SUM(CASE WHEN best_ki_nm = 999999 THEN 1 ELSE 0 END) as ki_999999,
                    SUM(CASE WHEN best_kd_nm = 999999 THEN 1 ELSE 0 END) as kd_999999,
                    SUM(CASE WHEN best_ec50_nm = 999999 THEN 1 ELSE 0 END) as ec50_999999
                FROM dm_molecule_target_summary;
            """)
            counts = cur.fetchone()
            print(f"  Found placeholders: IC50={counts[0]}, Ki={counts[1]}, Kd={counts[2]}, EC50={counts[3]}")
            
            # Replace ALL 999999 values with NULL
            cur.execute("""
                UPDATE dm_molecule_target_summary
                SET 
                    best_ic50_nm = CASE WHEN best_ic50_nm = 999999 THEN NULL ELSE best_ic50_nm END,
                    best_ki_nm = CASE WHEN best_ki_nm = 999999 THEN NULL ELSE best_ki_nm END,
                    best_kd_nm = CASE WHEN best_kd_nm = 999999 THEN NULL ELSE best_kd_nm END,
                    best_ec50_nm = CASE WHEN best_ec50_nm = 999999 THEN NULL ELSE best_ec50_nm END,
                    median_ic50_nm = CASE WHEN median_ic50_nm = 999999 THEN NULL ELSE median_ic50_nm END,
                    median_ki_nm = CASE WHEN median_ki_nm = 999999 THEN NULL ELSE median_ki_nm END,
                    median_kd_nm = CASE WHEN median_kd_nm = 999999 THEN NULL ELSE median_kd_nm END,
                    median_ec50_nm = CASE WHEN median_ec50_nm = 999999 THEN NULL ELSE median_ec50_nm END,
                    ic50_min_nm = CASE WHEN ic50_min_nm = 999999 THEN NULL ELSE ic50_min_nm END,
                    ic50_max_nm = CASE WHEN ic50_max_nm = 999999 THEN NULL ELSE ic50_max_nm END,
                    ki_min_nm = CASE WHEN ki_min_nm = 999999 THEN NULL ELSE ki_min_nm END,
                    ki_max_nm = CASE WHEN ki_max_nm = 999999 THEN NULL ELSE ki_max_nm END,
                    kd_min_nm = CASE WHEN kd_min_nm = 999999 THEN NULL ELSE kd_min_nm END,
                    kd_max_nm = CASE WHEN kd_max_nm = 999999 THEN NULL ELSE kd_max_nm END,
                    ec50_min_nm = CASE WHEN ec50_min_nm = 999999 THEN NULL ELSE ec50_min_nm END,
                    ec50_max_nm = CASE WHEN ec50_max_nm = 999999 THEN NULL ELSE ec50_max_nm END;
            """)
            con.commit()
            
            # Verify
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN best_ic50_nm = 999999 THEN 1 ELSE 0 END) as ic50_999999,
                    SUM(CASE WHEN best_ki_nm = 999999 THEN 1 ELSE 0 END) as ki_999999
                FROM dm_molecule_target_summary;
            """)
            after = cur.fetchone()
            print(f"  After fix: IC50={after[0]}, Ki={after[1]} (should be 0)")
            
    print("  ✅ Done")


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def run_data_quality_checks(config: DatabaseConfig) -> Dict[str, int]:
    """Run comprehensive data quality checks after pipeline completion."""
    print("\n🔬 Running Data Quality Checks...")
    
    checks = {}
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            # Check 1: Molecules with invalid SMILES that didn't parse
            cur.execute("SELECT COUNT(*) FROM dm_molecule WHERE mol IS NULL")
            checks['invalid_mol_objects'] = cur.fetchone()[0]
            
            # Check 2: Molecules missing parent keys
            cur.execute("""
                SELECT COUNT(*) FROM dm_molecule 
                WHERE parent_inchi_key_14 IS NULL 
                  AND canonical_smiles IS NOT NULL
            """)
            checks['missing_parent_keys'] = cur.fetchone()[0]
            
            # Check 3: Concepts with suspiciously many forms
            cur.execute("SELECT COUNT(*) FROM dm_molecule_concept WHERE n_forms > 100")
            checks['concepts_with_many_forms'] = cur.fetchone()[0]
            
            # Check 4: Activity values outside reasonable range
            cur.execute("""
                SELECT COUNT(*) FROM dm_compound_target_activity 
                WHERE activity_value < 0.0001 OR activity_value > 1000000000
            """)
            checks['outlier_activities'] = cur.fetchone()[0]
            
            # Check 5: Duplicate concept names
            cur.execute("""
                SELECT COUNT(*) FROM (
                    SELECT preferred_name, COUNT(*) 
                    FROM dm_molecule_concept 
                    WHERE preferred_name IS NOT NULL
                    GROUP BY preferred_name 
                    HAVING COUNT(*) > 1
                ) x
            """)
            checks['duplicate_concept_names'] = cur.fetchone()[0]
            
            # Check 6: Synonyms quality
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN length(synonym) < 2 THEN 1 ELSE 0 END) as too_short,
                    SUM(CASE WHEN length(synonym) > 200 THEN 1 ELSE 0 END) as too_long
                FROM dm_molecule_synonyms
            """)
            row = cur.fetchone()
            checks['short_synonyms'] = row[0] or 0
            checks['long_synonyms'] = row[1] or 0
            
            # Check 7: Trial mappings without concept
            cur.execute("SELECT COUNT(*) FROM map_ctgov_molecules WHERE concept_id IS NULL")
            checks['trial_mappings_no_concept'] = cur.fetchone()[0]
            
            # Check 8: pChEMBL values outside valid range
            cur.execute("""
                SELECT COUNT(*) FROM dm_compound_target_activity 
                WHERE pchembl_value IS NOT NULL 
                  AND (pchembl_value < 2 OR pchembl_value > 14)
            """)
            checks['invalid_pchembl'] = cur.fetchone()[0]
    
    # Print results
    print("\n  📊 Data Quality Results:")
    all_good = True
    for check_name, count in checks.items():
        if count > 0:
            if count < 100:
                severity = "⚠️ "
            elif count < 10000:
                severity = "🔶"
            else:
                severity = "❌"
            all_good = False
        else:
            severity = "✅"
        print(f"    {severity} {check_name}: {count:,}")
    
    if all_good:
        print("\n  ✅ All quality checks passed!")
    
    return checks


# ============================================================================
# REPORTING
# ============================================================================

# ============================================================================
# REPORTING - COMPREHENSIVE PIPELINE REPORT
# ============================================================================

def generate_pipeline_report(config: DatabaseConfig) -> str:
    """
    Generate a comprehensive report of the pipeline state.
    
    Includes:
      - Molecule statistics (by source, salt forms, stereo)
      - Synonym statistics
      - Activity statistics (raw and deduplicated)
      - Clinical trial mappings
      - Indication data (if available)
      - FDA label linkage (DailyMed, OpenFDA)
      - Recent pipeline runs
    """
    
    report_lines = [
        "",
        "=" * 70,
        "MOLECULAR INTEGRATION PIPELINE REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        ""
    ]
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            
            # ================================================================
            # Check if main tables exist
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_molecule'
                )
            """)
            if not cur.fetchone()[0]:
                return "\n".join(report_lines + [
                    "⚠️  Pipeline has not been run yet.",
                    "   Run: python build_molecular_mappings.py",
                    ""
                ])
            
            # ================================================================
            # MOLECULES
            # ================================================================
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT concept_id) as concepts,
                    SUM(CASE WHEN 'CHEMBL' = ANY(sources) THEN 1 ELSE 0 END) as from_chembl,
                    SUM(CASE WHEN 'DC' = ANY(sources) THEN 1 ELSE 0 END) as from_dc,
                    SUM(CASE WHEN 'BDB' = ANY(sources) THEN 1 ELSE 0 END) as from_bdb,
                    SUM(CASE WHEN array_length(sources, 1) > 1 THEN 1 ELSE 0 END) as multi_source,
                    SUM(CASE WHEN is_salt THEN 1 ELSE 0 END) as salts,
                    SUM(CASE WHEN stereo_type = 'DEFINED' THEN 1 ELSE 0 END) as defined_stereo,
                    SUM(CASE WHEN stereo_type = 'ACHIRAL' THEN 1 ELSE 0 END) as achiral,
                    SUM(CASE WHEN mfp2 IS NOT NULL THEN 1 ELSE 0 END) as with_fingerprints
                FROM dm_molecule
            """)
            mol_stats = cur.fetchone()
            
            report_lines.extend([
                "MOLECULES",
                "-" * 40,
                f"  Total molecules:        {mol_stats[0]:>12,}",
                f"  Unique concepts:        {mol_stats[1]:>12,}",
                f"  From ChEMBL:            {mol_stats[2]:>12,}",
                f"  From DrugCentral:       {mol_stats[3]:>12,}",
                f"  From BindingDB:         {mol_stats[4]:>12,}",
                f"  Multi-source:           {mol_stats[5]:>12,}",
                f"  Salt forms:             {mol_stats[6]:>12,}",
                f"  Defined stereochem:     {mol_stats[7]:>12,}",
                f"  Achiral:                {mol_stats[8]:>12,}",
                f"  With fingerprints:      {mol_stats[9]:>12,}",
                ""
            ])
            
            # ================================================================
            # SYNONYMS
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_molecule_synonyms'
                )
            """)
            has_synonyms = cur.fetchone()[0]
            
            if has_synonyms:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT concept_id) as concepts_covered,
                        COUNT(DISTINCT mol_id) as molecules_covered,
                        COUNT(DISTINCT source) as sources
                    FROM dm_molecule_synonyms
                """)
                syn_stats = cur.fetchone()
                
                # Get source breakdown
                cur.execute("""
                    SELECT source, COUNT(*) as count
                    FROM dm_molecule_synonyms
                    GROUP BY source
                    ORDER BY count DESC
                """)
                syn_sources = cur.fetchall()
                
                report_lines.extend([
                    "SYNONYMS",
                    "-" * 40,
                    f"  Total synonyms:         {syn_stats[0]:>12,}",
                    f"  Concepts covered:       {syn_stats[1]:>12,}",
                    f"  Molecules covered:      {syn_stats[2]:>12,}",
                    f"  Sources:                {syn_stats[3]:>12,}",
                    "",
                    "  By Source:",
                ])
                for source, count in syn_sources:
                    report_lines.append(f"    {source or 'Unknown':<20} {count:>10,}")
                report_lines.append("")
            
            # ================================================================
            # ACTIVITIES (Raw - Materialized View)
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_matviews 
                    WHERE matviewname = 'dm_compound_target_activity'
                )
            """)
            has_activity_view = cur.fetchone()[0]
            
            if has_activity_view:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT concept_id) as concepts,
                        COUNT(DISTINCT gene_symbol) as targets,
                        COUNT(DISTINCT source) as sources,
                        ROUND(AVG(pchembl_value)::NUMERIC, 2) as avg_pchembl,
                        ROUND(MIN(pchembl_value)::NUMERIC, 2) as min_pchembl,
                        ROUND(MAX(pchembl_value)::NUMERIC, 2) as max_pchembl
                    FROM dm_compound_target_activity
                    WHERE pchembl_value IS NOT NULL
                """)
                act_stats = cur.fetchone()
                
                # Activity type breakdown
                cur.execute("""
                    SELECT activity_type, COUNT(*) as count
                    FROM dm_compound_target_activity
                    GROUP BY activity_type
                    ORDER BY count DESC
                    LIMIT 6
                """)
                act_types = cur.fetchall()
                
                report_lines.extend([
                    "ACTIVITIES (Raw Materialized View)",
                    "-" * 40,
                    f"  Total measurements:     {act_stats[0]:>12,}",
                    f"  Concepts with activity: {act_stats[1]:>12,}",
                    f"  Target genes:           {act_stats[2]:>12,}",
                    f"  Data sources:           {act_stats[3]:>12,}",
                    f"  Avg pChEMBL:            {float(act_stats[4] or 0):>12.2f}",
                    f"  pChEMBL range:          {float(act_stats[5] or 0):.2f} - {float(act_stats[6] or 0):.2f}",
                    "",
                    "  By Activity Type:",
                ])
                for act_type, count in act_types:
                    report_lines.append(f"    {act_type or 'Unknown':<20} {count:>10,}")
                report_lines.append("")
            
            # ================================================================
            # ACTIVITIES (Summary - Deduplicated)
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_molecule_target_summary'
                )
            """)
            has_summary = cur.fetchone()[0]
            
            if has_summary:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT concept_id) as unique_concepts,
                        COUNT(DISTINCT gene_symbol) as unique_targets,
                        SUM(CASE WHEN data_confidence = 'HIGH' THEN 1 ELSE 0 END) as high_conf,
                        SUM(CASE WHEN data_confidence = 'MEDIUM' THEN 1 ELSE 0 END) as medium_conf,
                        SUM(CASE WHEN data_confidence = 'LOW' THEN 1 ELSE 0 END) as low_conf,
                        SUM(CASE WHEN n_sources > 1 THEN 1 ELSE 0 END) as multi_source,
                        ROUND(AVG(n_total_measurements)::NUMERIC, 1) as avg_measurements,
                        ROUND(AVG(best_pchembl)::NUMERIC, 2) as avg_best_pchembl
                    FROM dm_molecule_target_summary
                """)
                sum_stats = cur.fetchone()
                
                report_lines.extend([
                    "ACTIVITIES (Summary - Deduplicated)",
                    "-" * 40,
                    f"  Concept-target pairs:   {sum_stats[0]:>12,}",
                    f"  Unique concepts:        {sum_stats[1]:>12,}",
                    f"  Unique targets:         {sum_stats[2]:>12,}",
                    f"  High confidence:        {sum_stats[3]:>12,}",
                    f"  Medium confidence:      {sum_stats[4]:>12,}",
                    f"  Low confidence:         {sum_stats[5]:>12,}",
                    f"  Multi-source:           {sum_stats[6]:>12,}",
                    f"  Avg measurements/pair:  {float(sum_stats[7] or 0):>12.1f}",
                    f"  Avg best pChEMBL:       {float(sum_stats[8] or 0):>12.2f}",
                    ""
                ])
            
            # ================================================================
            # CLINICAL TRIAL MAPPINGS
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'map_ctgov_molecules'
                )
            """)
            has_trials = cur.fetchone()[0]
            
            if has_trials:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_mappings,
                        COUNT(DISTINCT nct_id) as trials_mapped,
                        COUNT(DISTINCT concept_id) as concepts_in_trials,
                        COUNT(*) FILTER (WHERE match_type = 'EXACT') as exact,
                        COUNT(*) FILTER (WHERE match_type = 'SALT_STRIPPED') as salt_stripped,
                        COUNT(*) FILTER (WHERE match_type = 'COMBO_PART') as combo,
                        COUNT(*) FILTER (WHERE match_type = 'FUZZY') as fuzzy
                    FROM map_ctgov_molecules
                """)
                trial_stats = cur.fetchone()
                
                report_lines.extend([
                    "CLINICAL TRIAL MAPPINGS",
                    "-" * 40,
                    f"  Total mappings:         {trial_stats[0]:>12,}",
                    f"  Trials mapped:          {trial_stats[1]:>12,}",
                    f"  Concepts in trials:     {trial_stats[2]:>12,}",
                    "",
                    "  By Match Type:",
                    f"    Exact:                {trial_stats[3]:>12,}",
                    f"    Salt-stripped:        {trial_stats[4]:>12,}",
                    f"    Combo parts:          {trial_stats[5]:>12,}",
                    f"    Fuzzy:                {trial_stats[6]:>12,}",
                    ""
                ])
            
            # ================================================================
            # INDICATIONS
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_indication'
                )
            """)
            has_indications = cur.fetchone()[0]
            
            if has_indications:
                cur.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM dm_indication) as total_indications,
                        (SELECT COUNT(*) FROM dm_drug_indication) as total_mappings,
                        (SELECT COUNT(DISTINCT concept_id) FROM dm_drug_indication) as drugs_with_indications,
                        (SELECT COUNT(*) FROM dm_drug_indication WHERE is_approved) as approved_mappings,
                        (SELECT COUNT(DISTINCT concept_id) FROM dm_drug_indication WHERE is_approved) as approved_drugs
                """)
                ind_stats = cur.fetchone()
                
                # Therapeutic area breakdown
                cur.execute("""
                    SELECT therapeutic_area, COUNT(*) as count
                    FROM dm_indication 
                    WHERE therapeutic_area IS NOT NULL
                    GROUP BY therapeutic_area 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                area_breakdown = cur.fetchall()
                
                report_lines.extend([
                    "INDICATIONS (from ChEMBL)",
                    "-" * 40,
                    f"  Total indications:      {ind_stats[0]:>12,}",
                    f"  Drug-indication links:  {ind_stats[1]:>12,}",
                    f"  Drugs with indications: {ind_stats[2]:>12,}",
                    f"  Approved mappings:      {ind_stats[3]:>12,}",
                    f"  Approved drugs:         {ind_stats[4]:>12,}",
                    "",
                    "  Top Therapeutic Areas:",
                ])
                for area, count in area_breakdown:
                    report_lines.append(f"    {area or 'Unknown':<22} {count:>8,}")
                report_lines.append("")
            
            # ================================================================
            # FDA LABEL LINKAGE - DAILYMED
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'map_dailymed_molecules'
                )
            """)
            has_dailymed = cur.fetchone()[0]
            
            if has_dailymed:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_mappings,
                        COUNT(DISTINCT set_id) as products_mapped,
                        COUNT(DISTINCT concept_id) as concepts_linked,
                        COUNT(*) FILTER (WHERE match_type = 'GENERIC_NAME') as generic_matches,
                        COUNT(*) FILTER (WHERE match_type = 'SUBSTANCE') as substance_matches,
                        COUNT(*) FILTER (WHERE match_type = 'BRAND_NAME') as brand_matches,
                        COUNT(*) FILTER (WHERE match_type = 'PRODUCT_NAME') as product_matches
                    FROM map_dailymed_molecules
                """)
                dm_stats = cur.fetchone()
                
                report_lines.extend([
                    "DAILYMED LABEL LINKAGE",
                    "-" * 40,
                    f"  Total mappings:         {dm_stats[0]:>12,}",
                    f"  Products mapped:        {dm_stats[1]:>12,}",
                    f"  Concepts linked:        {dm_stats[2]:>12,}",
                    "",
                    "  By Match Type:",
                    f"    Generic name:         {dm_stats[3]:>12,}",
                    f"    Substance:            {dm_stats[4]:>12,}",
                    f"    Brand name:           {dm_stats[5]:>12,}",
                    f"    Product name:         {dm_stats[6]:>12,}",
                    ""
                ])
            
            # ================================================================
            # FDA LABEL LINKAGE - OPENFDA
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'map_openfda_molecules'
                )
            """)
            has_openfda = cur.fetchone()[0]
            
            if has_openfda:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_mappings,
                        COUNT(DISTINCT set_id) as labels_mapped,
                        COUNT(DISTINCT concept_id) as concepts_linked,
                        COUNT(*) FILTER (WHERE match_type = 'SUBSTANCE') as substance_matches,
                        COUNT(*) FILTER (WHERE match_type = 'GENERIC_NAME') as generic_matches,
                        COUNT(*) FILTER (WHERE match_type = 'BRAND_NAME') as brand_matches
                    FROM map_openfda_molecules
                """)
                of_stats = cur.fetchone()
                
                report_lines.extend([
                    "OPENFDA LABEL LINKAGE",
                    "-" * 40,
                    f"  Total mappings:         {of_stats[0]:>12,}",
                    f"  Labels mapped:          {of_stats[1]:>12,}",
                    f"  Concepts linked:        {of_stats[2]:>12,}",
                    "",
                    "  By Match Type:",
                    f"    Substance:            {of_stats[3]:>12,}",
                    f"    Generic name:         {of_stats[4]:>12,}",
                    f"    Brand name:           {of_stats[5]:>12,}",
                    ""
                ])
            
            # ================================================================
            # COMBINED COVERAGE SUMMARY
            # ================================================================
            if has_synonyms and (has_trials or has_indications or has_dailymed or has_openfda):
                report_lines.extend([
                    "COMBINED COVERAGE SUMMARY",
                    "-" * 40,
                ])
                
                # Concepts with clinical data
                cur.execute("""
                    SELECT COUNT(DISTINCT concept_id) FROM dm_molecule_concept
                """)
                total_concepts = cur.fetchone()[0]
                
                coverage_items = []
                
                if has_summary:
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM dm_molecule_target_summary")
                    with_activity = cur.fetchone()[0]
                    coverage_items.append(('With target activity', with_activity))
                
                if has_trials:
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM map_ctgov_molecules")
                    with_trials = cur.fetchone()[0]
                    coverage_items.append(('In clinical trials', with_trials))
                
                if has_indications:
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM dm_drug_indication")
                    with_indications = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM dm_drug_indication WHERE is_approved")
                    approved = cur.fetchone()[0]
                    coverage_items.append(('With indications', with_indications))
                    coverage_items.append(('Approved drugs', approved))
                
                if has_dailymed:
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM map_dailymed_molecules")
                    with_dailymed = cur.fetchone()[0]
                    coverage_items.append(('With DailyMed label', with_dailymed))
                
                if has_openfda:
                    cur.execute("SELECT COUNT(DISTINCT concept_id) FROM map_openfda_molecules")
                    with_openfda = cur.fetchone()[0]
                    coverage_items.append(('With OpenFDA label', with_openfda))
                
                # Combined FDA labels
                if has_dailymed and has_openfda:
                    cur.execute("""
                        SELECT COUNT(DISTINCT concept_id) FROM (
                            SELECT concept_id FROM map_dailymed_molecules
                            UNION
                            SELECT concept_id FROM map_openfda_molecules
                        ) combined
                    """)
                    with_any_label = cur.fetchone()[0]
                    coverage_items.append(('With any FDA label', with_any_label))
                
                report_lines.append(f"  Total concepts:         {total_concepts:>12,}")
                report_lines.append("")
                for label, count in coverage_items:
                    pct = (count / total_concepts * 100) if total_concepts > 0 else 0
                    report_lines.append(f"  {label:<22} {count:>10,} ({pct:>5.1f}%)")
                report_lines.append("")
            
            # ================================================================
            # DATA QUALITY INDICATORS
            # ================================================================
            report_lines.extend([
                "DATA QUALITY INDICATORS",
                "-" * 40,
            ])
            
            quality_checks = []
            
            # Check for NULL concept names
            cur.execute("SELECT COUNT(*) FROM dm_molecule_concept WHERE preferred_name IS NULL")
            null_names = cur.fetchone()[0]
            quality_checks.append(('Concepts with NULL names', null_names, null_names == 0))
            
            # Check for orphaned molecules
            cur.execute("SELECT COUNT(*) FROM dm_molecule WHERE concept_id IS NULL")
            orphaned = cur.fetchone()[0]
            quality_checks.append(('Orphaned molecules (no concept)', orphaned, orphaned == 0))
            
            # Check for molecules without fingerprints
            cur.execute("SELECT COUNT(*) FROM dm_molecule WHERE mfp2 IS NULL")
            no_fp = cur.fetchone()[0]
            quality_checks.append(('Molecules without fingerprints', no_fp, no_fp == 0))
            
            if has_summary:
                # Check for placeholder values
                cur.execute("""
                    SELECT COUNT(*) FROM dm_molecule_target_summary 
                    WHERE best_ic50_nm = 999999 OR best_ki_nm = 999999
                """)
                placeholders = cur.fetchone()[0]
                quality_checks.append(('Summary rows with 999999 placeholders', placeholders, placeholders == 0))
            
            for check_name, count, passed in quality_checks:
                icon = "✅" if passed else "⚠️ "
                report_lines.append(f"  {icon} {check_name}: {count:,}")
            report_lines.append("")
            
            # ================================================================
            # RECENT PIPELINE RUNS
            # ================================================================
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_pipeline_audit'
                )
            """)
            has_audit = cur.fetchone()[0]
            
            if has_audit:
                cur.execute("""
                    SELECT phase, operation, rows_affected, 
                           ROUND(duration_seconds::NUMERIC, 1), status,
                           run_timestamp
                    FROM dm_pipeline_audit
                    ORDER BY run_timestamp DESC
                    LIMIT 15
                """)
                audit_entries = cur.fetchall()
                
                if audit_entries:
                    report_lines.extend([
                        "RECENT PIPELINE RUNS",
                        "-" * 40,
                    ])
                    for phase, op, rows, duration, status, ts in audit_entries:
                        status_icon = "✅" if status == 'SUCCESS' else "❌"
                        report_lines.append(
                            f"  {ts.strftime('%Y-%m-%d %H:%M')} | {status_icon} {phase}/{op}: "
                            f"{rows:,} rows in {duration}s"
                        )
                    report_lines.append("")
            
            # ================================================================
            # FOOTER
            # ================================================================
            report_lines.extend([
                "=" * 70,
                "END OF REPORT",
                "=" * 70,
                ""
            ])
    
    return "\n".join(report_lines)


# ============================================================================
# REFRESH FUNCTIONS
# ============================================================================

def refresh_materialized_views(config: DatabaseConfig):
    """Refresh materialized views without full rebuild."""
    print("\n🔄 Refreshing Materialized Views...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            start_time = time.time()
            
            print("  🔄 Refreshing dm_compound_target_activity...")
            try:
                cur.execute("REFRESH MATERIALIZED VIEW dm_compound_target_activity")
                con.commit()
                print("     ✓ Activity view refreshed")
            except psycopg2.Error as e:
                print(f"     ❌ Failed: {e}")
                return
            
            log_audit(cur, 'REFRESH', 'activity_view', 0, time.time() - start_time, 'SUCCESS')
            con.commit()
    
    print("  ✅ Views refreshed")


def update_summary_table(config: DatabaseConfig):
    """Rebuild the summary table from current activity view."""
    print("\n🔄 Rebuilding Summary Table...")
    create_materialized_analytics_view_summary(config)


def ensure_schema_exists(config: DatabaseConfig):
    """Ensure all schema tables exist (create if missing, don't drop existing)."""
    print("\n📋 Ensuring schema exists...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            # Check if main table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'dm_molecule'
                )
            """)
            tables_exist = cur.fetchone()[0]
            
            if not tables_exist:
                print("  📋 Creating schema (tables don't exist)...")
                cur.execute(CREATE_SCHEMA_SQL)
                con.commit()
                print("  ✅ Schema created")
            else:
                print("  ✅ Schema already exists")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(full_rebuild: bool = True, skip_validation: bool = False, 
         enable_fuzzy: bool = True, fuzzy_threshold: float = 0.8,
         from_phase: Optional[int] = None, limit: Optional[int] = None):
    """
    Run the complete molecular integration pipeline.
    
    Args:
        full_rebuild: If True, drops and recreates all tables.
        skip_validation: If True, skip data quality checks.
        enable_fuzzy: If True, enable fuzzy matching for clinical trials.
        fuzzy_threshold: Minimum similarity for fuzzy matches (0-1).
        from_phase: If set, start from this phase (1-6). Earlier phases are skipped.
        limit: Maximum molecules/items to process per source (for fast/debug mode).
    """
    config = DEFAULT_CONFIG
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   BioAgent Molecular Integration Pipeline v2.2                     ║")
    print("║   With RDKit Normalization, Hierarchy & Deduplication              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    if limit:
        print(f"\n⚡ FAST/DEBUG MODE: Limiting to ~{limit:,} items per source")
    
    if from_phase:
        print(f"\n⏩ Starting from Phase {from_phase} (skipping earlier phases)")
    
    total_start = time.time()
    
    # 0. Prerequisites check
    print("\n🔍 Checking prerequisites...")
    
    if not check_rdkit_extension(config):
        print("  ❌ RDKit extension not available or not working!")
        print("     Please install: CREATE EXTENSION rdkit;")
        sys.exit(1)
    print("  ✅ RDKit extension OK")
    
    tables_ok, missing = check_prerequisites(config)
    if not tables_ok:
        print(f"  ❌ Missing required tables:")
        for t in missing:
            print(f"     - {t}")
        print("\n  Please run the prerequisite data loaders first.")
        sys.exit(1)
    print("  ✅ All source tables present")
    
    # Get source counts for progress tracking
    source_counts = get_source_counts(config)
    
    # Determine starting phase
    start_phase = from_phase or 0
    
    # 0. Create Extensions and Audit Table (always run)
    print("\n📋 Phase 0: Setting up extensions and audit table...")
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(CREATE_EXTENSIONS_SQL)
            cur.execute(CREATE_AUDIT_TABLE_SQL)
            con.commit()
    print("  ✅ Extensions ready")
    
    # 0.1 Handle schema based on mode
    if full_rebuild and start_phase == 0:
        # Full rebuild: drop and recreate everything
        fast_drop_schema(config)
        print("  ✅ Old schema dropped")
        
        print("\n📋 Creating schema...")
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(CREATE_SCHEMA_SQL)
                con.commit()
        print("  ✅ Schema created")
    else:
        # Replay mode or incremental: ensure schema exists but don't drop
        ensure_schema_exists(config)
    
    # 0.2 Create RDKit helper functions (always ensure they exist)
    print("\n🧪 Ensuring RDKit helper functions exist...")
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(CREATE_RDKIT_FUNCTIONS_SQL)
            con.commit()
    print("  ✅ RDKit functions ready")
    
    # 1. Populate Molecules with Normalization
    if start_phase <= 1:
        populate_molecules(config, source_counts, limit=limit)
    else:
        print("\n⏩ Skipping Phase 1: populate_molecules")
    
    # 2. Build Hierarchy (Concepts and Stereo Forms)
    if start_phase <= 2:
        build_molecule_hierarchy(config)
    else:
        print("⏩ Skipping Phase 2: build_molecule_hierarchy")
    
    # 3. Populate Synonyms
    if start_phase <= 3:
        # For synonyms, use a higher limit since they're smaller
        syn_limit = limit * 5 if limit else None
        populate_synonyms(config, limit=syn_limit)
    else:
        print("⏩ Skipping Phase 3: populate_synonyms")
    
    # 4. Map Clinical Trials
    if start_phase <= 4:
        map_clinical_trials(config, enable_fuzzy=enable_fuzzy, 
                          fuzzy_threshold=fuzzy_threshold, limit=limit)
    else:
        print("⏩ Skipping Phase 4: map_clinical_trials")
    
    # 5. Create Analytics View (raw activities)
    if start_phase <= 5:
        create_materialized_analytics_view(config, limit=limit)
        # 5B. Create Summary Table (deduplicated)
        create_materialized_analytics_view_summary(config)
    else:
        print("⏩ Skipping Phase 5: create_materialized_analytics_view")
    
    # 6. Create Search Functions
    if start_phase <= 6:
        create_search_functions(config)
    else:
        print("⏩ Skipping Phase 6: create_search_functions")

    if start_phase <= 7:
        fix_null_concept_names(config)
    else:
        print("⏩ Skipping Phase 7: fix_null_concept_names")

    if start_phase <= 8:
        fix_placeholder_values(config)
    else:
        print("⏩ Skipping Phase 8: fix_placeholder_values")
    
    # Phase 9: Populate Indications
    if start_phase <= 9:
        populate_indications(config, limit=limit)
    else:
        print("⏩ Skipping Phase 9: populate_indications")
    
    # Phase 10: Link DailyMed Products
    if start_phase <= 10:
        link_dailymed_products(config, limit=limit)
    else:
        print("⏩ Skipping Phase 10: link_dailymed_products")
    
    # Phase 11: Link OpenFDA Labels
    if start_phase <= 11:
        link_openfda_labels(config, limit=limit)
    else:
        print("⏩ Skipping Phase 11: link_openfda_labels")
    
    # Phase 12: Create Indication Search Functions
    if start_phase <= 12:
        create_indication_search_functions(config)
    else:
        print("⏩ Skipping Phase 12: create_indication_functions")

    # 7. Validate (always run)
    print("\n🔍 Validating pipeline...")
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute("SELECT * FROM validate_molecule_hierarchy()")
            results = cur.fetchall()
            all_passed = True
            for check_name, status, count, details in results:
                icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
                if status == "FAIL":
                    all_passed = False
                print(f"  {icon} {check_name}: {status} ({count:,}) - {details}")
    
    # 8. Data Quality Checks
    if not skip_validation:
        run_data_quality_checks(config)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 PIPELINE COMPLETE SUCCESSFULLY!")
    else:
        print("⚠️  PIPELINE COMPLETE WITH WARNINGS")
    print(f"   Total time: {total_time/60:.1f} minutes")
    if limit:
        print(f"   ⚡ Ran in FAST MODE (limit={limit:,})")
    print("=" * 70)
    
    # Print report
    print(generate_pipeline_report(config))


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Molecular Integration Pipeline - Build unified molecule database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Full rebuild
  %(prog)s --fast                 # Fast debug mode (~10k molecules)
  %(prog)s --limit 50000          # Custom limit (50k molecules per source)
  %(prog)s --incremental          # Incremental update (no schema drop)
  %(prog)s --from-phase 3         # Replay from Phase 3 (synonyms) onwards
  %(prog)s --from-phase 4         # Replay from Phase 4 (trial mapping) onwards
  %(prog)s --from-phase 5         # Rebuild only analytics views
  %(prog)s --refresh-only         # Refresh materialized views only
  %(prog)s --report               # Generate report without running pipeline
  %(prog)s --validate-only        # Run validation checks only

Phases:
  1 = populate_molecules          (load from ChEMBL, DrugCentral, BindingDB)
  2 = build_molecule_hierarchy    (create concepts and stereo forms)
  3 = populate_synonyms           (build synonym dictionary)
  4 = map_clinical_trials         (match CT.gov interventions)
  5 = create_analytics_views      (materialized view + summary table)
  6 = create_search_functions     (SQL helper functions)
  7 = fix_null_concept_names      (fix NULL names)
  8 = fix_placeholder_values      (fix 999999 placeholders)
  9 = populate_indications        (load ChEMBL drug indications)
 10 = link_dailymed_products      (link DailyMed FDA labels)
 11 = link_openfda_labels         (link OpenFDA labels)
 12 = create_indication_functions (indication search SQL functions)

Fast Mode Defaults (--fast):
  - ChEMBL:      10,000 molecules
  - DrugCentral:  2,000 molecules
  - BindingDB:    5,000 molecules
  - Synonyms:    50,000 entries
  - Trials:      10,000 interventions
        """
    )
    
    parser.add_argument('--fast', action='store_true',
                        help='Fast debug mode - limit to ~10k molecules (same as --limit 10000)')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Maximum molecules/items to process per source')
    parser.add_argument('--incremental', action='store_true', 
                        help='Run incremental update instead of full rebuild')
    parser.add_argument('--from-phase', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Start from this phase (1-7), skipping earlier phases')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data quality validation')
    parser.add_argument('--refresh-only', action='store_true',
                        help='Only refresh materialized views')
    parser.add_argument('--report', action='store_true',
                        help='Generate report without running pipeline')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run validation checks only')
    parser.add_argument('--no-fuzzy', action='store_true',
                        help='Disable fuzzy matching for clinical trials')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.8,
                        help='Minimum similarity for fuzzy matches (0-1, default: 0.8)')
    
    args = parser.parse_args()
    
    # Handle --fast flag
    limit = args.limit
    if args.fast and not limit:
        limit = FAST_MODE_LIMITS['chembl']  # 10000 by default
    
    if args.report:
        print(generate_pipeline_report(DEFAULT_CONFIG))
    elif args.validate_only:
        print("\n🔍 Running validation checks...")
        with get_connection(DEFAULT_CONFIG) as con:
            with con.cursor() as cur:
                cur.execute("SELECT * FROM validate_molecule_hierarchy()")
                results = cur.fetchall()
                for check_name, status, count, details in results:
                    icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
                    print(f"  {icon} {check_name}: {status} ({count:,}) - {details}")
        run_data_quality_checks(DEFAULT_CONFIG)
    elif args.refresh_only:
        refresh_materialized_views(DEFAULT_CONFIG)
        update_summary_table(DEFAULT_CONFIG)
    else:
        # If --from-phase is set, don't do full rebuild
        full_rebuild = not args.incremental and args.from_phase is None
        
        main(
            full_rebuild=full_rebuild, 
            skip_validation=args.skip_validation,
            enable_fuzzy=not args.no_fuzzy,
            fuzzy_threshold=args.fuzzy_threshold,
            from_phase=args.from_phase,
            limit=limit
        )