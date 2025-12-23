#!/usr/bin/env python3
"""
Molecular Mapping Pipeline with RDKit-based Normalization.

Purpose:
  1. Consolidate chemical structures from ChEMBL, DrugCentral, and BindingDB into a single `dm_molecule` table.
  2. Use RDKit within PostgreSQL for structure normalization:
     - Salt removal
     - Parent structure generation  
     - Stereochemistry detection
     - Hierarchical molecule concepts
  3. Build a massive synonym dictionary `dm_molecule_synonyms`.
  4. Map "Messy" Text Data (CT.gov Interventions, OpenFDA) -> `dm_molecule`.
  5. Create a unified `dm_activity` view linking Molecules -> Targets.

Prerequisites:
  - PostgreSQL with RDKit extension installed
  - All source tables (chembl_*, bindingdb_*, drugcentral_*, ctgov_*) must exist.
  - dm_target must exist (from your previous scripts).
"""

from __future__ import annotations

import re
import psycopg2
import psycopg2.extras
from tqdm import tqdm
from typing import Set, Dict, List

# Handle imports
try:
    from .config import DatabaseConfig, get_connection, DEFAULT_CONFIG
except ImportError:
    from config import DatabaseConfig, get_connection, DEFAULT_CONFIG


# ============================================================================
# SCHEMA DEFINITION (Updated with Normalization Hierarchy)
# ============================================================================

CREATE_SCHEMA_SQL = """
-- Ensure RDKit extension is available
CREATE EXTENSION IF NOT EXISTS rdkit;

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
    rxnorm_cui TEXT,           -- RxNorm Concept ID
    drugbank_id TEXT,          -- DrugBank ID  
    unii TEXT,                 -- FDA UNII (Unique Ingredient Identifier)
    
    -- Metadata
    has_stereo_variants BOOLEAN DEFAULT FALSE,
    has_salt_forms BOOLEAN DEFAULT FALSE,
    n_forms INTEGER DEFAULT 1,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concept_inchi14 ON dm_molecule_concept(parent_inchi_key_14);
CREATE INDEX IF NOT EXISTS idx_concept_name ON dm_molecule_concept(preferred_name);


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
    stereo_type TEXT,  -- 'ACHIRAL', 'DEFINED', 'RACEMIC', 'PARTIAL', 'UNKNOWN'
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
    parent_inchi_key_14 TEXT,        -- Connectivity only (links to concept)
    parent_stereo_inchi_key TEXT,    -- With stereo (links to stereo form)
    parent_smiles TEXT,              -- Desalted, neutralized SMILES
    
    -- Salt/Form information
    is_salt BOOLEAN DEFAULT FALSE,
    salt_form TEXT,                  -- 'FREE_BASE', 'HYDROCHLORIDE', 'PHOSPHATE', etc.
    n_components INTEGER DEFAULT 1,  -- Number of disconnected fragments
    
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
    mfp2 bfp,      -- Morgan fingerprint (radius 2)
    ffp2 bfp,      -- Feature fingerprint
    
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
CREATE INDEX IF NOT EXISTS idx_dm_mol_chembl ON dm_molecule(chembl_id);
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
    syn_type TEXT,  -- 'BRAND', 'GENERIC', 'RESEARCH_CODE', 'INN', 'USAN'
    source TEXT,
    
    UNIQUE(mol_id, synonym)
);

CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_lower ON dm_molecule_synonyms(synonym_lower);
CREATE INDEX IF NOT EXISTS idx_dm_mol_syn_concept ON dm_molecule_synonyms(concept_id);


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
    match_type TEXT,  -- 'EXACT', 'FUZZY', 'COMBO_PART', 'MANUAL'
    confidence FLOAT,
    
    UNIQUE(intervention_id, mol_id)
);

CREATE INDEX IF NOT EXISTS idx_map_ctgov_mol ON map_ctgov_molecules(mol_id);
CREATE INDEX IF NOT EXISTS idx_map_ctgov_concept ON map_ctgov_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_map_ctgov_nct ON map_ctgov_molecules(nct_id);


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
    
    UNIQUE(set_id, mol_id)
);

CREATE INDEX IF NOT EXISTS idx_map_prod_mol ON map_product_molecules(mol_id);
CREATE INDEX IF NOT EXISTS idx_map_prod_concept ON map_product_molecules(concept_id);
CREATE INDEX IF NOT EXISTS idx_map_prod_set ON map_product_molecules(set_id);
"""


# ============================================================================
# RDKIT HELPER FUNCTIONS IN POSTGRESQL
# ============================================================================

CREATE_RDKIT_FUNCTIONS_SQL = """
-- ============================================================================
-- RDKit-based normalization functions
-- ============================================================================

-- Function: Get the largest fragment (salt stripping)
CREATE OR REPLACE FUNCTION get_largest_fragment(input_mol mol)
RETURNS mol AS $$
DECLARE
    frags text[];
    largest_frag mol;
    current_frag mol;
    max_atoms int := 0;
    current_atoms int;
BEGIN
    IF input_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Split by '.' and find largest fragment
    frags := string_to_array(mol_to_smiles(input_mol), '.');
    
    IF array_length(frags, 1) = 1 THEN
        RETURN input_mol;
    END IF;
    
    FOR i IN 1..array_length(frags, 1) LOOP
        current_frag := mol_from_smiles(frags[i]::cstring);
        IF current_frag IS NOT NULL THEN
            current_atoms := mol_numheavyatoms(current_frag);
            IF current_atoms > max_atoms THEN
                max_atoms := current_atoms;
                largest_frag := current_frag;
            END IF;
        END IF;
    END LOOP;
    
    RETURN largest_frag;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Detect salt type from SMILES
CREATE OR REPLACE FUNCTION detect_salt_type(input_smiles TEXT)
RETURNS TEXT AS $$
BEGIN
    IF input_smiles IS NULL THEN
        RETURN 'UNKNOWN';
    END IF;
    
    -- Check for common salts (order matters - more specific first)
    IF input_smiles ~ '\\.[Cc][Ll]($|[^a-zA-Z])' OR input_smiles ~ '\\[Cl-\\]' THEN
        RETURN 'HYDROCHLORIDE';
    ELSIF input_smiles ~ 'O=P\\(O\\)\\(O\\)O' OR input_smiles ~ 'OP\\(O\\)\\(O\\)=O' OR input_smiles ~ '\\[O-\\]P' THEN
        RETURN 'PHOSPHATE';
    ELSIF input_smiles ~ '\\.[Bb][Rr]($|[^a-zA-Z])' OR input_smiles ~ '\\[Br-\\]' THEN
        RETURN 'HYDROBROMIDE';
    ELSIF input_smiles ~ 'OS\\(=O\\)\\(=O\\)O' OR input_smiles ~ '\\[O-\\]S\\(=O\\)' THEN
        RETURN 'SULFATE';
    ELSIF input_smiles ~ 'CS\\(=O\\)\\(=O\\)O' OR input_smiles ~ 'CS\\(\\[O-\\]\\)' THEN
        RETURN 'MESYLATE';
    ELSIF input_smiles ~ 'CC\\(=O\\)O($|[^a-zA-Z])' OR input_smiles ~ 'CC\\(=O\\)\\[O-\\]' THEN
        RETURN 'ACETATE';
    ELSIF input_smiles ~ 'OC\\(=O\\)C\\(O\\)' OR input_smiles ~ 'tartrat' THEN
        RETURN 'TARTRATE';
    ELSIF input_smiles ~ 'OC\\(=O\\)CC\\(O\\)\\(CC\\(=O\\)O\\)C\\(=O\\)O' THEN
        RETURN 'CITRATE';
    ELSIF input_smiles ~ 'OC\\(=O\\)CC\\(=O\\)O' OR input_smiles ~ 'succinat' THEN
        RETURN 'SUCCINATE';
    ELSIF input_smiles ~ 'c1ccc\\(S\\(=O\\)\\(=O\\)O\\)cc1' THEN
        RETURN 'TOSYLATE';
    ELSIF input_smiles ~ '\\[Na\\+\\]' OR input_smiles ~ '\\.Na($|[^a-zA-Z])' THEN
        RETURN 'SODIUM';
    ELSIF input_smiles ~ '\\[K\\+\\]' OR input_smiles ~ '\\.K($|[^a-zA-Z])' THEN
        RETURN 'POTASSIUM';
    ELSIF input_smiles ~ '\\[Ca\\+2\\]' OR input_smiles ~ '\\.Ca($|[^a-zA-Z])' THEN
        RETURN 'CALCIUM';
    ELSIF input_smiles LIKE '%.%' THEN
        RETURN 'SALT_OTHER';
    ELSE
        RETURN 'FREE_BASE';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Classify stereochemistry type
CREATE OR REPLACE FUNCTION classify_stereo(input_mol mol)
RETURNS TEXT AS $$
DECLARE
    smiles_str TEXT;
    n_at_signs INT;
    n_double_bond_stereo INT;
BEGIN
    IF input_mol IS NULL THEN
        RETURN 'UNKNOWN';
    END IF;
    
    smiles_str := mol_to_smiles(input_mol);
    
    -- Count chiral centers (@ symbols not in @@)
    -- Simple heuristic: count @ characters
    n_at_signs := length(smiles_str) - length(replace(smiles_str, '@', ''));
    
    -- Check for double bond stereo (/ and \)
    n_double_bond_stereo := length(smiles_str) - length(replace(replace(smiles_str, '/', ''), '\', ''));
    
    IF n_at_signs = 0 AND n_double_bond_stereo = 0 THEN
        RETURN 'ACHIRAL';
    ELSIF n_at_signs > 0 THEN
        -- Check if stereo is defined (has @@ or just @)
        IF smiles_str LIKE '%@@%' OR smiles_str ~ '@[^@]' THEN
            RETURN 'DEFINED';
        ELSE
            RETURN 'RACEMIC';
        END IF;
    ELSE
        RETURN 'PARTIAL';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Count chiral centers
CREATE OR REPLACE FUNCTION count_chiral_centers(input_mol mol)
RETURNS INTEGER AS $$
DECLARE
    smiles_str TEXT;
    n_centers INT;
BEGIN
    IF input_mol IS NULL THEN
        RETURN 0;
    END IF;
    
    smiles_str := mol_to_smiles(input_mol);
    -- Count @ symbols (each @ or @@ represents a chiral center)
    n_centers := (length(smiles_str) - length(replace(smiles_str, '@', '')));
    -- Divide by 2 for @@ (two @ symbols per center when using @@)
    -- This is approximate
    RETURN GREATEST(1, n_centers / 2) * (CASE WHEN n_centers > 0 THEN 1 ELSE 0 END);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Count components (disconnected fragments)
CREATE OR REPLACE FUNCTION count_components(input_smiles TEXT)
RETURNS INTEGER AS $$
BEGIN
    IF input_smiles IS NULL OR input_smiles = '' THEN
        RETURN 0;
    END IF;
    RETURN array_length(string_to_array(input_smiles, '.'), 1);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Generate parent InChI key (14 char connectivity)
-- This strips stereochemistry from the InChI key
CREATE OR REPLACE FUNCTION get_parent_inchi_key_14(input_mol mol)
RETURNS TEXT AS $$
DECLARE
    parent_mol mol;
    full_key TEXT;
BEGIN
    IF input_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Get largest fragment (strip salts)
    parent_mol := get_largest_fragment(input_mol);
    
    IF parent_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Get InChI key and take first 14 chars (connectivity layer)
    full_key := mol_inchikey(parent_mol);
    
    IF full_key IS NULL OR length(full_key) < 14 THEN
        RETURN NULL;
    END IF;
    
    RETURN left(full_key, 14);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Generate parent InChI key (full, with stereo, desalted)
CREATE OR REPLACE FUNCTION get_parent_stereo_inchi_key(input_mol mol)
RETURNS TEXT AS $$
DECLARE
    parent_mol mol;
BEGIN
    IF input_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Get largest fragment (strip salts)
    parent_mol := get_largest_fragment(input_mol);
    
    IF parent_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN mol_inchikey(parent_mol);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Function: Get parent SMILES (largest fragment)
CREATE OR REPLACE FUNCTION get_parent_smiles(input_mol mol)
RETURNS TEXT AS $$
DECLARE
    parent_mol mol;
BEGIN
    IF input_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    parent_mol := get_largest_fragment(input_mol);
    
    IF parent_mol IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN mol_to_smiles(parent_mol);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Comprehensive normalization function that returns all derived properties
CREATE OR REPLACE FUNCTION normalize_molecule(input_smiles TEXT)
RETURNS TABLE (
    mol_obj mol,
    parent_mol mol,
    parent_smiles TEXT,
    parent_inchi_key_14 TEXT,
    parent_stereo_inchi_key TEXT,
    is_salt BOOLEAN,
    salt_form TEXT,
    n_components INTEGER,
    stereo_type TEXT,
    n_chiral_centers INTEGER,
    mfp2_fp bfp,
    ffp2_fp bfp
) AS $$
DECLARE
    input_mol mol;
    parent mol;
    p_smiles TEXT;
    p_inchi14 TEXT;
    p_stereo_inchi TEXT;
    salt_detected BOOLEAN;
    salt_type TEXT;
    n_comp INTEGER;
    stereo TEXT;
    n_chiral INTEGER;
BEGIN
    -- Parse input SMILES
    input_mol := mol_from_smiles(input_smiles::cstring);
    
    IF input_mol IS NULL THEN
        RETURN;
    END IF;
    
    -- Count components
    n_comp := count_components(input_smiles);
    salt_detected := (n_comp > 1);
    
    -- Get parent (largest fragment)
    parent := get_largest_fragment(input_mol);
    
    IF parent IS NULL THEN
        parent := input_mol;
    END IF;
    
    -- Generate derived properties
    p_smiles := mol_to_smiles(parent);
    p_inchi14 := get_parent_inchi_key_14(input_mol);
    p_stereo_inchi := get_parent_stereo_inchi_key(input_mol);
    salt_type := detect_salt_type(input_smiles);
    stereo := classify_stereo(parent);
    n_chiral := count_chiral_centers(parent);
    
    RETURN QUERY SELECT 
        input_mol,
        parent,
        p_smiles,
        p_inchi14,
        p_stereo_inchi,
        salt_detected,
        salt_type,
        n_comp,
        stereo,
        n_chiral,
        morganbv_fp(parent),
        featmorganbv_fp(parent);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


# ============================================================================
# PHASE 1: POPULATE MOLECULES WITH NORMALIZATION
# ============================================================================

def populate_molecules(config: DatabaseConfig):
    """Load molecules from all sources with RDKit normalization."""
    print("\n🧪 PHASE 1: Consolidating Molecules with RDKit Normalization...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            
            # --- 1.1 Load ChEMBL (High Quality Structure) ---
            print("  📥 Loading ChEMBL molecules with normalization...")
            cur.execute("""
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
                    -- RDKit normalization
                    mol_from_smiles(s.canonical_smiles::cstring),
                    get_parent_smiles(mol_from_smiles(s.canonical_smiles::cstring)),
                    get_parent_inchi_key_14(mol_from_smiles(s.canonical_smiles::cstring)),
                    get_parent_stereo_inchi_key(mol_from_smiles(s.canonical_smiles::cstring)),
                    count_components(s.canonical_smiles) > 1,
                    detect_salt_type(s.canonical_smiles),
                    count_components(s.canonical_smiles),
                    classify_stereo(get_largest_fragment(mol_from_smiles(s.canonical_smiles::cstring))),
                    count_chiral_centers(get_largest_fragment(mol_from_smiles(s.canonical_smiles::cstring))),
                    morganbv_fp(get_largest_fragment(mol_from_smiles(s.canonical_smiles::cstring))),
                    featmorganbv_fp(get_largest_fragment(mol_from_smiles(s.canonical_smiles::cstring)))
                FROM compound_structures s
                JOIN molecule_dictionary md ON s.molregno = md.molregno
                WHERE s.standard_inchi_key IS NOT NULL
                  AND s.canonical_smiles IS NOT NULL
                ON CONFLICT (inchi_key) DO UPDATE SET
                    chembl_id = EXCLUDED.chembl_id,
                    sources = CASE 
                        WHEN 'CHEMBL' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'CHEMBL')
                    END;
            """)
            chembl_count = cur.rowcount
            print(f"     ChEMBL molecules loaded: {chembl_count:,}")
            con.commit()

            # --- 1.2 Load DrugCentral ---
            print("  📥 Merging DrugCentral molecules...")
            cur.execute("""
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
                    get_parent_smiles(mol_from_smiles(smiles::cstring)),
                    get_parent_inchi_key_14(mol_from_smiles(smiles::cstring)),
                    get_parent_stereo_inchi_key(mol_from_smiles(smiles::cstring)),
                    count_components(smiles) > 1,
                    detect_salt_type(smiles),
                    count_components(smiles),
                    classify_stereo(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    count_chiral_centers(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    morganbv_fp(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    featmorganbv_fp(get_largest_fragment(mol_from_smiles(smiles::cstring)))
                FROM drugcentral_drugs
                WHERE inchi_key IS NOT NULL
                  AND smiles IS NOT NULL
                ORDER BY inchi_key, id DESC
                ON CONFLICT (inchi_key) DO UPDATE SET
                    drugcentral_id = EXCLUDED.drugcentral_id,
                    pref_name = COALESCE(dm_molecule.pref_name, EXCLUDED.pref_name),
                    sources = CASE 
                        WHEN 'DC' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'DC')
                    END;
            """)
            dc_count = cur.rowcount
            print(f"     DrugCentral molecules merged: {dc_count:,}")
            con.commit()

            # --- 1.3 Load BindingDB ---
            print("  📥 Merging BindingDB molecules...")
            cur.execute("""
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
                    get_parent_smiles(mol_from_smiles(smiles::cstring)),
                    get_parent_inchi_key_14(mol_from_smiles(smiles::cstring)),
                    get_parent_stereo_inchi_key(mol_from_smiles(smiles::cstring)),
                    count_components(smiles) > 1,
                    detect_salt_type(smiles),
                    count_components(smiles),
                    classify_stereo(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    count_chiral_centers(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    morganbv_fp(get_largest_fragment(mol_from_smiles(smiles::cstring))),
                    featmorganbv_fp(get_largest_fragment(mol_from_smiles(smiles::cstring)))
                FROM bindingdb_molecules
                WHERE inchi_key IS NOT NULL
                  AND smiles IS NOT NULL
                ORDER BY inchi_key, id DESC
                ON CONFLICT (inchi_key) DO UPDATE SET
                    bindingdb_monomer_id = EXCLUDED.bindingdb_monomer_id,
                    pubchem_cid = COALESCE(dm_molecule.pubchem_cid, EXCLUDED.pubchem_cid),
                    sources = CASE 
                        WHEN 'BDB' = ANY(dm_molecule.sources) THEN dm_molecule.sources
                        ELSE array_append(dm_molecule.sources, 'BDB')
                    END;
            """)
            bdb_count = cur.rowcount
            print(f"     BindingDB molecules merged: {bdb_count:,}")
            con.commit()
            
            # Get total count
            cur.execute("SELECT COUNT(*) FROM dm_molecule")
            total = cur.fetchone()[0]
            print(f"\n  📊 Total molecules in dm_molecule: {total:,}")


# ============================================================================
# PHASE 2: BUILD MOLECULE HIERARCHY
# ============================================================================

def build_molecule_hierarchy(config: DatabaseConfig):
    """Build the concept and stereo hierarchy from normalized molecules."""
    print("\n🏗️  PHASE 2: Building Molecule Hierarchy...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            
            # --- 2.1 Create Drug Concepts (Level 1) ---
            print("  📊 Creating drug concepts (Level 1 - connectivity grouping)...")
            cur.execute("""
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
                    MODE() WITHIN GROUP (ORDER BY pref_name) FILTER (WHERE pref_name IS NOT NULL) as preferred_name,
                    COUNT(DISTINCT parent_stereo_inchi_key) > 1 as has_stereo_variants,
                    bool_or(is_salt) as has_salt_forms,
                    COUNT(*) as n_forms
                FROM dm_molecule
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
            print(f"     Drug concepts created: {concept_count:,}")
            con.commit()
            
            # --- 2.2 Create Stereo Forms (Level 2) ---
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
                        WHEN dm.stereo_type = 'PARTIAL' THEN dm.n_chiral_centers / 2
                        ELSE 0
                    END as n_defined_centers,
                    -- Mark as API if it's a defined stereoisomer and not a salt
                    (dm.stereo_type = 'DEFINED' AND NOT dm.is_salt) as is_api
                FROM dm_molecule dm
                JOIN dm_molecule_concept mc ON dm.parent_inchi_key_14 = mc.parent_inchi_key_14
                WHERE dm.parent_stereo_inchi_key IS NOT NULL
                ORDER BY dm.parent_stereo_inchi_key, dm.is_salt ASC, dm.mol_id
                ON CONFLICT (parent_stereo_inchi_key) DO NOTHING;
            """)
            stereo_count = cur.rowcount
            print(f"     Stereo forms created: {stereo_count:,}")
            con.commit()
            
            # --- 2.3 Link Molecules to Hierarchy ---
            print("  🔗 Linking molecules to hierarchy...")
            cur.execute("""
                UPDATE dm_molecule dm
                SET 
                    concept_id = mc.concept_id,
                    stereo_id = ms.stereo_id
                FROM dm_molecule_concept mc
                LEFT JOIN dm_molecule_stereo ms 
                    ON mc.concept_id = ms.concept_id 
                    AND dm.parent_stereo_inchi_key = ms.parent_stereo_inchi_key
                WHERE dm.parent_inchi_key_14 = mc.parent_inchi_key_14
                  AND (dm.concept_id IS NULL OR dm.stereo_id IS NULL);
            """)
            linked_count = cur.rowcount
            print(f"     Molecules linked: {linked_count:,}")
            con.commit()
            
            # Print summary statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_molecules,
                    COUNT(DISTINCT concept_id) as unique_concepts,
                    COUNT(DISTINCT stereo_id) as unique_stereo_forms,
                    SUM(CASE WHEN is_salt THEN 1 ELSE 0 END) as salt_forms,
                    SUM(CASE WHEN stereo_type = 'DEFINED' THEN 1 ELSE 0 END) as defined_stereo,
                    SUM(CASE WHEN stereo_type = 'RACEMIC' THEN 1 ELSE 0 END) as racemic,
                    SUM(CASE WHEN stereo_type = 'ACHIRAL' THEN 1 ELSE 0 END) as achiral
                FROM dm_molecule
            """)
            stats = cur.fetchone()
            print(f"\n  📊 Hierarchy Statistics:")
            print(f"     Total molecules:     {stats[0]:,}")
            print(f"     Unique concepts:     {stats[1]:,}")
            print(f"     Unique stereo forms: {stats[2]:,}")
            print(f"     Salt forms:          {stats[3]:,}")
            print(f"     Defined stereo:      {stats[4]:,}")
            print(f"     Racemic:             {stats[5]:,}")
            print(f"     Achiral:             {stats[6]:,}")


# ============================================================================
# PHASE 3: POPULATE SYNONYMS
# ============================================================================

def populate_synonyms(config: DatabaseConfig):
    """Build synonym dictionary linked to both molecules and concepts."""
    print("\n📚 PHASE 3: Building Synonym Dictionary...")
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            
            # --- 3.1 ChEMBL Synonyms ---
            print("  📥 Indexing ChEMBL synonyms...")
            cur.execute("""
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
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            print(f"     ChEMBL synonyms added: {cur.rowcount:,}")
            
            # Add pref_names as synonyms
            cur.execute("""
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
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            print(f"     Pref names added: {cur.rowcount:,}")

            # --- 3.2 DrugCentral Synonyms ---
            print("  📥 Indexing DrugCentral synonyms...")
            cur.execute("""
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
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            print(f"     DrugCentral synonyms added: {cur.rowcount:,}")
            
            # --- 3.3 BindingDB names ---
            print("  📥 Indexing BindingDB names...")
            cur.execute("""
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
                  AND bm.ligand_name NOT LIKE '%::%'  -- Skip compound names with ::
                ON CONFLICT (mol_id, synonym) DO NOTHING;
            """)
            print(f"     BindingDB names added: {cur.rowcount:,}")

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
    name = re.sub(r'\s\d+(\.\d+)?\s?(mg|g|ml|mcg|ug|%)\b', '', name) 
    name = re.sub(r'\s(injection|tablets?|capsules?|solution|cream|gel|ointment|patch|spray)s?$', '', name)
    name = re.sub(r'\s(extended.release|immediate.release|sustained.release)$', '', name)
    # Remove parenthetical content
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    return name.strip()

def map_clinical_trials(config: DatabaseConfig):
    """Map CT.gov interventions to molecules AND concepts."""
    print("\n🏥 PHASE 4: Mapping CT.gov Interventions...")
    
    with get_connection(config) as con:
        # Load synonyms into memory
        print("  🧠 Loading synonyms into memory...")
        syn_map = {}  # cleaned_name -> (mol_id, concept_id)
        
        with con.cursor(name='syn_cursor') as cur:
            cur.itersize = 10000
            cur.execute("""
                SELECT mol_id, concept_id, synonym_lower 
                FROM dm_molecule_synonyms
                WHERE length(synonym_lower) >= 3
            """)
            for mol_id, concept_id, syn in tqdm(cur, desc="Loading Synonyms"):
                if syn and len(syn) >= 3:
                    syn_map[syn] = (mol_id, concept_id)
        
        print(f"     Loaded {len(syn_map):,} unique synonyms.")

        # Fetch and match interventions
        print("  🔎 Matching Interventions...")
        matches_found = 0
        batch_inserts = []
        
        with con.cursor() as cur:
            cur.execute("""
                SELECT id, nct_id, name 
                FROM ctgov_interventions 
                WHERE intervention_type IN ('DRUG', 'BIOLOGICAL', 'DIETARY SUPPLEMENT')
            """)
            
            interventions = cur.fetchall()
            
            for int_id, nct_id, raw_name in tqdm(interventions, desc="Matching"):
                if not raw_name: 
                    continue
                
                cleaned = clean_name(raw_name)
                
                # 1. Exact Match
                if cleaned in syn_map:
                    mol_id, concept_id = syn_map[cleaned]
                    batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'EXACT', 1.0))
                    matches_found += 1
                    continue
                
                # 2. Try without common suffixes
                for suffix in [' hcl', ' hydrochloride', ' sodium', ' potassium', 
                              ' phosphate', ' sulfate', ' acetate', ' maleate',
                              ' tartrate', ' citrate', ' mesylate', ' besylate']:
                    if cleaned.endswith(suffix):
                        base_name = cleaned[:-len(suffix)].strip()
                        if base_name in syn_map:
                            mol_id, concept_id = syn_map[base_name]
                            batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'SALT_STRIPPED', 0.95))
                            matches_found += 1
                            break
                else:
                    # 3. Split combo drugs
                    if " and " in cleaned or " + " in cleaned or "/" in cleaned:
                        parts = re.split(r' and | \+ |/', cleaned)
                        for part in parts:
                            p_clean = part.strip()
                            if p_clean in syn_map:
                                mol_id, concept_id = syn_map[p_clean]
                                batch_inserts.append((nct_id, int_id, raw_name, mol_id, concept_id, 'COMBO_PART', 0.9))
            
            # Bulk Insert
            if batch_inserts:
                print(f"  💾 Inserting {len(batch_inserts):,} matches...")
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO map_ctgov_molecules 
                       (nct_id, intervention_id, match_name, mol_id, concept_id, match_type, confidence) 
                       VALUES %s ON CONFLICT (intervention_id, mol_id) DO NOTHING""",
                    batch_inserts,
                    page_size=1000
                )
                con.commit()
    
    print(f"  ✅ CT.gov Mapping Complete. Found {matches_found:,} matches.")


# ============================================================================
# PHASE 5: UNIFIED ANALYTICS VIEW
# ============================================================================

def create_materialized_analytics_view(config: DatabaseConfig):
    """Create materialized view for compound-target activities."""
    print("\n🔗 PHASE 5: Creating Materialized Analytics View...")
    
    sql_cleanup = """
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_views WHERE viewname = 'dm_compound_target_activity') THEN
            DROP VIEW dm_compound_target_activity CASCADE;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'dm_compound_target_activity') THEN
            DROP MATERIALIZED VIEW dm_compound_target_activity CASCADE;
        END IF;
    END $$;
    """
    
    sql_view = """
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
    WHERE act.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
      AND act.standard_value IS NOT NULL
      AND t.gene_symbol IS NOT NULL
    
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
    WHERE ba.kd_nm IS NOT NULL;
    """
    
    sql_indexes = """
    CREATE INDEX idx_cta_gene_symbol ON dm_compound_target_activity(gene_symbol);
    CREATE INDEX idx_cta_mol_id ON dm_compound_target_activity(mol_id);
    CREATE INDEX idx_cta_concept_id ON dm_compound_target_activity(concept_id);
    CREATE INDEX idx_cta_potency ON dm_compound_target_activity(pchembl_value DESC NULLS LAST);
    CREATE INDEX idx_cta_activity_type ON dm_compound_target_activity(activity_type);
    CREATE INDEX idx_cta_gene_potency ON dm_compound_target_activity(gene_symbol, pchembl_value DESC);
    CREATE INDEX idx_cta_source ON dm_compound_target_activity(source);
    """

    with get_connection(config) as con:
        with con.cursor() as cur:
            print("  🧹 Cleaning up existing views...")
            cur.execute(sql_cleanup)
            
            print("  🏗️  Materializing view (this may take a few minutes)...")
            cur.execute(sql_view)
            
            print("  ⚡ Building indexes...")
            cur.execute(sql_indexes)
            
            con.commit()
    
    print("  ✅ Materialized view created successfully.")


# ============================================================================
# PHASE 6: CREATE SEARCH AND QUERY FUNCTIONS
# ============================================================================

def create_search_functions(config: DatabaseConfig):
    """Create SQL functions for similarity search and concept-based queries."""
    print("\n🔍 PHASE 6: Creating Search Functions...")
    
    sql_functions = """
    -- ========================================================================
    -- Find similar molecules by SMILES
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_similar_molecules(
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
    BEGIN
        query_fp := morganbv_fp(mol_from_smiles(query_smiles::cstring));
        EXECUTE format('SET rdkit.tanimoto_threshold = %s', similarity_threshold);
        
        RETURN QUERY
        SELECT 
            dm.mol_id,
            dm.concept_id,
            dm.pref_name,
            mc.preferred_name as concept_name,
            dm.chembl_id,
            dm.canonical_smiles,
            tanimoto_sml(query_fp, dm.mfp2)::FLOAT as tanimoto_similarity,
            dm.is_salt,
            dm.salt_form,
            dm.stereo_type
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        WHERE dm.mfp2 %% query_fp
        ORDER BY tanimoto_sml(query_fp, dm.mfp2) DESC
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find all forms of a drug concept
    -- ========================================================================
    CREATE OR REPLACE FUNCTION get_drug_forms(
        drug_name TEXT
    )
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
            mc.preferred_name,
            dm.mol_id,
            dm.pref_name,
            dm.inchi_key,
            dm.is_salt,
            dm.salt_form,
            dm.stereo_type,
            dm.chembl_id,
            dm.drugcentral_id
        FROM dm_molecule_concept mc
        JOIN dm_molecule dm ON mc.concept_id = dm.concept_id
        WHERE mc.preferred_name ILIKE '%' || drug_name || '%'
           OR mc.concept_id IN (
               SELECT DISTINCT concept_id 
               FROM dm_molecule_synonyms 
               WHERE synonym_lower ILIKE '%' || lower(drug_name) || '%'
           )
        ORDER BY mc.preferred_name, dm.is_salt, dm.stereo_type;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find trials for a drug concept (all forms)
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_trials_for_drug_concept(
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
            mc.preferred_name as concept_name,
            dm.mol_id,
            dm.pref_name as molecule_form,
            dm.salt_form,
            s.nct_id,
            s.brief_title as trial_title,
            s.overall_status as trial_status,
            s.phase
        FROM dm_molecule_concept mc
        JOIN dm_molecule dm ON mc.concept_id = dm.concept_id
        JOIN map_ctgov_molecules map ON dm.mol_id = map.mol_id
        JOIN ctgov_studies s ON map.nct_id = s.nct_id
        WHERE mc.preferred_name ILIKE '%' || drug_name || '%'
           OR mc.concept_id IN (
               SELECT DISTINCT concept_id 
               FROM dm_molecule_synonyms 
               WHERE synonym_lower ILIKE '%' || lower(drug_name) || '%'
           )
        ORDER BY s.start_date DESC NULLS LAST
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find targets for a drug concept (aggregated across all forms)
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_targets_for_drug_concept(
        drug_name TEXT,
        min_pchembl FLOAT DEFAULT 5.0
    )
    RETURNS TABLE (
        concept_name TEXT,
        gene_symbol TEXT,
        activity_type TEXT,
        n_measurements BIGINT,
        best_pchembl NUMERIC,
        median_activity_nm NUMERIC,
        forms_tested TEXT[]
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            mc.preferred_name as concept_name,
            cta.gene_symbol,
            cta.activity_type,
            COUNT(*) as n_measurements,
            MAX(cta.pchembl_value) as best_pchembl,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cta.activity_value) as median_activity_nm,
            ARRAY_AGG(DISTINCT dm.salt_form) as forms_tested
        FROM dm_molecule_concept mc
        JOIN dm_molecule dm ON mc.concept_id = dm.concept_id
        JOIN dm_compound_target_activity cta ON dm.mol_id = cta.mol_id
        WHERE (mc.preferred_name ILIKE '%' || drug_name || '%'
               OR mc.concept_id IN (
                   SELECT DISTINCT concept_id 
                   FROM dm_molecule_synonyms 
                   WHERE synonym_lower ILIKE '%' || lower(drug_name) || '%'
               ))
          AND (cta.pchembl_value >= min_pchembl OR cta.pchembl_value IS NULL)
        GROUP BY mc.preferred_name, cta.gene_symbol, cta.activity_type
        ORDER BY best_pchembl DESC NULLS LAST;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Find trials by structure similarity
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_trials_by_similar_molecules(
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
    BEGIN
        query_fp := morganbv_fp(mol_from_smiles(query_smiles::cstring));
        EXECUTE format('SET rdkit.tanimoto_threshold = %s', similarity_threshold);
        
        RETURN QUERY
        SELECT DISTINCT
            dm.mol_id,
            dm.pref_name as molecule_name,
            mc.preferred_name as concept_name,
            tanimoto_sml(query_fp, dm.mfp2)::FLOAT as tanimoto_similarity,
            s.nct_id,
            s.brief_title as trial_title,
            s.overall_status as trial_status,
            s.phase
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        JOIN map_ctgov_molecules map ON dm.mol_id = map.mol_id
        JOIN ctgov_studies s ON map.nct_id = s.nct_id
        WHERE dm.mfp2 %% query_fp
        ORDER BY tanimoto_sml(query_fp, dm.mfp2) DESC, s.start_date DESC NULLS LAST
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Substructure search
    -- ========================================================================
    CREATE OR REPLACE FUNCTION find_molecules_by_substructure(
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
    BEGIN
        RETURN QUERY
        SELECT 
            dm.mol_id,
            dm.concept_id,
            dm.pref_name,
            mc.preferred_name as concept_name,
            dm.chembl_id,
            dm.canonical_smiles,
            dm.is_salt,
            dm.salt_form
        FROM dm_molecule dm
        LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
        WHERE dm.mol @> qmol_from_smarts(query_smarts::cstring)
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql;


    -- ========================================================================
    -- Get concept summary (for a drug like ruxolitinib)
    -- ========================================================================
    CREATE OR REPLACE FUNCTION get_concept_summary(
        drug_name TEXT
    )
    RETURNS TABLE (
        concept_id BIGINT,
        concept_name TEXT,
        parent_inchi_key_14 TEXT,
        n_forms INTEGER,
        n_stereo_variants INTEGER,
        n_salt_forms INTEGER,
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
            mc.preferred_name,
            mc.parent_inchi_key_14,
            mc.n_forms,
            (SELECT COUNT(DISTINCT stereo_id) FROM dm_molecule WHERE concept_id = mc.concept_id)::INTEGER,
            (SELECT COUNT(*) FROM dm_molecule WHERE concept_id = mc.concept_id AND is_salt = TRUE)::INTEGER,
            (SELECT COUNT(DISTINCT nct_id) FROM map_ctgov_molecules WHERE concept_id = mc.concept_id),
            (SELECT COUNT(DISTINCT gene_symbol) FROM dm_compound_target_activity WHERE concept_id = mc.concept_id),
            (SELECT jsonb_agg(jsonb_build_object(
                'mol_id', dm.mol_id,
                'name', dm.pref_name,
                'inchi_key', dm.inchi_key,
                'salt_form', dm.salt_form,
                'stereo_type', dm.stereo_type,
                'chembl_id', dm.chembl_id
            )) FROM dm_molecule dm WHERE dm.concept_id = mc.concept_id)
        FROM dm_molecule_concept mc
        JOIN concept_match cm ON mc.concept_id = cm.concept_id;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(sql_functions)
            con.commit()
    
    print("  ✅ Search functions created!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = DEFAULT_CONFIG
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   BioAgent Molecular Integration Pipeline                          ║")
    print("║   With RDKit-based Normalization & Hierarchy                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # 0. Create Schema
    print("\n📋 Creating schema...")
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(CREATE_SCHEMA_SQL)
            con.commit()
    print("  ✅ Schema created.")
    
    # 0.1 Create RDKit helper functions
    print("\n🧪 Creating RDKit helper functions...")
    with get_connection(config) as con:
        with con.cursor() as cur:
            cur.execute(CREATE_RDKIT_FUNCTIONS_SQL)
            con.commit()
    print("  ✅ RDKit functions created.")
    
    # 1. Populate Molecules with Normalization
    populate_molecules(config)
    
    # 2. Build Hierarchy (Concepts and Stereo Forms)
    build_molecule_hierarchy(config)
    
    # 3. Populate Synonyms
    populate_synonyms(config)
    
    # 4. Map Clinical Trials
    map_clinical_trials(config)
    
    # 5. Create Analytics View
    create_materialized_analytics_view(config)
    
    # 6. Create Search Functions
    create_search_functions(config)
    
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        EXAMPLE QUERIES                                ║
╚══════════════════════════════════════════════════════════════════════╝

-- 1. Get all forms of ruxolitinib (including salts, stereoisomers)
SELECT * FROM get_drug_forms('ruxolitinib');

-- 2. Get concept summary (how many forms, trials, targets)
SELECT * FROM get_concept_summary('ruxolitinib');

-- 3. Find all trials for any form of ruxolitinib
SELECT * FROM find_trials_for_drug_concept('ruxolitinib');

-- 4. Find targets for ruxolitinib (aggregated across all forms)
SELECT * FROM find_targets_for_drug_concept('ruxolitinib', 6.0);

-- 5. Find similar molecules to ruxolitinib
SELECT * FROM find_similar_molecules(
    'N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1',  -- Ruxolitinib SMILES
    0.7,  -- 70% similarity threshold
    20    -- max results
);

-- 6. Find trials for structurally similar molecules
SELECT * FROM find_trials_by_similar_molecules(
    'N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1',
    0.6,
    50
);

-- 7. Direct query: JAK2 inhibitors in clinical trials
SELECT 
    mc.preferred_name as drug,
    MAX(cta.pchembl_value) as potency,
    COUNT(DISTINCT s.nct_id) as n_trials,
    ARRAY_AGG(DISTINCT s.phase) as phases
FROM dm_molecule_concept mc
JOIN dm_molecule dm ON mc.concept_id = dm.concept_id  
JOIN dm_compound_target_activity cta ON dm.mol_id = cta.mol_id
JOIN map_ctgov_molecules map ON mc.concept_id = map.concept_id
JOIN ctgov_studies s ON map.nct_id = s.nct_id
WHERE cta.gene_symbol = 'JAK2'
  AND cta.pchembl_value >= 7
GROUP BY mc.preferred_name
ORDER BY potency DESC, n_trials DESC;

-- 8. Substructure search (find all molecules with pyrrolo[2,3-d]pyrimidine)
SELECT * FROM find_molecules_by_substructure(
    'c1cnc2[nH]ccc2n1',  -- Pyrrolo[2,3-d]pyrimidine core
    50
);
""")


if __name__ == "__main__":
    main()