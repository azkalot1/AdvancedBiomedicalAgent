-- Create tables for mapping ClinicalTrials.gov interventions to ChEMBL molecules
-- Run this in your main "database" database

-- 1. Intervention to ChEMBL molecule mapping
CREATE TABLE IF NOT EXISTS public.intervention_chembl_mapping (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES public.ctgov_interventions(id) ON DELETE CASCADE,
    matched_name VARCHAR(1000),              -- Which name/alias matched
    chembl_molregno INTEGER NOT NULL,        -- ChEMBL internal ID
    chembl_id VARCHAR(20) NOT NULL,          -- ChEMBL public ID (e.g., CHEMBL25)
    canonical_smiles TEXT,                   -- SMILES structure
    inchi_key VARCHAR(27),                   -- InChI Key for structure search
    max_phase SMALLINT,                      -- Drug development phase (0-4, 4=approved)
    confidence_score FLOAT DEFAULT 1.0,      -- Matching confidence (0-1)
    mapping_method VARCHAR(50),              -- 'exact', 'synonym', 'fuzzy', 'manual'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(intervention_id, chembl_molregno)
);

COMMENT ON TABLE public.intervention_chembl_mapping IS 
'Maps ClinicalTrials.gov interventions to ChEMBL molecules';

COMMENT ON COLUMN public.intervention_chembl_mapping.max_phase IS 
'0=preclinical, 1=phase I, 2=phase II, 3=phase III, 4=approved';

COMMENT ON COLUMN public.intervention_chembl_mapping.confidence_score IS 
'1.0=exact match, 0.9=synonym match, <0.9=fuzzy match';

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_icm_intervention 
    ON public.intervention_chembl_mapping(intervention_id);

CREATE INDEX IF NOT EXISTS idx_icm_chembl_id 
    ON public.intervention_chembl_mapping(chembl_id);

CREATE INDEX IF NOT EXISTS idx_icm_molregno 
    ON public.intervention_chembl_mapping(chembl_molregno);

CREATE INDEX IF NOT EXISTS idx_icm_inchi_key 
    ON public.intervention_chembl_mapping(inchi_key) 
    WHERE inchi_key IS NOT NULL;


-- 2. Cached drug-target binding data
CREATE TABLE IF NOT EXISTS public.intervention_target_binding (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES public.ctgov_interventions(id) ON DELETE CASCADE,
    chembl_id VARCHAR(20) NOT NULL,          -- ChEMBL molecule ID
    target_chembl_id VARCHAR(20),            -- ChEMBL target ID
    target_name VARCHAR(500),                -- Protein target name
    uniprot_id VARCHAR(20),                  -- UniProt accession
    gene_name VARCHAR(100),                  -- Gene symbol
    organism VARCHAR(100),                   -- Homo sapiens, etc.
    activity_type VARCHAR(50),               -- IC50, Ki, EC50, Kd
    standard_value NUMERIC,                  -- Activity value
    standard_units VARCHAR(20),              -- nM, uM, etc.
    pchembl_value NUMERIC,                   -- -log(molar IC50/Ki/EC50)
    assay_type VARCHAR(50),                  -- B=binding, F=functional
    confidence_score INTEGER,                -- ChEMBL confidence (0-9, 9=highest)
    source_doc_id INTEGER,                   -- ChEMBL document ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE public.intervention_target_binding IS 
'Cached drug-target binding data from ChEMBL for CTG interventions';

COMMENT ON COLUMN public.intervention_target_binding.pchembl_value IS 
'Normalized activity value: -log(molar IC50/Ki/EC50). Higher = more potent.';

-- Indexes for binding queries
CREATE INDEX IF NOT EXISTS idx_itb_intervention 
    ON public.intervention_target_binding(intervention_id);

CREATE INDEX IF NOT EXISTS idx_itb_chembl_id 
    ON public.intervention_target_binding(chembl_id);

CREATE INDEX IF NOT EXISTS idx_itb_target 
    ON public.intervention_target_binding(target_chembl_id);

CREATE INDEX IF NOT EXISTS idx_itb_uniprot 
    ON public.intervention_target_binding(uniprot_id) 
    WHERE uniprot_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_itb_activity 
    ON public.intervention_target_binding(activity_type, standard_value) 
    WHERE standard_value IS NOT NULL;


-- 3. Foreign data wrapper for easy ChEMBL access (optional)
-- This allows querying ChEMBL database directly from main database

-- Uncomment to enable (requires postgres_fdw extension)
/*
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

CREATE SERVER IF NOT EXISTS chembl_server
    FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'localhost', dbname 'chembl_34', port '5432');

CREATE USER MAPPING IF NOT EXISTS FOR database_user
    SERVER chembl_server
    OPTIONS (user 'database_user', password 'database_password');

-- Import key ChEMBL tables as foreign tables
IMPORT FOREIGN SCHEMA public
    LIMIT TO (
        molecule_dictionary,
        compound_structures,
        activities,
        assays,
        target_dictionary,
        target_components,
        component_sequences,
        molecule_synonyms
    )
    FROM SERVER chembl_server
    INTO public;
*/


-- 4. Useful views for common queries

-- View: Approved drugs in clinical trials
CREATE OR REPLACE VIEW public.intervention_approved_drugs AS
SELECT 
    i.id as intervention_id,
    i.nct_id,
    i.name as intervention_name,
    icm.chembl_id,
    icm.canonical_smiles,
    icm.max_phase,
    icm.confidence_score,
    COUNT(DISTINCT itb.target_chembl_id) as num_known_targets
FROM public.ctgov_interventions i
JOIN public.intervention_chembl_mapping icm ON i.id = icm.intervention_id
LEFT JOIN public.intervention_target_binding itb ON i.id = itb.intervention_id
WHERE icm.max_phase = 4  -- FDA approved
GROUP BY i.id, i.nct_id, i.name, icm.chembl_id, icm.canonical_smiles, icm.max_phase, icm.confidence_score;

COMMENT ON VIEW public.intervention_approved_drugs IS 
'FDA-approved drugs used in clinical trials with their ChEMBL mappings';


-- View: High-confidence drug-target binding (<100nM IC50/Ki)
CREATE OR REPLACE VIEW public.intervention_potent_binding AS
SELECT 
    i.nct_id,
    i.name as drug_name,
    itb.chembl_id,
    itb.target_name,
    itb.uniprot_id,
    itb.gene_name,
    itb.activity_type,
    itb.standard_value,
    itb.standard_units,
    itb.pchembl_value
FROM public.ctgov_interventions i
JOIN public.intervention_target_binding itb ON i.id = itb.intervention_id
WHERE itb.activity_type IN ('IC50', 'Ki', 'Kd')
  AND itb.standard_units = 'nM'
  AND itb.standard_value < 100  -- High affinity
  AND itb.organism = 'Homo sapiens'
ORDER BY itb.standard_value;

COMMENT ON VIEW public.intervention_potent_binding IS 
'High-affinity drug-target interactions (<100nM) for trial interventions';


-- 5. Helper functions

-- Function: Get all targets for a trial intervention
CREATE OR REPLACE FUNCTION public.get_intervention_targets(p_nct_id VARCHAR)
RETURNS TABLE (
    target_name VARCHAR,
    uniprot_id VARCHAR,
    activity_type VARCHAR,
    best_value NUMERIC,
    units VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        itb.target_name,
        itb.uniprot_id,
        itb.activity_type,
        MIN(itb.standard_value) as best_value,
        itb.standard_units as units
    FROM public.ctgov_interventions i
    JOIN public.intervention_target_binding itb ON i.id = itb.intervention_id
    WHERE i.nct_id = p_nct_id
      AND itb.standard_value IS NOT NULL
    GROUP BY itb.target_name, itb.uniprot_id, itb.activity_type, itb.standard_units
    ORDER BY best_value;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.get_intervention_targets IS 
'Get all known protein targets for interventions in a clinical trial';


-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.intervention_chembl_mapping TO database_user;
GRANT SELECT, INSERT ON public.intervention_target_binding TO database_user;
GRANT USAGE, SELECT ON SEQUENCE intervention_chembl_mapping_id_seq TO database_user;
GRANT USAGE, SELECT ON SEQUENCE intervention_target_binding_id_seq TO database_user;
GRANT SELECT ON public.intervention_approved_drugs TO database_user;
GRANT SELECT ON public.intervention_potent_binding TO database_user;
GRANT EXECUTE ON FUNCTION public.get_intervention_targets TO database_user;


-- Summary
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '✓ ChEMBL mapping tables created successfully!';
    RAISE NOTICE '';
    RAISE NOTICE 'Created tables:';
    RAISE NOTICE '  - intervention_chembl_mapping';
    RAISE NOTICE '  - intervention_target_binding';
    RAISE NOTICE '';
    RAISE NOTICE 'Created views:';
    RAISE NOTICE '  - intervention_approved_drugs';
    RAISE NOTICE '  - intervention_potent_binding';
    RAISE NOTICE '';
    RAISE NOTICE 'Next step: Run mapping script';
    RAISE NOTICE '  python src/bioagent/data/ingest/map_chembl.py';
    RAISE NOTICE '';
END $$;




