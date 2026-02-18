-- Enhanced ChEMBL integration with comprehensive molecule annotation
-- Supports the JSON structure with primary/off-targets, mechanisms, and multiple IDs

-- 1. Enhanced molecule mapping with multiple identifiers
CREATE TABLE IF NOT EXISTS public.intervention_molecule (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES public.ctgov_interventions(id) ON DELETE CASCADE,
    raw_name VARCHAR(1000),                  -- Original name from CTG
    
    -- ChEMBL data
    chembl_molregno INTEGER NOT NULL,
    chembl_id VARCHAR(20) NOT NULL,
    preferred_name VARCHAR(500),
    
    -- Chemical structure
    canonical_smiles TEXT,
    isomeric_smiles TEXT,
    inchi TEXT,
    inchi_key VARCHAR(27),
    molecular_formula VARCHAR(100),
    molecular_weight FLOAT,
    
    -- External identifiers (stored as JSONB for flexibility)
    external_ids JSONB,                      -- {"cid": 12345, "unii": "...", "rxcui": "...", "drugbank": "..."}
    
    -- Drug classification
    atc_codes VARCHAR[],                     -- ATC classification codes
    max_phase SMALLINT,                      -- 0-4, 4=approved
    molecule_type VARCHAR(50),               -- Small molecule, Protein, Antibody, etc.
    is_approved BOOLEAN GENERATED ALWAYS AS (max_phase = 4) STORED,
    
    -- Mapping metadata
    confidence_score FLOAT DEFAULT 1.0,
    mapping_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(intervention_id, chembl_molregno)
);

COMMENT ON TABLE public.intervention_molecule IS 
'Comprehensive molecule data for clinical trial interventions with multiple identifiers';

COMMENT ON COLUMN public.intervention_molecule.external_ids IS 
'JSON object with external database IDs: {"cid": PubChem CID, "unii": FDA UNII, "rxcui": RxNorm, "drugbank": DrugBank ID}';

COMMENT ON COLUMN public.intervention_molecule.atc_codes IS 
'ATC (Anatomical Therapeutic Chemical) classification codes';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_im_intervention ON public.intervention_molecule(intervention_id);
CREATE INDEX IF NOT EXISTS idx_im_chembl_id ON public.intervention_molecule(chembl_id);
CREATE INDEX IF NOT EXISTS idx_im_inchi_key ON public.intervention_molecule(inchi_key) WHERE inchi_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_im_external_ids ON public.intervention_molecule USING gin(external_ids);
CREATE INDEX IF NOT EXISTS idx_im_approved ON public.intervention_molecule(is_approved) WHERE is_approved = TRUE;


-- 2. Drug-target binding with primary/off-target classification
CREATE TABLE IF NOT EXISTS public.intervention_target (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES public.ctgov_interventions(id) ON DELETE CASCADE,
    molecule_id INTEGER REFERENCES public.intervention_molecule(id) ON DELETE CASCADE,
    
    -- Target identification
    target_chembl_id VARCHAR(20),
    target_name VARCHAR(500),
    gene_symbol VARCHAR(50),                 -- BTK, ITK, EGFR, etc.
    uniprot_id VARCHAR(20),
    organism VARCHAR(100) DEFAULT 'Homo sapiens',
    target_type VARCHAR(50),                 -- SINGLE PROTEIN, PROTEIN COMPLEX, etc.
    
    -- Target classification
    is_primary_target BOOLEAN DEFAULT FALSE,  -- TRUE if this is the intended therapeutic target
    target_class VARCHAR(100),               -- Kinase, GPCR, Ion channel, etc.
    
    -- Mechanism of action
    action_type VARCHAR(100),                -- Inhibitor, Agonist, Antagonist, Modulator, etc.
    mechanism VARCHAR(500),                  -- "covalent inhibitor", "competitive inhibitor", etc.
    
    -- Binding data (best values across all assays)
    best_ic50_nm NUMERIC,
    best_ki_nm NUMERIC,
    best_ec50_nm NUMERIC,
    best_kd_nm NUMERIC,
    pchembl_value NUMERIC,                   -- Normalized potency (-log molar)
    
    -- Assay statistics
    num_assays INTEGER DEFAULT 0,
    confidence_score INTEGER,                -- ChEMBL confidence (0-9)
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(molecule_id, target_chembl_id)
);

COMMENT ON TABLE public.intervention_target IS 
'Drug-target interactions with primary/off-target classification and mechanism of action';

COMMENT ON COLUMN public.intervention_target.is_primary_target IS 
'TRUE if this is the intended therapeutic target (determined by mechanism studies, clinical indication, etc.)';

COMMENT ON COLUMN public.intervention_target.pchembl_value IS 
'Normalized potency: -log(molar IC50/Ki/EC50). Higher values = more potent. >6 is significant.';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_it_intervention ON public.intervention_target(intervention_id);
CREATE INDEX IF NOT EXISTS idx_it_molecule ON public.intervention_target(molecule_id);
CREATE INDEX IF NOT EXISTS idx_it_gene ON public.intervention_target(gene_symbol) WHERE gene_symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_it_uniprot ON public.intervention_target(uniprot_id) WHERE uniprot_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_it_primary ON public.intervention_target(is_primary_target) WHERE is_primary_target = TRUE;
CREATE INDEX IF NOT EXISTS idx_it_potency ON public.intervention_target(pchembl_value DESC NULLS LAST) WHERE pchembl_value IS NOT NULL;


-- 3. All activity measurements (for detailed analysis)
CREATE TABLE IF NOT EXISTS public.intervention_activity (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES public.intervention_molecule(id) ON DELETE CASCADE,
    target_id INTEGER REFERENCES public.intervention_target(id) ON DELETE CASCADE,
    
    -- Activity measurement
    activity_type VARCHAR(50),               -- IC50, Ki, EC50, Kd
    standard_value NUMERIC,
    standard_units VARCHAR(20),
    pchembl_value NUMERIC,
    
    -- Assay details
    assay_chembl_id VARCHAR(20),
    assay_type VARCHAR(50),                  -- B=binding, F=functional
    assay_description TEXT,
    
    -- Source
    doc_chembl_id VARCHAR(20),               -- ChEMBL document ID
    confidence_score INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE public.intervention_activity IS 
'Individual bioactivity measurements from ChEMBL assays';

CREATE INDEX IF NOT EXISTS idx_ia_molecule ON public.intervention_activity(molecule_id);
CREATE INDEX IF NOT EXISTS idx_ia_target ON public.intervention_activity(target_id);
CREATE INDEX IF NOT EXISTS idx_ia_type_value ON public.intervention_activity(activity_type, standard_value) 
    WHERE standard_value IS NOT NULL;


-- 4. View: Comprehensive molecule annotation (for JSON export)
CREATE OR REPLACE VIEW public.intervention_molecule_annotation AS
SELECT 
    i.id as intervention_id,
    i.nct_id,
    i.name as intervention_name,
    im.raw_name,
    
    -- Molecule details
    jsonb_build_object(
        'preferred_name', im.preferred_name,
        'chembl_id', im.chembl_id,
        'inchikey', im.inchi_key,
        'smiles', im.canonical_smiles,
        'formula', im.molecular_formula,
        'weight', im.molecular_weight,
        'ids', im.external_ids,
        'atc', im.atc_codes,
        'max_phase', im.max_phase,
        'is_approved', im.is_approved,
        'molecule_type', im.molecule_type
    ) as molecule,
    
    -- Primary targets
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'gene', it.gene_symbol,
                'name', it.target_name,
                'uniprot', it.uniprot_id,
                'action', it.action_type,
                'mechanism', it.mechanism,
                'ic50_nm', it.best_ic50_nm,
                'ki_nm', it.best_ki_nm,
                'pchembl', it.pchembl_value
            ) ORDER BY it.pchembl_value DESC NULLS LAST
        )
        FROM intervention_target it
        WHERE it.intervention_id = i.id 
          AND it.is_primary_target = TRUE
    ) as targets_primary,
    
    -- Off-targets (secondary targets)
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'gene', it.gene_symbol,
                'name', it.target_name,
                'uniprot', it.uniprot_id,
                'ic50_nm', it.best_ic50_nm,
                'ki_nm', it.best_ki_nm,
                'pchembl', it.pchembl_value
            ) ORDER BY it.pchembl_value DESC NULLS LAST
        )
        FROM intervention_target it
        WHERE it.intervention_id = i.id 
          AND it.is_primary_target = FALSE
          AND it.pchembl_value > 6  -- Only significant off-targets
    ) as targets_off,
    
    -- Best potency organized by target
    (
        SELECT jsonb_object_agg(
            it.gene_symbol,
            jsonb_build_object(
                'IC50', it.best_ic50_nm,
                'Ki', it.best_ki_nm,
                'EC50', it.best_ec50_nm,
                'Kd', it.best_kd_nm,
                'pChEMBL', it.pchembl_value
            )
        )
        FROM intervention_target it
        WHERE it.intervention_id = i.id
          AND it.gene_symbol IS NOT NULL
    ) as best_potency_by_target
    
FROM ctgov_interventions i
JOIN intervention_molecule im ON i.id = im.intervention_id
WHERE i.intervention_type IN ('DRUG', 'BIOLOGICAL');

COMMENT ON VIEW public.intervention_molecule_annotation IS 
'Complete molecule annotation with targets, potencies, and identifiers for JSON export';


-- 5. Function: Get molecule annotation as JSON
CREATE OR REPLACE FUNCTION public.get_intervention_molecules(p_nct_id VARCHAR)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'interventions_resolved', 
        COALESCE(jsonb_agg(
            jsonb_build_object(
                'raw', ima.raw_name,
                'molecule', ima.molecule,
                'targets_primary', ima.targets_primary,
                'targets_off', ima.targets_off,
                'best_potency_nM', ima.best_potency_by_target
            )
        ), '[]'::jsonb)
    )
    INTO result
    FROM intervention_molecule_annotation ima
    WHERE ima.nct_id = p_nct_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.get_intervention_molecules IS 
'Returns comprehensive molecule annotation for a trial in the desired JSON format';


-- 6. Helper view: Primary target summary
CREATE OR REPLACE VIEW public.intervention_primary_targets AS
SELECT 
    i.nct_id,
    i.name as drug_name,
    im.preferred_name,
    im.chembl_id,
    it.gene_symbol,
    it.target_name,
    it.uniprot_id,
    it.action_type,
    it.mechanism,
    it.best_ic50_nm,
    it.best_ki_nm,
    it.pchembl_value,
    im.atc_codes,
    im.is_approved
FROM ctgov_interventions i
JOIN intervention_molecule im ON i.id = im.intervention_id
JOIN intervention_target it ON im.id = it.molecule_id
WHERE it.is_primary_target = TRUE
ORDER BY it.pchembl_value DESC NULLS LAST;

COMMENT ON VIEW public.intervention_primary_targets IS 
'Primary therapeutic targets for clinical trial interventions';


-- 7. Helper function: Classify primary vs off-targets based on potency
CREATE OR REPLACE FUNCTION public.classify_primary_targets()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    -- Classify as primary target if:
    -- 1. Most potent target (highest pChEMBL value)
    -- 2. At least 10x more potent than next target (>1 log difference)
    
    WITH ranked_targets AS (
        SELECT 
            id,
            molecule_id,
            pchembl_value,
            ROW_NUMBER() OVER (PARTITION BY molecule_id ORDER BY pchembl_value DESC NULLS LAST) as rank,
            LAG(pchembl_value) OVER (PARTITION BY molecule_id ORDER BY pchembl_value DESC NULLS LAST) as next_potency
        FROM intervention_target
        WHERE pchembl_value IS NOT NULL
    ),
    primary_targets AS (
        SELECT id
        FROM ranked_targets
        WHERE rank = 1 
          AND (next_potency IS NULL OR pchembl_value - next_potency >= 1.0)  -- At least 10x more potent
    )
    UPDATE intervention_target
    SET is_primary_target = (id IN (SELECT id FROM primary_targets))
    WHERE id IN (SELECT id FROM ranked_targets);
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.classify_primary_targets IS 
'Auto-classify primary vs off-targets based on potency selectivity';


-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON public.intervention_molecule TO database_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.intervention_target TO database_user;
GRANT SELECT, INSERT ON public.intervention_activity TO database_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO database_user;
GRANT SELECT ON public.intervention_molecule_annotation TO database_user;
GRANT SELECT ON public.intervention_primary_targets TO database_user;
GRANT EXECUTE ON FUNCTION public.get_intervention_molecules TO database_user;
GRANT EXECUTE ON FUNCTION public.classify_primary_targets TO database_user;


-- Example usage
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '✓ Enhanced ChEMBL tables created!';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables:';
    RAISE NOTICE '  - intervention_molecule (comprehensive molecule data)';
    RAISE NOTICE '  - intervention_target (primary/off-targets with mechanism)';
    RAISE NOTICE '  - intervention_activity (individual assay measurements)';
    RAISE NOTICE '';
    RAISE NOTICE 'Views:';
    RAISE NOTICE '  - intervention_molecule_annotation (JSON-ready view)';
    RAISE NOTICE '  - intervention_primary_targets (primary target summary)';
    RAISE NOTICE '';
    RAISE NOTICE 'Functions:';
    RAISE NOTICE '  - get_intervention_molecules(nct_id) -> JSONB';
    RAISE NOTICE '  - classify_primary_targets() -> auto-classify targets';
    RAISE NOTICE '';
    RAISE NOTICE 'Example query:';
    RAISE NOTICE '  SELECT get_intervention_molecules(''NCT00000620'');';
    RAISE NOTICE '';
    RAISE NOTICE 'Next: Run enhanced mapping script';
    RAISE NOTICE '  python src/bioagent/data/ingest/map_chembl_enhanced.py';
    RAISE NOTICE '';
END $$;




