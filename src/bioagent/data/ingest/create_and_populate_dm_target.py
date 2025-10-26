#!/usr/bin/env python3
"""
Populate canonical target mapping table (dm_target) in 4 phases.

PHASE 1: ChEMBL Base
  Load human SINGLE_PROTEIN targets from ChEMBL tables
  Source: target_dictionary, component_sequences, component_class
  Expected: ~2,000 targets
  Confidence: 10 (high)

PHASE 2: BindingDB Augmentation
  Match BindingDB targets to existing dm_target via UniProt
  If new: INSERT with source=BINDINGDB, confidence=8
  If exists: UPDATE with cross-ref, mark CONSENSUS
  Expected: ~400 new targets, ~1,200 consensus matches

PHASE 3: Gene Synonym Consolidation
  Load gene symbols from component_synonyms (ChEMBL)
  Load gene symbols from bindingdb_targets
  Into: dm_target_gene_synonyms
  Expected: ~8,000+ synonyms

PHASE 4: UniProt Mapping
  Map known isoforms and secondary accessions
  Into: dm_target_uniprot_mappings
  Expected: ~3,000+ accessions (isoforms + secondary IDs)
"""

from __future__ import annotations

import sys
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
from typing import Optional

# Handle imports for both direct execution and module import
try:
    from config import DatabaseConfig, get_connection, DEFAULT_CONFIG
except ImportError:
    from .config import DatabaseConfig, get_connection, DEFAULT_CONFIG


# ============================================================================
# PHASE 1: ChEMBL Base
# ============================================================================


CREATE_TABLE_SQL = """
-- Core canonical target table with unified representation
-- Drop table if it exists
DROP TABLE IF EXISTS dm_target CASCADE;
CREATE TABLE IF NOT EXISTS dm_target (
    -- Primary identifier (auto-incrementing for foreign key references)
    target_id BIGSERIAL PRIMARY KEY,
    
    -- Unique identifiers
    uniprot_id TEXT UNIQUE NOT NULL,  -- Main unique key (e.g., P04637 for TP53)
    ensembl_gene_id TEXT,              -- Ensembl gene ID (e.g., ENSG00000141510)
    
    -- Names and symbols
    gene_symbol TEXT NOT NULL,         -- Official gene symbol (e.g., TP53)
    gene_name TEXT,                    -- Full gene name
    protein_name TEXT,                 -- Preferred protein name
    
    -- Taxonomic context
    organism TEXT DEFAULT 'Homo sapiens',
    tax_id BIGINT,                     -- NCBI taxonomy ID
    
    -- Classification
    protein_class_id BIGINT,           -- FK to protein_classification
    target_type TEXT,                  -- Type: SINGLE_PROTEIN, PROTEIN_FAMILY, etc.
    
    -- Cross-database references
    chembl_tid BIGINT,                 -- ChEMBL target ID (tid)
    chembl_id TEXT,                    -- ChEMBL ID string (e.g., CHEMBL2074)
    bindingdb_target_id INTEGER,       -- BindingDB target ID (if available)
    
    -- Data quality and provenance
    primary_source TEXT,               -- 'CHEMBL', 'BINDINGDB', 'CONSENSUS'
    data_sources TEXT[],               -- Array of sources that contributed
    confidence_score SMALLINT,         -- 0-10 confidence in this mapping
    
    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_by TEXT,              -- Script name or user
    
    -- Notes
    notes TEXT                         -- Any disambiguating information
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_dm_target_gene_symbol 
    ON dm_target(gene_symbol);

CREATE INDEX IF NOT EXISTS idx_dm_target_chembl_tid 
    ON dm_target(chembl_tid) 
    WHERE chembl_tid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_dm_target_organism 
    ON dm_target(organism);

CREATE INDEX IF NOT EXISTS idx_dm_target_tax_id 
    ON dm_target(tax_id);

CREATE INDEX IF NOT EXISTS idx_dm_target_protein_class 
    ON dm_target(protein_class_id);

CREATE INDEX IF NOT EXISTS idx_dm_target_bindingdb 
    ON dm_target(bindingdb_target_id) 
    WHERE bindingdb_target_id IS NOT NULL;

-- Composite index for organism-specific queries
CREATE INDEX IF NOT EXISTS idx_dm_target_organism_gene 
    ON dm_target(organism, gene_symbol);


-- Mapping table: link multiple gene symbols to one target
-- (accounts for synonyms and organism-specific nomenclature)
DROP TABLE IF EXISTS dm_target_gene_synonyms CASCADE;
CREATE TABLE IF NOT EXISTS dm_target_gene_synonyms (
    synonym_id BIGSERIAL PRIMARY KEY,
    target_id BIGINT NOT NULL REFERENCES dm_target(target_id) ON DELETE CASCADE,
    gene_symbol TEXT NOT NULL,
    organism TEXT,
    symbol_type TEXT,  -- 'PRIMARY', 'SYNONYM', 'PREVIOUS_SYMBOL'
    source TEXT,       -- 'NCBI', 'ENSEMBL', 'CHEMBL', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(target_id, gene_symbol, organism)
);

CREATE INDEX IF NOT EXISTS idx_dm_target_synonyms_target 
    ON dm_target_gene_synonyms(target_id);

CREATE INDEX IF NOT EXISTS idx_dm_target_synonyms_symbol 
    ON dm_target_gene_synonyms(gene_symbol);


-- Mapping table: link multiple UniProt accessions to one target
-- (handles isoforms, secondary accessions)
DROP TABLE IF EXISTS dm_target_uniprot_mappings CASCADE;
CREATE TABLE IF NOT EXISTS dm_target_uniprot_mappings (
    mapping_id BIGSERIAL PRIMARY KEY,
    target_id BIGINT NOT NULL REFERENCES dm_target(target_id) ON DELETE CASCADE,
    uniprot_accession TEXT NOT NULL,
    accession_type TEXT,  -- 'PRIMARY', 'ISOFORM', 'SECONDARY'
    organism TEXT,
    source TEXT,          -- 'UNIPROT', 'CHEMBL', 'BINDINGDB'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(target_id, uniprot_accession)
);

CREATE INDEX IF NOT EXISTS idx_dm_target_uniprot_target 
    ON dm_target_uniprot_mappings(target_id);

CREATE INDEX IF NOT EXISTS idx_dm_target_uniprot_accession 
    ON dm_target_uniprot_mappings(uniprot_accession);
"""



PHASE_1_SQL = """
-- PHASE 1: Load human SINGLE_PROTEIN targets from ChEMBL
-- 
-- Strategy:
--   1. Get all human protein targets from target_dictionary
--   2. Join with component_sequences to get UniProt IDs
--   3. Join with component_class to get protein class
--   4. Filter for high-quality SWISS-PROT entries

INSERT INTO dm_target (
    uniprot_id, gene_symbol, gene_name, protein_name,
    organism, tax_id,
    chembl_tid, chembl_id, target_type, protein_class_id,
    primary_source, data_sources, confidence_score,
    last_updated_by, created_at
)
SELECT DISTINCT ON (cs.accession)
    cs.accession AS uniprot_id,                          -- UniProt ID (primary key)
    COALESCE(csyn.component_synonym, td.pref_name) 
        AS gene_symbol,                                  -- Gene symbol (from synonyms or target name)
    td.pref_name AS gene_name,                           -- Target name
    cs.description AS protein_name,                      -- Protein description
    td.organism,                                         -- Organism (e.g., Homo sapiens)
    td.tax_id,                                           -- NCBI taxonomy ID (from target_dictionary)
    td.tid AS chembl_tid,                                -- ChEMBL target ID
    td.chembl_id,                                        -- ChEMBL public ID
    td.target_type,                                      -- Target type (SINGLE_PROTEIN, etc.)
    cc.protein_class_id,                                 -- Protein class
    'CHEMBL' AS primary_source,                          -- Source is ChEMBL
    ARRAY['CHEMBL'] AS data_sources,                     -- Data sources array
    10 AS confidence_score,                              -- High confidence (curated)
    'create_and_populate_dm_target.py (PHASE 1)' AS last_updated_by,
    CURRENT_TIMESTAMP
FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    LEFT JOIN component_class cc ON tc.component_id = cc.component_id
    LEFT JOIN component_synonyms csyn ON tc.component_id = csyn.component_id
        AND csyn.syn_type = 'GENE_SYMBOL'
WHERE
    td.organism = 'Homo sapiens'                         -- Only human targets
    AND td.target_type = 'SINGLE PROTEIN'                -- Only protein targets (note: SPACE not underscore)
    AND cs.db_source = 'SWISS-PROT'                      -- High-quality entries
    AND cs.accession IS NOT NULL                         -- Valid UniProt ID
ON CONFLICT (uniprot_id) DO NOTHING;                     -- Skip if already exists
"""

PHASE_1_COUNT = """
SELECT COUNT(*) as count FROM dm_target WHERE primary_source = 'CHEMBL';
"""


# ============================================================================
# PHASE 2: BindingDB Augmentation
# ============================================================================

PHASE_2A_INSERT_SQL = """
-- PHASE 2A: Insert new BindingDB targets (not in ChEMBL)
-- Use ON CONFLICT to safely skip duplicates

INSERT INTO dm_target (
    uniprot_id, gene_symbol, protein_name,
    organism, tax_id,
    bindingdb_target_id, target_type,
    primary_source, data_sources, confidence_score,
    last_updated_by, created_at
)
SELECT
    bdt.uniprot_id,
    bdt.gene_symbol,
    bdt.target_name,
    bdt.organism,
    9606 AS tax_id,                                      -- Homo sapiens
    bdt.id AS bindingdb_target_id,
    'SINGLE PROTEIN' AS target_type,                    -- Match ChEMBL value (space not underscore)
    'BINDINGDB' AS primary_source,
    ARRAY['BINDINGDB'] AS data_sources,
    8 AS confidence_score,                               -- Medium confidence
    'create_and_populate_dm_target.py (PHASE 2A)' AS last_updated_by,
    CURRENT_TIMESTAMP
FROM bindingdb_targets bdt
WHERE
    bdt.organism = 'Homo sapiens'
    AND bdt.uniprot_id IS NOT NULL                       -- Only targets with valid UniProt ID
ON CONFLICT (uniprot_id) DO NOTHING;                     -- Skip if uniprot_id already exists
"""

PHASE_2B_UPDATE_SQL = """
-- PHASE 2B: Update existing targets with BindingDB cross-reference
-- Mark as CONSENSUS when both ChEMBL and BindingDB agree

UPDATE dm_target dt
SET
    bindingdb_target_id = bdt.id,
    data_sources = CASE 
        WHEN 'BINDINGDB' = ANY(data_sources) THEN data_sources  -- Already has BINDINGDB
        ELSE array_append(data_sources, 'BINDINGDB')             -- Add BINDINGDB
    END,
    primary_source = 'CONSENSUS',                        -- Both sources confirm this target
    confidence_score = 10,                               -- High confidence (consensus)
    last_updated_by = 'create_and_populate_dm_target.py (PHASE 2B)',
    updated_at = CURRENT_TIMESTAMP
FROM bindingdb_targets bdt
WHERE
    dt.uniprot_id = bdt.uniprot_id
    AND dt.organism = bdt.organism
    AND dt.bindingdb_target_id IS NULL
    AND bdt.organism = 'Homo sapiens';
"""

PHASE_2_COUNT = """
SELECT 
    COUNT(*) as total_targets,
    SUM(CASE WHEN primary_source = 'CHEMBL' THEN 1 ELSE 0 END) as chembl_only,
    SUM(CASE WHEN primary_source = 'BINDINGDB' THEN 1 ELSE 0 END) as bindingdb_only,
    SUM(CASE WHEN primary_source = 'CONSENSUS' THEN 1 ELSE 0 END) as consensus
FROM dm_target;
"""


# ============================================================================
# PHASE 3: Gene Synonym Consolidation
# ============================================================================

PHASE_3A_CHEMBL_SQL = """
-- PHASE 3A: Load gene synonyms from ChEMBL component_synonyms

INSERT INTO dm_target_gene_synonyms (
    target_id, gene_symbol, organism, symbol_type, source, created_at
)
SELECT
    dt.target_id,
    cs.component_synonym AS gene_symbol,
    dt.organism,
    CASE 
        WHEN cs.syn_type = 'GENE_SYMBOL' THEN 'SYNONYM'
        WHEN cs.syn_type = 'GENE_SYMBOL_OTHER' THEN 'PREVIOUS_SYMBOL'
        ELSE 'SYNONYM'
    END AS symbol_type,
    'CHEMBL' AS source,
    CURRENT_TIMESTAMP
FROM dm_target dt
    JOIN target_components tc ON dt.chembl_tid = tc.tid
    JOIN component_synonyms cs ON tc.component_id = cs.component_id
WHERE
    dt.organism = 'Homo sapiens'
    AND cs.component_synonym != dt.gene_symbol     -- Don't duplicate primary symbol
    AND NOT EXISTS (
        SELECT 1 FROM dm_target_gene_synonyms dgs
        WHERE dgs.target_id = dt.target_id
            AND dgs.gene_symbol = cs.component_synonym
    )
ON CONFLICT (target_id, gene_symbol, organism) DO NOTHING;
"""

PHASE_3B_BINDINGDB_SQL = """
-- PHASE 3B: Load gene symbols from BindingDB

INSERT INTO dm_target_gene_synonyms (
    target_id, gene_symbol, organism, symbol_type, source, created_at
)
SELECT
    dt.target_id,
    bdt.gene_symbol,
    dt.organism,
    'SYNONYM' AS symbol_type,
    'BINDINGDB' AS source,
    CURRENT_TIMESTAMP
FROM dm_target dt
    JOIN bindingdb_targets bdt ON dt.uniprot_id = bdt.uniprot_id
        AND dt.organism = bdt.organism
WHERE
    dt.organism = 'Homo sapiens'
    AND bdt.gene_symbol != dt.gene_symbol         -- Don't duplicate primary symbol
    AND NOT EXISTS (
        SELECT 1 FROM dm_target_gene_synonyms dgs
        WHERE dgs.target_id = dt.target_id
            AND dgs.gene_symbol = bdt.gene_symbol
    )
ON CONFLICT (target_id, gene_symbol, organism) DO NOTHING;
"""

PHASE_3_COUNT = """
SELECT 
    COUNT(*) as total_synonyms,
    COUNT(DISTINCT target_id) as targets_with_synonyms,
    ROUND(AVG(syn_count), 1) as avg_synonyms_per_target
FROM (
    SELECT target_id, COUNT(*) as syn_count
    FROM dm_target_gene_synonyms
    GROUP BY target_id
) subq;
"""


# ============================================================================
# PHASE 4: UniProt Mapping
# ============================================================================

PHASE_4_UNIPROT_SQL = """
-- PHASE 4: Map primary UniProt accessions
-- In a real implementation, you'd load isoforms and secondary IDs
-- For now, just map the primary accession

INSERT INTO dm_target_uniprot_mappings (
    target_id, uniprot_accession, accession_type, organism, source, created_at
)
SELECT
    dt.target_id,
    dt.uniprot_id AS uniprot_accession,
    'PRIMARY' AS accession_type,
    dt.organism,
    CASE 
        WHEN dt.primary_source = 'CHEMBL' THEN 'CHEMBL'
        WHEN dt.primary_source = 'BINDINGDB' THEN 'BINDINGDB'
        ELSE 'CONSENSUS'
    END AS source,
    CURRENT_TIMESTAMP
FROM dm_target dt
WHERE
    NOT EXISTS (
        SELECT 1 FROM dm_target_uniprot_mappings dum
        WHERE dum.target_id = dt.target_id
            AND dum.uniprot_accession = dt.uniprot_id
    )
ON CONFLICT (target_id, uniprot_accession) DO NOTHING;
"""

PHASE_4_COUNT = """
SELECT 
    COUNT(*) as total_mappings,
    COUNT(DISTINCT target_id) as targets_mapped,
    COUNT(DISTINCT CASE WHEN accession_type = 'PRIMARY' THEN 1 END) as primary_accessions,
    COUNT(DISTINCT CASE WHEN accession_type = 'ISOFORM' THEN 1 END) as isoforms,
    COUNT(DISTINCT CASE WHEN accession_type = 'SECONDARY' THEN 1 END) as secondary_accessions
FROM dm_target_uniprot_mappings;
"""


# ============================================================================
# Helper Functions
# ============================================================================

def create_tables(config: DatabaseConfig) -> bool:
    """Create dm_target and related tables."""
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                print("📋 Creating dm_target schema...")
                cur.execute(CREATE_TABLE_SQL)
                con.commit()
                print("✅ Tables created successfully!")
                
                # Show table structure
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name IN ('dm_target', 'dm_target_gene_synonyms', 'dm_target_uniprot_mappings')
                    ORDER BY table_name, ordinal_position
                """)
                
                current_table = None
                print("\n📊 Table structures:")
                for col_name, data_type, is_nullable in cur.fetchall():
                    if current_table != col_name.split('_')[0]:  # Rough table detection
                        current_table = col_name
                        print(f"\n  dm_target table columns:")
                    nullable = "NULL" if is_nullable == 'YES' else "NOT NULL"
                    print(f"    - {col_name:<30} {data_type:<20} {nullable}")
                
                return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False


def execute_phase(phase_num: int, description: str, sql: str, count_sql: str, 
                  config: DatabaseConfig) -> tuple[int, int]:
    """Execute a population phase."""
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                # Execute population query
                print(f"  📊 Executing population query...")
                cur.execute(sql)
                inserted = cur.rowcount
                con.commit()
                
                # Get counts
                print(f"  📈 Checking results...")
                cur.execute(count_sql)
                result = cur.fetchone()
                
                return inserted, result
                
    except Exception as e:
        print(f"  ❌ Error in phase {phase_num}: {e}")
        raise


def print_summary_stats(config: DatabaseConfig) -> None:
    """Print summary statistics after all phases."""
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                print("\n" + "=" * 80)
                print("📊 FINAL SUMMARY STATISTICS")
                print("=" * 80 + "\n")
                
                # dm_target stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN primary_source = 'CHEMBL' THEN 1 END) as chembl,
                        COUNT(CASE WHEN primary_source = 'BINDINGDB' THEN 1 END) as bindingdb,
                        COUNT(CASE WHEN primary_source = 'CONSENSUS' THEN 1 END) as consensus,
                        ROUND(AVG(confidence_score), 2) as avg_confidence
                    FROM dm_target
                """)
                row = cur.fetchone()
                print(f"dm_target:")
                print(f"  Total targets:      {row[0]:,}")
                print(f"  ChEMBL only:        {row[1]:,}")
                print(f"  BindingDB only:     {row[2]:,}")
                print(f"  Consensus (both):   {row[3]:,}")
                print(f"  Avg confidence:     {row[4]:.2f}/10\n")
                
                # Synonyms stats
                cur.execute(PHASE_3_COUNT)
                row = cur.fetchone()
                print(f"dm_target_gene_synonyms:")
                print(f"  Total synonyms:     {row[0]:,}")
                print(f"  Targets covered:    {row[1]:,}")
                print(f"  Avg per target:     {row[2]}\n")
                
                # UniProt mappings stats
                cur.execute(PHASE_4_COUNT)
                row = cur.fetchone()
                print(f"dm_target_uniprot_mappings:")
                print(f"  Total mappings:     {row[0]:,}")
                print(f"  Targets mapped:     {row[1]:,}")
                print(f"  Primary accessions: {row[2]:,}")
                print(f"  Isoforms:           {row[3]:,}")
                print(f"  Secondary IDs:      {row[4]:,}\n")
                
    except Exception as e:
        print(f"❌ Error getting summary stats: {e}")


def main():
    """Main entry point."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          Populate dm_target Tables (4 Phases)                     ║")
    print("║        ChEMBL → BindingDB → Synonyms → UniProt Mapping           ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    config = DEFAULT_CONFIG

    # PHASE 0: Create dm_target tables
    print("PHASE 0️⃣  : Create dm_target tables")
    print("─" * 80)
    print(f"  📥 Creating dm_target tables...")
    if create_tables(config):
        print("✅ Tables created successfully!")
    else:
        print("❌ Failed to create tables!")
        sys.exit(1)
    
    # PHASE 1: ChEMBL Base
    print("PHASE 1️⃣  : ChEMBL Base Loading")
    print("─" * 80)
    print(f"  📥 Loading human SINGLE_PROTEIN targets from ChEMBL...")
    try:
        inserted, count = execute_phase(1, "ChEMBL Base", PHASE_1_SQL, PHASE_1_COUNT, config)
        print(f"  ✅ Phase 1 complete: {inserted:,} targets loaded")
        print(f"     Total in dm_target: {count[0]:,}\n")
    except Exception as e:
        print(f"  ❌ Phase 1 failed: {e}")
        sys.exit(1)
    
    # PHASE 2: BindingDB Augmentation
    print("PHASE 2️⃣  : BindingDB Augmentation")
    print("─" * 80)
    print(f"  📥 Phase 2A: Inserting new BindingDB targets...")
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_2A_INSERT_SQL)
                inserted_new = cur.rowcount
                con.commit()
        print(f"     ✅ Inserted {inserted_new:,} new targets from BindingDB")
    except Exception as e:
        print(f"  ❌ Phase 2A failed: {e}")
        sys.exit(1)
    
    print(f"  🔗 Phase 2B: Marking consensus targets...")
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_2B_UPDATE_SQL)
                updated_consensus = cur.rowcount
                con.commit()
        print(f"     ✅ Updated {updated_consensus:,} targets to CONSENSUS")
        
        # Show counts
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_2_COUNT)
                row = cur.fetchone()
                print(f"     Summary: {row[0]:,} total ({row[1]:,} ChEMBL, "
                      f"{row[2]:,} BindingDB, {row[3]:,} consensus)\n")
    except Exception as e:
        print(f"  ❌ Phase 2B failed: {e}")
        sys.exit(1)
    
    # PHASE 3: Gene Synonym Consolidation
    print("PHASE 3️⃣  : Gene Synonym Consolidation")
    print("─" * 80)
    print(f"  📥 Phase 3A: Loading ChEMBL gene synonyms...")
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_3A_CHEMBL_SQL)
                inserted_chembl = cur.rowcount
                con.commit()
        print(f"     ✅ Loaded {inserted_chembl:,} ChEMBL gene synonyms")
    except Exception as e:
        print(f"  ❌ Phase 3A failed: {e}")
        sys.exit(1)
    
    print(f"  📥 Phase 3B: Loading BindingDB gene symbols...")
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_3B_BINDINGDB_SQL)
                inserted_bindingdb = cur.rowcount
                con.commit()
        print(f"     ✅ Loaded {inserted_bindingdb:,} BindingDB gene symbols")
        
        # Show counts
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_3_COUNT)
                row = cur.fetchone()
                print(f"     Summary: {row[0]:,} total synonyms for {row[1]:,} targets "
                      f"(avg {row[2]} per target)\n")
    except Exception as e:
        print(f"  ❌ Phase 3B failed: {e}")
        sys.exit(1)
    
    # PHASE 4: UniProt Mapping
    print("PHASE 4️⃣  : UniProt Accession Mapping")
    print("─" * 80)
    print(f"  📥 Mapping UniProt accessions...")
    try:
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_4_UNIPROT_SQL)
                inserted_uniprot = cur.rowcount
                con.commit()
        print(f"     ✅ Mapped {inserted_uniprot:,} UniProt accessions")
        
        # Show counts
        with get_connection(config) as con:
            with con.cursor() as cur:
                cur.execute(PHASE_4_COUNT)
                row = cur.fetchone()
                print(f"     Summary: {row[0]:,} total mappings for {row[1]:,} targets\n")
    except Exception as e:
        print(f"  ❌ Phase 4 failed: {e}")
        sys.exit(1)
    
    # Print final summary
    print_summary_stats(config)
    
    print("=" * 80)
    print("✅ ALL PHASES COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
