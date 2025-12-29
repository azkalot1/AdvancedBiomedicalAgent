#!/usr/bin/env python3
# generate_ctgov_enriched_search.py
"""
Generate enriched clinical trials search table.

Creates a denormalized table optimized for flexible searching with:
- Aggregated conditions, interventions, sponsors
- Normalized text for trigram similarity
- Full-text search vectors
- Filterable metadata columns
- Proper indexes for all search patterns

Usage:
    python generate_ctgov_enriched_search.py create      # Create table and indexes
    python generate_ctgov_enriched_search.py populate    # Populate data
    python generate_ctgov_enriched_search.py refresh     # Refresh data (truncate + reload)
    python generate_ctgov_enriched_search.py verify      # Verify table state
    python generate_ctgov_enriched_search.py test        # Test search queries
    python generate_ctgov_enriched_search.py drop        # Drop table and indexes
    python generate_ctgov_enriched_search.py full        # Create + populate + verify + test
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql as psql
from psycopg2.extras import execute_values
from tqdm import tqdm

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, DEFAULT_CONFIG, get_connection
except ImportError:
    from config import DatabaseConfig, DEFAULT_CONFIG, get_connection


# =============================================================================
# TABLE DEFINITION
# =============================================================================

TABLE_NAME = "rag_study_search"

CREATE_EXTENSIONS_SQL = """
-- Ensure required extensions exist
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
"""

CREATE_TABLE_SQL = """
-- Drop existing table if exists
DROP TABLE IF EXISTS public.rag_study_search CASCADE;

-- Create the enriched search table
CREATE TABLE public.rag_study_search (
    -- Primary key
    nct_id TEXT PRIMARY KEY,
    
    -- === Core Metadata (for filtering) ===
    brief_title TEXT,
    official_title TEXT,
    overall_status TEXT,
    phase TEXT,
    study_type TEXT,
    enrollment INTEGER,
    enrollment_type TEXT,
    
    -- Dates as TEXT (original) and DATE (parsed for range queries)
    start_date TEXT,
    start_date_parsed DATE,
    completion_date TEXT,
    completion_date_parsed DATE,
    primary_completion_date TEXT,
    primary_completion_date_parsed DATE,
    results_first_submitted_date DATE,
    
    -- Regulatory flags
    has_dmc BOOLEAN,
    is_fda_regulated_drug BOOLEAN,
    is_fda_regulated_device BOOLEAN,
    
    -- === Aggregated Arrays (for containment queries) ===
    conditions TEXT[] DEFAULT ARRAY[]::TEXT[],
    mesh_conditions TEXT[] DEFAULT ARRAY[]::TEXT[],
    interventions TEXT[] DEFAULT ARRAY[]::TEXT[],
    mesh_interventions TEXT[] DEFAULT ARRAY[]::TEXT[],
    intervention_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    sponsors TEXT[] DEFAULT ARRAY[]::TEXT[],
    lead_sponsor TEXT,
    collaborators TEXT[] DEFAULT ARRAY[]::TEXT[],
    countries TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- === Normalized Text (for trigram similarity) ===
    conditions_norm TEXT,
    mesh_conditions_norm TEXT,
    interventions_norm TEXT,
    mesh_interventions_norm TEXT,
    sponsors_norm TEXT,
    
    -- === Full-Text Search Vectors ===
    title_description_tsv TSVECTOR,
    terms_tsv TSVECTOR,
    
    -- === Timestamps ===
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add comments
COMMENT ON TABLE public.rag_study_search IS 'Denormalized clinical trials search table with aggregated terms and search vectors';
COMMENT ON COLUMN public.rag_study_search.conditions_norm IS 'Lowercased, unaccented conditions text for trigram search';
COMMENT ON COLUMN public.rag_study_search.title_description_tsv IS 'Full-text vector for title and description search';
COMMENT ON COLUMN public.rag_study_search.terms_tsv IS 'Full-text vector for conditions, interventions, MeSH terms';
"""

CREATE_INDEXES_SQL = """
-- === Trigram indexes for similarity search ===
CREATE INDEX IF NOT EXISTS idx_rag_search_cond_trgm 
    ON public.rag_study_search USING gin (conditions_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_rag_search_mesh_cond_trgm 
    ON public.rag_study_search USING gin (mesh_conditions_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_rag_search_intr_trgm 
    ON public.rag_study_search USING gin (interventions_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_rag_search_mesh_intr_trgm 
    ON public.rag_study_search USING gin (mesh_interventions_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_rag_search_sponsors_trgm 
    ON public.rag_study_search USING gin (sponsors_norm gin_trgm_ops);

-- === Full-text search indexes ===
CREATE INDEX IF NOT EXISTS idx_rag_search_title_desc_tsv 
    ON public.rag_study_search USING gin (title_description_tsv);

CREATE INDEX IF NOT EXISTS idx_rag_search_terms_tsv 
    ON public.rag_study_search USING gin (terms_tsv);

-- === Filter indexes (B-tree for equality/range) ===
CREATE INDEX IF NOT EXISTS idx_rag_search_status 
    ON public.rag_study_search (overall_status);

CREATE INDEX IF NOT EXISTS idx_rag_search_phase 
    ON public.rag_study_search (phase);

CREATE INDEX IF NOT EXISTS idx_rag_search_study_type 
    ON public.rag_study_search (study_type);

CREATE INDEX IF NOT EXISTS idx_rag_search_start_date 
    ON public.rag_study_search (start_date_parsed);

CREATE INDEX IF NOT EXISTS idx_rag_search_completion 
    ON public.rag_study_search (completion_date_parsed);

CREATE INDEX IF NOT EXISTS idx_rag_search_enrollment 
    ON public.rag_study_search (enrollment);

CREATE INDEX IF NOT EXISTS idx_rag_search_has_results 
    ON public.rag_study_search (results_first_submitted_date) 
    WHERE results_first_submitted_date IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_rag_search_fda_drug 
    ON public.rag_study_search (is_fda_regulated_drug) 
    WHERE is_fda_regulated_drug = true;

CREATE INDEX IF NOT EXISTS idx_rag_search_fda_device 
    ON public.rag_study_search (is_fda_regulated_device) 
    WHERE is_fda_regulated_device = true;

-- === Array containment indexes (GIN) ===
CREATE INDEX IF NOT EXISTS idx_rag_search_conditions_arr 
    ON public.rag_study_search USING gin (conditions);

CREATE INDEX IF NOT EXISTS idx_rag_search_interventions_arr 
    ON public.rag_study_search USING gin (interventions);

CREATE INDEX IF NOT EXISTS idx_rag_search_sponsors_arr 
    ON public.rag_study_search USING gin (sponsors);

CREATE INDEX IF NOT EXISTS idx_rag_search_countries_arr 
    ON public.rag_study_search USING gin (countries);

CREATE INDEX IF NOT EXISTS idx_rag_search_intervention_types_arr 
    ON public.rag_study_search USING gin (intervention_types);

-- === Lead sponsor index ===
CREATE INDEX IF NOT EXISTS idx_rag_search_lead_sponsor 
    ON public.rag_study_search (lead_sponsor);
"""


# =============================================================================
# BATCH POPULATION QUERY (using %s for psycopg2)
# =============================================================================

BATCH_SELECT_QUERY = """
WITH 
nct_batch AS (
    SELECT unnest(%(nct_ids)s::text[]) AS nct_id
),

-- Aggregate conditions per study
conditions_agg AS (
    SELECT 
        c.nct_id,
        array_agg(DISTINCT c.downcase_name) FILTER (WHERE c.downcase_name IS NOT NULL AND c.downcase_name != '') AS conditions,
        string_agg(DISTINCT c.downcase_name, ' ') FILTER (WHERE c.downcase_name IS NOT NULL AND c.downcase_name != '') AS conditions_text
    FROM ctgov_conditions c
    INNER JOIN nct_batch b ON c.nct_id = b.nct_id
    GROUP BY c.nct_id
),

-- Aggregate MeSH conditions
mesh_conditions_agg AS (
    SELECT 
        bc.nct_id,
        array_agg(DISTINCT bc.downcase_mesh_term) FILTER (WHERE bc.downcase_mesh_term IS NOT NULL AND bc.downcase_mesh_term != '') AS mesh_conditions,
        string_agg(DISTINCT bc.downcase_mesh_term, ' ') FILTER (WHERE bc.downcase_mesh_term IS NOT NULL AND bc.downcase_mesh_term != '') AS mesh_conditions_text
    FROM ctgov_browse_conditions bc
    INNER JOIN nct_batch b ON bc.nct_id = b.nct_id
    GROUP BY bc.nct_id
),

-- Aggregate interventions
interventions_agg AS (
    SELECT 
        i.nct_id,
        array_agg(DISTINCT lower(i.name)) FILTER (WHERE i.name IS NOT NULL AND i.name != '') AS interventions,
        string_agg(DISTINCT lower(i.name), ' ') FILTER (WHERE i.name IS NOT NULL AND i.name != '') AS interventions_text,
        array_agg(DISTINCT lower(i.intervention_type)) FILTER (WHERE i.intervention_type IS NOT NULL AND i.intervention_type != '') AS intervention_types
    FROM ctgov_interventions i
    INNER JOIN nct_batch b ON i.nct_id = b.nct_id
    GROUP BY i.nct_id
),

-- Aggregate MeSH interventions
mesh_interventions_agg AS (
    SELECT 
        bi.nct_id,
        array_agg(DISTINCT bi.downcase_mesh_term) FILTER (WHERE bi.downcase_mesh_term IS NOT NULL AND bi.downcase_mesh_term != '') AS mesh_interventions,
        string_agg(DISTINCT bi.downcase_mesh_term, ' ') FILTER (WHERE bi.downcase_mesh_term IS NOT NULL AND bi.downcase_mesh_term != '') AS mesh_interventions_text
    FROM ctgov_browse_interventions bi
    INNER JOIN nct_batch b ON bi.nct_id = b.nct_id
    GROUP BY bi.nct_id
),

-- Aggregate sponsors
sponsors_agg AS (
    SELECT 
        sp.nct_id,
        array_agg(DISTINCT lower(sp.name)) FILTER (WHERE sp.name IS NOT NULL AND sp.name != '') AS sponsors,
        string_agg(DISTINCT lower(sp.name), ' ') FILTER (WHERE sp.name IS NOT NULL AND sp.name != '') AS sponsors_text,
        MAX(CASE WHEN sp.lead_or_collaborator = 'lead' THEN lower(sp.name) END) AS lead_sponsor,
        array_agg(DISTINCT lower(sp.name)) FILTER (WHERE sp.lead_or_collaborator = 'collaborator' AND sp.name IS NOT NULL) AS collaborators
    FROM ctgov_sponsors sp
    INNER JOIN nct_batch b ON sp.nct_id = b.nct_id
    GROUP BY sp.nct_id
),

-- Aggregate countries
countries_agg AS (
    SELECT 
        co.nct_id,
        array_agg(DISTINCT co.name) FILTER (WHERE co.name IS NOT NULL AND co.name != '') AS countries
    FROM ctgov_countries co
    INNER JOIN nct_batch b ON co.nct_id = b.nct_id
    GROUP BY co.nct_id
),

-- Get brief summary for full-text
summaries AS (
    SELECT bs.nct_id, bs.description
    FROM ctgov_brief_summaries bs
    INNER JOIN nct_batch b ON bs.nct_id = b.nct_id
),

-- Get detailed description
details AS (
    SELECT dd.nct_id, dd.description
    FROM ctgov_detailed_descriptions dd
    INNER JOIN nct_batch b ON dd.nct_id = b.nct_id
)

SELECT 
    s.nct_id,
    s.brief_title,
    s.official_title,
    s.overall_status,
    s.phase,
    s.study_type,
    s.enrollment,
    s.enrollment_type,
    s.start_date,
    s.completion_date,
    s.primary_completion_date,
    s.results_first_submitted_date,
    s.has_dmc,
    s.is_fda_regulated_drug,
    s.is_fda_regulated_device,
    
    -- Arrays
    COALESCE(c.conditions, ARRAY[]::text[]) AS conditions,
    COALESCE(mc.mesh_conditions, ARRAY[]::text[]) AS mesh_conditions,
    COALESCE(i.interventions, ARRAY[]::text[]) AS interventions,
    COALESCE(mi.mesh_interventions, ARRAY[]::text[]) AS mesh_interventions,
    COALESCE(i.intervention_types, ARRAY[]::text[]) AS intervention_types,
    COALESCE(sp.sponsors, ARRAY[]::text[]) AS sponsors,
    sp.lead_sponsor,
    COALESCE(sp.collaborators, ARRAY[]::text[]) AS collaborators,
    COALESCE(cou.countries, ARRAY[]::text[]) AS countries,
    
    -- Normalized text for trigram
    lower(unaccent(COALESCE(c.conditions_text, ''))) AS conditions_norm,
    lower(unaccent(COALESCE(mc.mesh_conditions_text, ''))) AS mesh_conditions_norm,
    lower(unaccent(COALESCE(i.interventions_text, ''))) AS interventions_text,
    lower(unaccent(COALESCE(mi.mesh_interventions_text, ''))) AS mesh_interventions_text,
    lower(unaccent(COALESCE(sp.sponsors_text, ''))) AS sponsors_text,
    
    -- Full-text source (will compute tsvector on insert)
    COALESCE(s.brief_title, '') || ' ' || 
    COALESCE(s.official_title, '') || ' ' ||
    COALESCE(sum.description, '') || ' ' ||
    COALESCE(det.description, '') AS title_description_text,
    
    COALESCE(c.conditions_text, '') || ' ' ||
    COALESCE(mc.mesh_conditions_text, '') || ' ' ||
    COALESCE(i.interventions_text, '') || ' ' ||
    COALESCE(mi.mesh_interventions_text, '') AS terms_text

FROM ctgov_studies s
INNER JOIN nct_batch b ON s.nct_id = b.nct_id
LEFT JOIN conditions_agg c ON s.nct_id = c.nct_id
LEFT JOIN mesh_conditions_agg mc ON s.nct_id = mc.nct_id
LEFT JOIN interventions_agg i ON s.nct_id = i.nct_id
LEFT JOIN mesh_interventions_agg mi ON s.nct_id = mi.nct_id
LEFT JOIN sponsors_agg sp ON s.nct_id = sp.nct_id
LEFT JOIN countries_agg cou ON s.nct_id = cou.nct_id
LEFT JOIN summaries sum ON s.nct_id = sum.nct_id
LEFT JOIN details det ON s.nct_id = det.nct_id
"""

INSERT_QUERY = """
INSERT INTO public.rag_study_search (
    nct_id,
    brief_title,
    official_title,
    overall_status,
    phase,
    study_type,
    enrollment,
    enrollment_type,
    start_date,
    start_date_parsed,
    completion_date,
    completion_date_parsed,
    primary_completion_date,
    primary_completion_date_parsed,
    results_first_submitted_date,
    has_dmc,
    is_fda_regulated_drug,
    is_fda_regulated_device,
    conditions,
    mesh_conditions,
    interventions,
    mesh_interventions,
    intervention_types,
    sponsors,
    lead_sponsor,
    collaborators,
    countries,
    conditions_norm,
    mesh_conditions_norm,
    interventions_norm,
    mesh_interventions_norm,
    sponsors_norm,
    title_description_tsv,
    terms_tsv
) VALUES %s
ON CONFLICT (nct_id) DO UPDATE SET
    brief_title = EXCLUDED.brief_title,
    official_title = EXCLUDED.official_title,
    overall_status = EXCLUDED.overall_status,
    phase = EXCLUDED.phase,
    study_type = EXCLUDED.study_type,
    enrollment = EXCLUDED.enrollment,
    enrollment_type = EXCLUDED.enrollment_type,
    start_date = EXCLUDED.start_date,
    start_date_parsed = EXCLUDED.start_date_parsed,
    completion_date = EXCLUDED.completion_date,
    completion_date_parsed = EXCLUDED.completion_date_parsed,
    primary_completion_date = EXCLUDED.primary_completion_date,
    primary_completion_date_parsed = EXCLUDED.primary_completion_date_parsed,
    results_first_submitted_date = EXCLUDED.results_first_submitted_date,
    has_dmc = EXCLUDED.has_dmc,
    is_fda_regulated_drug = EXCLUDED.is_fda_regulated_drug,
    is_fda_regulated_device = EXCLUDED.is_fda_regulated_device,
    conditions = EXCLUDED.conditions,
    mesh_conditions = EXCLUDED.mesh_conditions,
    interventions = EXCLUDED.interventions,
    mesh_interventions = EXCLUDED.mesh_interventions,
    intervention_types = EXCLUDED.intervention_types,
    sponsors = EXCLUDED.sponsors,
    lead_sponsor = EXCLUDED.lead_sponsor,
    collaborators = EXCLUDED.collaborators,
    countries = EXCLUDED.countries,
    conditions_norm = EXCLUDED.conditions_norm,
    mesh_conditions_norm = EXCLUDED.mesh_conditions_norm,
    interventions_norm = EXCLUDED.interventions_norm,
    mesh_interventions_norm = EXCLUDED.mesh_interventions_norm,
    sponsors_norm = EXCLUDED.sponsors_norm,
    title_description_tsv = EXCLUDED.title_description_tsv,
    terms_tsv = EXCLUDED.terms_tsv,
    updated_at = NOW()
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_parse_date(date_str: str | None) -> str | None:
    """Safely parse a date string, returning None if invalid."""
    if not date_str or not date_str.strip():
        return None
    try:
        # Try to parse - PostgreSQL will handle the actual parsing
        # We just need to check if it looks valid
        return date_str.strip()
    except Exception:
        return None


def check_prerequisites(cur) -> tuple[bool, list[str]]:
    """Check that required tables and extensions exist."""
    issues = []
    
    # Check for required extensions
    cur.execute("""
        SELECT extname FROM pg_extension 
        WHERE extname IN ('pg_trgm', 'unaccent')
    """)
    extensions = {row[0] for row in cur.fetchall()}
    
    if 'pg_trgm' not in extensions:
        issues.append("Extension 'pg_trgm' not installed")
    if 'unaccent' not in extensions:
        issues.append("Extension 'unaccent' not installed")
    
    # Check for required tables
    required_tables = [
        'ctgov_studies',
        'ctgov_conditions',
        'ctgov_browse_conditions',
        'ctgov_interventions',
        'ctgov_browse_interventions',
        'ctgov_sponsors',
        'ctgov_countries',
        'ctgov_brief_summaries',
        'ctgov_detailed_descriptions',
    ]
    
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = ANY(%s)
    """, (required_tables,))
    existing_tables = {row[0] for row in cur.fetchall()}
    
    missing_tables = set(required_tables) - existing_tables
    if missing_tables:
        issues.append(f"Missing required tables: {', '.join(sorted(missing_tables))}")
    
    return len(issues) == 0, issues


def get_all_nct_ids(cur) -> list[str]:
    """Get all NCT IDs from ctgov_studies."""
    cur.execute("SELECT nct_id FROM ctgov_studies ORDER BY nct_id")
    return [row[0] for row in cur.fetchall()]


def batch_list(lst: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def row_to_insert_tuple(row: dict) -> tuple:
    """Convert a row dict to a tuple for insertion."""
    # Parse dates safely
    def try_parse_date(val):
        if val is None:
            return None
        try:
            # Return as-is, let Postgres handle parsing
            return val
        except Exception:
            return None
    
    return (
        row['nct_id'],
        row['brief_title'],
        row['official_title'],
        row['overall_status'],
        row['phase'],
        row['study_type'],
        row['enrollment'],
        row['enrollment_type'],
        row['start_date'],
        try_parse_date(row['start_date']),  # start_date_parsed
        row['completion_date'],
        try_parse_date(row['completion_date']),  # completion_date_parsed
        row['primary_completion_date'],
        try_parse_date(row['primary_completion_date']),  # primary_completion_date_parsed
        row['results_first_submitted_date'],
        row['has_dmc'],
        row['is_fda_regulated_drug'],
        row['is_fda_regulated_device'],
        row['conditions'],
        row['mesh_conditions'],
        row['interventions'],
        row['mesh_interventions'],
        row['intervention_types'],
        row['sponsors'],
        row['lead_sponsor'],
        row['collaborators'],
        row['countries'],
        row['conditions_norm'],
        row['mesh_conditions_norm'],
        row['interventions_text'],  # -> interventions_norm
        row['mesh_interventions_text'],  # -> mesh_interventions_norm
        row['sponsors_text'],  # -> sponsors_norm
        row['title_description_text'],  # Will be converted to tsvector
        row['terms_text'],  # Will be converted to tsvector
    )


# =============================================================================
# MAIN OPERATIONS
# =============================================================================

def create_table(config: DatabaseConfig) -> None:
    """Create the enriched search table and indexes."""
    print("🔨 Creating enriched search table...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Ensure extensions exist
            print("🔧 Ensuring required extensions...")
            cur.execute(CREATE_EXTENSIONS_SQL)
            conn.commit()
            
            # Check prerequisites
            print("🔍 Checking prerequisites...")
            ok, issues = check_prerequisites(cur)
            if not ok:
                print("❌ Prerequisites not met:")
                for issue in issues:
                    print(f"   - {issue}")
                raise RuntimeError("Prerequisites check failed")
            print("✅ Prerequisites OK")
            
            # Create table
            print("📋 Creating table structure...")
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
            print("✅ Table created")
            
            # Create indexes (do this after population for better performance)
            print("📇 Creating indexes (this may take a while on large datasets)...")
            cur.execute(CREATE_INDEXES_SQL)
            conn.commit()
            print("✅ Indexes created")
    
    print("✅ Table and indexes created successfully!")
    print("   Run 'python generate_ctgov_enriched_search.py populate' to load data")


def populate_table(config: DatabaseConfig, batch_size: int = 1000) -> None:
    """Populate the enriched search table with data."""
    print(f"📊 Populating enriched search table (batch size: {batch_size})...")
    
    with get_connection(config) as conn:
        # Use RealDictCursor for easier row handling
        from psycopg2.extras import RealDictCursor
        
        with conn.cursor() as cur:
            # Check table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (TABLE_NAME,))
            if not cur.fetchone()[0]:
                print(f"❌ Table '{TABLE_NAME}' does not exist. Run 'create' first.")
                return
            
            # Get all NCT IDs
            print("🔍 Getting study list...")
            nct_ids = get_all_nct_ids(cur)
            total_studies = len(nct_ids)
            print(f"📊 Found {total_studies:,} studies to process")
            
            if total_studies == 0:
                print("⚠️  No studies found in ctgov_studies")
                return
        
        # Process in batches using a dict cursor
        with conn.cursor(cursor_factory=RealDictCursor) as dict_cur:
            processed = 0
            errors = 0
            
            with tqdm(total=total_studies, unit=" studies", desc="Populating") as pbar:
                for batch_nct_ids in batch_list(nct_ids, batch_size):
                    try:
                        # Fetch batch data
                        dict_cur.execute(BATCH_SELECT_QUERY, {'nct_ids': batch_nct_ids})
                        rows = dict_cur.fetchall()
                        
                        if not rows:
                            pbar.update(len(batch_nct_ids))
                            continue
                        
                        # Prepare insert values
                        # We need to handle the tsvector conversion in the INSERT
                        insert_values = []
                        for row in rows:
                            insert_values.append((
                                row['nct_id'],
                                row['brief_title'],
                                row['official_title'],
                                row['overall_status'],
                                row['phase'],
                                row['study_type'],
                                row['enrollment'],
                                row['enrollment_type'],
                                row['start_date'],
                                row['start_date'],  # Will be cast to date
                                row['completion_date'],
                                row['completion_date'],  # Will be cast to date
                                row['primary_completion_date'],
                                row['primary_completion_date'],  # Will be cast to date
                                row['results_first_submitted_date'],
                                row['has_dmc'],
                                row['is_fda_regulated_drug'],
                                row['is_fda_regulated_device'],
                                row['conditions'],
                                row['mesh_conditions'],
                                row['interventions'],
                                row['mesh_interventions'],
                                row['intervention_types'],
                                row['sponsors'],
                                row['lead_sponsor'],
                                row['collaborators'],
                                row['countries'],
                                row['conditions_norm'],
                                row['mesh_conditions_norm'],
                                row['interventions_text'],
                                row['mesh_interventions_text'],
                                row['sponsors_text'],
                                row['title_description_text'],
                                row['terms_text'],
                            ))
                        
                        # Use execute_values for efficient batch insert
                        # Template handles date casting and tsvector conversion
                        template = """(
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::date, %s, %s::date, %s, %s::date, %s,
                            %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            to_tsvector('english', %s),
                            to_tsvector('english', %s)
                        )"""
                        
                        execute_values(
                            dict_cur,
                            INSERT_QUERY,
                            insert_values,
                            template=template,
                            page_size=batch_size
                        )
                        conn.commit()
                        
                        processed += len(rows)
                        pbar.update(len(batch_nct_ids))
                        
                    except Exception as e:
                        errors += 1
                        conn.rollback()
                        if errors <= 3:
                            print(f"\n⚠️  Error processing batch: {e}")
                        elif errors == 4:
                            print("\n⚠️  Suppressing further error messages...")
                        pbar.update(len(batch_nct_ids))
            
            # Final stats
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                final_count = cur.fetchone()[0]
            
            print(f"\n✅ Population complete!")
            print(f"   Processed: {processed:,} studies")
            print(f"   In table: {final_count:,} studies")
            if errors > 0:
                print(f"   Errors: {errors} batches")


def refresh_table(config: DatabaseConfig, batch_size: int = 1000) -> None:
    """Truncate and repopulate the table."""
    print("🔄 Refreshing enriched search table...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Check table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (TABLE_NAME,))
            if not cur.fetchone()[0]:
                print(f"❌ Table '{TABLE_NAME}' does not exist. Run 'create' first.")
                return
            
            # Truncate
            print("🗑️  Truncating existing data...")
            cur.execute(f"TRUNCATE TABLE {TABLE_NAME}")
            conn.commit()
            print("✅ Table truncated")
    
    # Repopulate
    populate_table(config, batch_size)


def verify_table(config: DatabaseConfig) -> None:
    """Verify the state of the enriched search table."""
    print("🔍 Verifying enriched search table...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Check table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (TABLE_NAME,))
            exists = cur.fetchone()[0]
            
            if not exists:
                print(f"❌ Table '{TABLE_NAME}' does not exist")
                return
            
            print(f"✅ Table '{TABLE_NAME}' exists")
            
            # Row count
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            row_count = cur.fetchone()[0]
            print(f"   Rows: {row_count:,}")
            
            # Compare with source
            cur.execute("SELECT COUNT(*) FROM ctgov_studies")
            source_count = cur.fetchone()[0]
            print(f"   Source studies: {source_count:,}")
            
            if row_count < source_count:
                missing = source_count - row_count
                pct = missing / source_count * 100 if source_count > 0 else 0
                print(f"   ⚠️  Missing {missing:,} studies ({pct:.1f}%)")
            elif row_count == source_count:
                print("   ✅ All studies indexed")
            
            # Check indexes
            cur.execute("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = %s
                ORDER BY indexname
            """, (TABLE_NAME,))
            indexes = cur.fetchall()
            print(f"\n   Indexes ({len(indexes)}):")
            for idx_name, idx_def in indexes:
                idx_type = "GIN" if "gin" in idx_def.lower() else "BTREE"
                print(f"     - {idx_name} ({idx_type})")
            
            if row_count == 0:
                print("\n   ⚠️  Table is empty, skipping data coverage stats")
                return
            
            # Sample data quality
            cur.execute(f"""
                SELECT 
                    COUNT(*) FILTER (WHERE array_length(conditions, 1) > 0) AS with_conditions,
                    COUNT(*) FILTER (WHERE array_length(interventions, 1) > 0) AS with_interventions,
                    COUNT(*) FILTER (WHERE array_length(mesh_conditions, 1) > 0) AS with_mesh_cond,
                    COUNT(*) FILTER (WHERE array_length(mesh_interventions, 1) > 0) AS with_mesh_intr,
                    COUNT(*) FILTER (WHERE lead_sponsor IS NOT NULL) AS with_sponsor,
                    COUNT(*) FILTER (WHERE title_description_tsv IS NOT NULL) AS with_tsv,
                    COUNT(*) FILTER (WHERE start_date_parsed IS NOT NULL) AS with_start_date,
                    COUNT(*) FILTER (WHERE overall_status IS NOT NULL) AS with_status
                FROM {TABLE_NAME}
            """)
            stats = cur.fetchone()
            
            print(f"\n   Data coverage:")
            labels = [
                "With conditions", "With interventions", "With MeSH conditions",
                "With MeSH interventions", "With lead sponsor", "With text search vector",
                "With parsed start date", "With status"
            ]
            for i, label in enumerate(labels):
                pct = stats[i] / row_count * 100 if row_count > 0 else 0
                print(f"     - {label}: {stats[i]:,} ({pct:.1f}%)")
            
            # Status distribution
            cur.execute(f"""
                SELECT overall_status, COUNT(*) 
                FROM {TABLE_NAME} 
                WHERE overall_status IS NOT NULL
                GROUP BY overall_status 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            status_dist = cur.fetchall()
            print(f"\n   Status distribution (top 10):")
            for status, count in status_dist:
                print(f"     - {status}: {count:,}")
            
            # Phase distribution
            cur.execute(f"""
                SELECT phase, COUNT(*) 
                FROM {TABLE_NAME} 
                WHERE phase IS NOT NULL
                GROUP BY phase 
                ORDER BY COUNT(*) DESC
            """)
            phase_dist = cur.fetchall()
            print(f"\n   Phase distribution:")
            for phase, count in phase_dist:
                print(f"     - {phase}: {count:,}")


def drop_table(config: DatabaseConfig) -> None:
    """Drop the enriched search table."""
    print(f"🗑️  Dropping table '{TABLE_NAME}'...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
            conn.commit()
    
    print("✅ Table dropped")


def test_search(config: DatabaseConfig) -> None:
    """Run some test queries to verify search functionality."""
    print("🧪 Testing search functionality...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Check if table has data
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            count = cur.fetchone()[0]
            if count == 0:
                print("⚠️  Table is empty, cannot run tests")
                return
            
            # Test 1: Trigram similarity search for condition
            print("\n--- Test 1: Condition trigram search ('breast cancer') ---")
            cur.execute(f"""
                SELECT nct_id, brief_title, 
                       similarity(conditions_norm, 'breast cancer') AS score,
                       conditions[1:3] AS top_conditions
                FROM {TABLE_NAME}
                WHERE conditions_norm %% 'breast cancer'
                ORDER BY score DESC
                LIMIT 5
            """)
            results = cur.fetchall()
            if results:
                for row in results:
                    title = row[1][:60] + "..." if row[1] and len(row[1]) > 60 else row[1]
                    print(f"  {row[0]} (score: {row[2]:.3f}): {title}")
            else:
                print("  No results found")
            
            # Test 2: Full-text search in descriptions
            print("\n--- Test 2: Full-text search ('immunotherapy checkpoint') ---")
            cur.execute(f"""
                SELECT nct_id, brief_title,
                       ts_rank(title_description_tsv, to_tsquery('english', 'immunotherapy & checkpoint')) AS score
                FROM {TABLE_NAME}
                WHERE title_description_tsv @@ to_tsquery('english', 'immunotherapy & checkpoint')
                ORDER BY score DESC
                LIMIT 5
            """)
            results = cur.fetchall()
            if results:
                for row in results:
                    title = row[1][:60] + "..." if row[1] and len(row[1]) > 60 else row[1]
                    print(f"  {row[0]} (score: {row[2]:.4f}): {title}")
            else:
                print("  No results found")
            
            # Test 3: Filter by status and phase
            print("\n--- Test 3: Recruiting Phase 3 trials ---")
            cur.execute(f"""
                SELECT nct_id, brief_title, phase, overall_status, enrollment
                FROM {TABLE_NAME}
                WHERE phase = 'Phase 3'
                  AND overall_status = 'Recruiting'
                ORDER BY enrollment DESC NULLS LAST
                LIMIT 5
            """)
            results = cur.fetchall()
            if results:
                for row in results:
                    title = row[1][:50] + "..." if row[1] and len(row[1]) > 50 else row[1]
                    print(f"  {row[0]} ({row[2]}, n={row[4]}): {title}")
            else:
                print("  No results found")
            
            # Test 4: Date range filter
            print("\n--- Test 4: Studies started in 2023 ---")
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM {TABLE_NAME}
                WHERE start_date_parsed >= '2023-01-01' 
                  AND start_date_parsed < '2024-01-01'
            """)
            count = cur.fetchone()[0]
            print(f"  Found {count:,} studies started in 2023")
            
            # Test 5: Intervention search
            print("\n--- Test 5: Intervention search ('pembrolizumab') ---")
            cur.execute(f"""
                SELECT nct_id, brief_title, 
                       similarity(interventions_norm, 'pembrolizumab') AS score,
                       interventions[1:3] AS top_interventions
                FROM {TABLE_NAME}
                WHERE interventions_norm %% 'pembrolizumab'
                ORDER BY score DESC
                LIMIT 5
            """)
            results = cur.fetchall()
            if results:
                for row in results:
                    title = row[1][:50] + "..." if row[1] and len(row[1]) > 50 else row[1]
                    print(f"  {row[0]} (score: {row[2]:.3f}): {title}")
            else:
                print("  No results found")
            
            # Test 6: Combined search
            print("\n--- Test 6: Combined condition + intervention + filter ---")
            cur.execute(f"""
                SELECT nct_id, brief_title, overall_status,
                       similarity(conditions_norm, 'lung cancer') AS cond_score,
                       similarity(interventions_norm, 'nivolumab') AS intr_score
                FROM {TABLE_NAME}
                WHERE conditions_norm %% 'lung cancer'
                  AND interventions_norm %% 'nivolumab'
                  AND overall_status IN ('Recruiting', 'Active, not recruiting', 'Completed')
                ORDER BY (similarity(conditions_norm, 'lung cancer') + similarity(interventions_norm, 'nivolumab')) DESC
                LIMIT 5
            """)
            results = cur.fetchall()
            if results:
                for row in results:
                    title = row[1][:45] + "..." if row[1] and len(row[1]) > 45 else row[1]
                    print(f"  {row[0]} ({row[2]}, c:{row[3]:.2f}, i:{row[4]:.2f}): {title}")
            else:
                print("  No results found (try different terms)")
    
    print("\n✅ Search tests complete!")


# =============================================================================
# CLI
# =============================================================================

def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Optional batch size for populate/refresh
    batch_size = 1000
    if len(sys.argv) >= 3 and command in ('populate', 'refresh', 'full'):
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print(f"❌ Invalid batch size: {sys.argv[2]}")
            sys.exit(1)
    
    try:
        if command == 'create':
            create_table(DEFAULT_CONFIG)
        
        elif command == 'populate':
            populate_table(DEFAULT_CONFIG, batch_size)
        
        elif command == 'refresh':
            refresh_table(DEFAULT_CONFIG, batch_size)
        
        elif command == 'verify':
            verify_table(DEFAULT_CONFIG)
        
        elif command == 'drop':
            confirm = input(f"Are you sure you want to drop '{TABLE_NAME}'? (yes/no): ")
            if confirm.lower() == 'yes':
                drop_table(DEFAULT_CONFIG)
            else:
                print("Cancelled.")
        
        elif command == 'test':
            test_search(DEFAULT_CONFIG)
        
        elif command == 'full':
            # Full setup: create + populate + verify + test
            create_table(DEFAULT_CONFIG)
            populate_table(DEFAULT_CONFIG, batch_size)
            verify_table(DEFAULT_CONFIG)
            test_search(DEFAULT_CONFIG)
        
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()