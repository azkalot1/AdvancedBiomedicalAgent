\echo === Building RAG study JSON objects ===

\set ON_ERROR_STOP on
SET client_min_messages = warning;

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;  -- optional, improves matching of diacritics
SET maintenance_work_mem = '1GB';
SET statement_timeout = '0';      -- disable timeout for long build
SET lock_timeout = '0';
SET search_path = public;

\echo === Creating helper indexes for RAG study JSON ===
\set ON_ERROR_STOP on
SET client_min_messages = warning;
SET maintenance_work_mem = '1GB';
SET statement_timeout = '0';
SET search_path = public;
-- ---------- per-table nct_id lookups ----------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_conditions_nct          ON ctgov_conditions (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_interventions_nct       ON ctgov_interventions (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_sponsors_nct            ON ctgov_sponsors (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_facilities_nct          ON ctgov_facilities (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_designs_nct             ON ctgov_designs (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_eligibilities_nct       ON ctgov_eligibilities (nct_id);
-- ---------- result group plumbing ----------
-- Used to derive section/suffix and join to measurements/counts/events
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_result_groups_nct_id_code
  ON ctgov_result_groups (nct_id, id, ctgov_group_code);
-- ---------- baseline tables (join via result_group_id) ----------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_baseline_counts_rg
  ON ctgov_baseline_counts (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_baseline_counts_nct
  ON ctgov_baseline_counts (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_baseline_meas_rg
  ON ctgov_baseline_measurements (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_baseline_meas_nct
  ON ctgov_baseline_measurements (nct_id);
-- ---------- outcomes ----------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcomes_nct
  ON ctgov_outcomes (nct_id);
-- measurements/counts join by outcome_id, then map to groups by result_group_id
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_meas_outcome
  ON ctgov_outcome_measurements (outcome_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_meas_rg
  ON ctgov_outcome_measurements (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_meas_nct
  ON ctgov_outcome_measurements (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_counts_outcome
  ON ctgov_outcome_counts (outcome_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_counts_rg
  ON ctgov_outcome_counts (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_counts_nct
  ON ctgov_outcome_counts (nct_id);
-- analyses + which groups were compared
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_analyses_outcome
  ON ctgov_outcome_analyses (outcome_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_analyses_nct
  ON ctgov_outcome_analyses (nct_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_analysis_groups_oa
  ON ctgov_outcome_analysis_groups (outcome_analysis_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_analysis_groups_rg
  ON ctgov_outcome_analysis_groups (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_outcome_analysis_groups_nct
  ON ctgov_outcome_analysis_groups (nct_id);
-- --------- adverse events ----------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_reported_events_rg
  ON ctgov_reported_events (result_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_reported_events_nct
  ON ctgov_reported_events (nct_id);
-- ---------- registry group -> interventions ----------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_dgi_dg
  ON ctgov_design_group_interventions (design_group_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctgov_dgi_intervention
  ON ctgov_design_group_interventions (intervention_id);
\echo === ANALYZE to update stats ===
-- Only analyze our tables, not system tables
ANALYZE ctgov_conditions;
ANALYZE ctgov_interventions;
ANALYZE ctgov_sponsors;
ANALYZE ctgov_facilities;
ANALYZE ctgov_designs;
ANALYZE ctgov_eligibilities;
\echo === Done: indexes created (concurrently) ===
-- (paste your helper view defs here: rag_result_groups, rag_canonical_groups, rag_registry_groups, rag_design_groups_ix)
drop materialized view if exists rag_study_json cascade;
-- 1a) Decompose result group codes, pick a canonical title per suffix
create or replace view rag_result_groups as
select
  rg.nct_id,
  rg.id                          as result_group_id,
  rg.ctgov_group_code,
  substr(rg.ctgov_group_code, 1, 2)  as result_section,   -- BG / OG / EG / FG
  substr(rg.ctgov_group_code, 3)::int as group_idx,       -- 0,1,2 ...
  coalesce(nullif(trim(rg.title), ''), '[No title]')      as title,
  rg.description
from public.ctgov_result_groups rg;
create or replace view rag_canonical_groups as
select distinct on (nct_id, group_idx)
  nct_id,
  group_idx,
  title        as canonical_title,
  description  as canonical_description
from rag_result_groups
order by nct_id, group_idx, result_section;
create or replace view rag_registry_groups as
select
  dg.nct_id,
  row_number() over (partition by dg.nct_id order by dg.id) - 1 as group_idx, -- synthetic
  coalesce(nullif(trim(dg.title), ''), '[Registry group]')      as canonical_title,
  dg.description                                               as canonical_description
from public.ctgov_design_groups dg;
create or replace view rag_design_groups_ix as
select
  dg.nct_id,
  dg.id as design_group_id,
  row_number() over (partition by dg.nct_id order by dg.id) - 1 as group_idx,
  dg.title,
  dg.description
from public.ctgov_design_groups dg;
create materialized view rag_study_json as
select
  s.nct_id,
  jsonb_build_object(
    -- 2a) Core metadata
    'metadata', jsonb_build_object(
      'brief_title', s.brief_title,
      'official_title', s.official_title,
      'overall_status', s.overall_status,
      'study_type', s.study_type,
      'phase', s.phase,
      'enrollment', s.enrollment,
      'enrollment_type', s.enrollment_type,
      'start_date', s.start_date,
      'primary_completion_date', s.primary_completion_date,
      'completion_date', s.completion_date,
      'why_stopped', s.why_stopped,
      'has_results', cv.were_results_reported
    ),
    -- 2b) Design / purpose / masking
    'design', (
      select jsonb_build_object(
        'allocation', d.allocation,
        'intervention_model', d.intervention_model,
        'primary_purpose', d.primary_purpose,
        'masking', d.masking,
        'masking_description', d.masking_description,
        'number_of_arms', s.number_of_arms,
        'number_of_groups', s.number_of_groups
      )
      from public.ctgov_designs d
      where d.nct_id = s.nct_id
      limit 1
    ),
    -- 2c) Population / eligibility
    'eligibility', (
      select jsonb_build_object(
        'gender', e.gender,
        'minimum_age', e.minimum_age,
        'maximum_age', e.maximum_age,
        'healthy_volunteers', e.healthy_volunteers,
        'criteria', e.criteria,
        'adult', e.adult,
        'child', e.child,
        'older_adult', e.older_adult
      )
      from public.ctgov_eligibilities e
      where e.nct_id = s.nct_id
      limit 1
    ),
    -- 2d) Labels for filtering
    'labels', jsonb_build_object(
      'conditions', (
        select jsonb_agg(distinct c.name) filter (where c.name is not null)
        from public.ctgov_conditions c
        where c.nct_id = s.nct_id
      ),
      'interventions', (
        select jsonb_agg(distinct i.name) filter (where i.name is not null)
        from public.ctgov_interventions i
        where i.nct_id = s.nct_id
      ),
      'sponsors', (
        select jsonb_agg(distinct sp.name) filter (where sp.name is not null)
        from public.ctgov_sponsors sp
        where sp.nct_id = s.nct_id and sp.lead_or_collaborator = 'lead'
      ),
      'countries', (
        select jsonb_agg(distinct f.country) filter (where f.country is not null)
        from public.ctgov_facilities f
        where f.nct_id = s.nct_id
      )
    ),
    -- 2e) Registry groups + assigned interventions (using rag_design_groups_ix; no window fn in WHERE)
    'registry_groups', (
      select jsonb_agg(
        jsonb_build_object(
          'group_idx', rg.group_idx,
          'title', rg.canonical_title,
          'description', rg.canonical_description,
          'interventions', (
            select jsonb_agg(distinct jsonb_build_object(
                     'name', i.name,
                     'type', i.intervention_type,
                     'description', i.description
                   ))
            from rag_design_groups_ix x
            join public.ctgov_design_group_interventions dgi
              on dgi.design_group_id = x.design_group_id
            join public.ctgov_interventions i
              on i.id = dgi.intervention_id
            where x.nct_id = s.nct_id
              and x.group_idx = rg.group_idx
          )
        )
      )
      from rag_registry_groups rg
      where rg.nct_id = s.nct_id
    ),
    -- 2f) Results groups with baselines
    'results_groups', (
      select jsonb_agg(
        jsonb_build_object(
          'group_idx', cg.group_idx,
          'title', cg.canonical_title,
          'description', cg.canonical_description,
          'baselines', (
            select jsonb_agg(
              jsonb_build_object(
                'units', bc.units,
                'scope', bc.scope,
                'count', bc.count
              )
            )
            from public.ctgov_baseline_counts bc
            join rag_result_groups rgr on rgr.result_group_id = bc.result_group_id
            where bc.nct_id = s.nct_id and rgr.group_idx = cg.group_idx
          ),
          'baseline_measurements', (
            select jsonb_agg(
              jsonb_build_object(
                'title', bm.title,
                'units', bm.units,
                'param_type', bm.param_type,
                'value', bm.param_value,
                'dispersion_type', bm.dispersion_type,
                'dispersion', bm.dispersion_value
              )
            )
            from public.ctgov_baseline_measurements bm
            join rag_result_groups rgr on rgr.result_group_id = bm.result_group_id
            where bm.nct_id = s.nct_id and rgr.group_idx = cg.group_idx
          )
        )
      )
      from rag_canonical_groups cg
      where cg.nct_id = s.nct_id
    ),
    -- 2g) Outcomes with per-group values and analyses
    'outcomes', (
      select jsonb_agg(
        jsonb_build_object(
          'id', o.id,
          'type', o.outcome_type,
          'title', o.title,
          'time_frame', o.time_frame,
          'units', o.units,
          'param_type', o.param_type,
          'measures', (
            select jsonb_agg(
              jsonb_build_object(
                'group_idx', rgr.group_idx,
                'group_title', cg.canonical_title,
                'classification', om.classification,
                'units', coalesce(om.units, o.units),
                'param_type', coalesce(om.param_type, o.param_type),
                'value', om.param_value,
                'value_num', om.param_value_num,
                'dispersion_type', om.dispersion_type,
                'dispersion', om.dispersion_value,
                'lower', om.dispersion_lower_limit,
                'upper', om.dispersion_upper_limit
              )
            )
            from public.ctgov_outcome_measurements om
            join rag_result_groups rgr on rgr.result_group_id = om.result_group_id
            left join rag_canonical_groups cg
              on cg.nct_id = om.nct_id and cg.group_idx = rgr.group_idx
            where om.outcome_id = o.id
          ),
          'counts', (
            select jsonb_agg(
              jsonb_build_object(
                'group_idx', rgr.group_idx,
                'group_title', cg.canonical_title,
                'scope', oc.scope,
                'units', oc.units,
                'count', oc.count
              )
            )
            from public.ctgov_outcome_counts oc
            join rag_result_groups rgr on rgr.result_group_id = oc.result_group_id
            left join rag_canonical_groups cg
              on cg.nct_id = oc.nct_id and cg.group_idx = rgr.group_idx
            where oc.outcome_id = o.id
          ),
          'analyses', (
            select jsonb_agg(
              jsonb_build_object(
                'method', oa.method,
                'p_value', oa.p_value,
                'ci_percent', oa.ci_percent,
                'ci_lower', oa.ci_lower_limit,
                'ci_upper', oa.ci_upper_limit,
                'comparison_groups', (
                  select jsonb_agg(
                    jsonb_build_object(
                      'group_idx', rgr.group_idx,
                      'group_title', cg.canonical_title
                    )
                  )
                  from public.ctgov_outcome_analysis_groups oag
                  join rag_result_groups rgr on rgr.result_group_id = oag.result_group_id
                  left join rag_canonical_groups cg
                    on cg.nct_id = oag.nct_id and cg.group_idx = rgr.group_idx
                  where oag.outcome_analysis_id = oa.id
                ),
                'description', coalesce(oa.groups_description, oa.method_description)
              )
            )
            from public.ctgov_outcome_analyses oa
            where oa.outcome_id = o.id
          )
        )
        order by o.outcome_type, o.id
      )
      from public.ctgov_outcomes o
      where o.nct_id = s.nct_id
    ),
    -- 2h) Adverse events
    'adverse_events', (
      select jsonb_agg(
        jsonb_build_object(
          'event_type', re.event_type,
          'term', re.adverse_event_term,
          'organ_system', re.organ_system,
          'group_idx', rgr.group_idx,
          'group_title', cg.canonical_title,
          'subjects_affected', re.subjects_affected,
          'subjects_at_risk', re.subjects_at_risk,
          'rate',
            case when re.subjects_at_risk > 0
                 then round(100.0 * re.subjects_affected::numeric / re.subjects_at_risk, 2)
                 else null end
        )
      )
      from public.ctgov_reported_events re
      join rag_result_groups rgr on rgr.result_group_id = re.result_group_id
      left join rag_canonical_groups cg
        on cg.nct_id = re.nct_id and cg.group_idx = rgr.group_idx
      where re.nct_id = s.nct_id
    )
  )::jsonb as study_json
from public.ctgov_studies s
left join public.ctgov_calculated_values cv on cv.nct_id = s.nct_id
with no data;
create table if not exists rag_study_corpus (
  nct_id     text primary key,
  study_json jsonb not null,
  updated_at timestamptz not null default now()
);
CREATE OR REPLACE FUNCTION public.build_study_json(p_nct_id text)
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
/*
  Returns a single JSON document for the given NCT ID.
  Assumes helper views exist:
    - rag_result_groups
    - rag_canonical_groups
    - rag_registry_groups
    - rag_design_groups_ix
*/
SELECT
  jsonb_build_object(

    -- 1) Core metadata
    'metadata', jsonb_build_object(
      'brief_title', s.brief_title,
      'official_title', s.official_title,
      'overall_status', s.overall_status,
      'study_type', s.study_type,
      'phase', s.phase,
      'enrollment', s.enrollment,
      'enrollment_type', s.enrollment_type,
      'start_date', s.start_date,
      'primary_completion_date', s.primary_completion_date,
      'completion_date', s.completion_date,
      'why_stopped', s.why_stopped,
      'has_results', cv.were_results_reported
    ),
    -- 2) Design
    'design', (
      SELECT jsonb_build_object(
        'allocation', d.allocation,
        'intervention_model', d.intervention_model,
        'primary_purpose', d.primary_purpose,
        'masking', d.masking,
        'masking_description', d.masking_description,
        'number_of_arms', s.number_of_arms,
        'number_of_groups', s.number_of_groups
      )
      FROM public.ctgov_designs d
      WHERE d.nct_id = p_nct_id
      LIMIT 1
    ),
    -- 3) Eligibility
    'eligibility', (
      SELECT jsonb_build_object(
        'gender', e.gender,
        'minimum_age', e.minimum_age,
        'maximum_age', e.maximum_age,
        'healthy_volunteers', e.healthy_volunteers,
        'criteria', e.criteria,
        'adult', e.adult,
        'child', e.child,
        'older_adult', e.older_adult
      )
      FROM public.ctgov_eligibilities e
      WHERE e.nct_id = p_nct_id
      LIMIT 1
    ),
    -- 4) Labels (filters)
    'labels', jsonb_build_object(
      'conditions', (
        SELECT jsonb_agg(DISTINCT c.name) FILTER (WHERE c.name IS NOT NULL)
        FROM public.ctgov_conditions c
        WHERE c.nct_id = p_nct_id
      ),
      'interventions', (
        SELECT jsonb_agg(DISTINCT i.name) FILTER (WHERE i.name IS NOT NULL)
        FROM public.ctgov_interventions i
        WHERE i.nct_id = p_nct_id
      ),
      'sponsors', (
        SELECT jsonb_agg(DISTINCT sp.name) FILTER (WHERE sp.name IS NOT NULL)
        FROM public.ctgov_sponsors sp
        WHERE sp.nct_id = p_nct_id
          AND sp.lead_or_collaborator = 'lead'
      ),
      'countries', (
        SELECT jsonb_agg(DISTINCT f.country) FILTER (WHERE f.country IS NOT NULL)
        FROM public.ctgov_facilities f
        WHERE f.nct_id = p_nct_id
      )
    ),
    -- 5) Registry groups + assigned interventions
    'registry_groups', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'group_idx', rg.group_idx,
          'title', rg.canonical_title,
          'description', rg.canonical_description,
          'interventions', (
            SELECT jsonb_agg(DISTINCT jsonb_build_object(
                     'name', i.name,
                     'type', i.intervention_type,
                     'description', i.description
                   ))
            FROM public.rag_design_groups_ix x
            JOIN public.ctgov_design_group_interventions dgi
              ON dgi.design_group_id = x.design_group_id
            JOIN public.ctgov_interventions i
              ON i.id = dgi.intervention_id
            WHERE x.nct_id = p_nct_id
              AND x.group_idx = rg.group_idx
          )
        )
      )
      FROM public.rag_registry_groups rg
      WHERE rg.nct_id = p_nct_id
    ),
    -- 6) Results groups + baselines
    'results_groups', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'group_idx', cg.group_idx,
          'title', cg.canonical_title,
          'description', cg.canonical_description,
          'baselines', (
            SELECT jsonb_agg(
              jsonb_build_object(
                'units', bc.units,
                'scope', bc.scope,
                'count', bc.count
              )
            )
            FROM public.ctgov_baseline_counts bc
            JOIN public.rag_result_groups rgr
              ON rgr.result_group_id = bc.result_group_id
            WHERE bc.nct_id = p_nct_id
              AND rgr.group_idx = cg.group_idx
          ),
          'baseline_measurements', (
            SELECT jsonb_agg(
              jsonb_build_object(
                'title', bm.title,
                'units', bm.units,
                'param_type', bm.param_type,
                'value', bm.param_value,
                'dispersion_type', bm.dispersion_type,
                'dispersion', bm.dispersion_value
              )
            )
            FROM public.ctgov_baseline_measurements bm
            JOIN public.rag_result_groups rgr
              ON rgr.result_group_id = bm.result_group_id
            WHERE bm.nct_id = p_nct_id
              AND rgr.group_idx = cg.group_idx
          )
        )
      )
      FROM public.rag_canonical_groups cg
      WHERE cg.nct_id = p_nct_id
    ),
    -- 7) Outcomes (values, counts, analyses)
    'outcomes', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'id', o.id,
          'type', o.outcome_type,
          'title', o.title,
          'time_frame', o.time_frame,
          'units', o.units,
          'param_type', o.param_type,
          'measures', (
            SELECT jsonb_agg(
              jsonb_build_object(
                'group_idx', rgr.group_idx,
                'group_title', cg.canonical_title,
                'classification', om.classification,
                'units', COALESCE(om.units, o.units),
                'param_type', COALESCE(om.param_type, o.param_type),
                'value', om.param_value,
                'value_num', om.param_value_num,
                'dispersion_type', om.dispersion_type,
                'dispersion', om.dispersion_value,
                'lower', om.dispersion_lower_limit,
                'upper', om.dispersion_upper_limit
              )
            )
            FROM public.ctgov_outcome_measurements om
            JOIN public.rag_result_groups rgr
              ON rgr.result_group_id = om.result_group_id
            LEFT JOIN public.rag_canonical_groups cg
              ON cg.nct_id = om.nct_id
             AND cg.group_idx = rgr.group_idx
            WHERE om.outcome_id = o.id
              AND om.nct_id = p_nct_id
          ),
          'counts', (
            SELECT jsonb_agg(
              jsonb_build_object(
                'group_idx', rgr.group_idx,
                'group_title', cg.canonical_title,
                'scope', oc.scope,
                'units', oc.units,
                'count', oc.count
              )
            )
            FROM public.ctgov_outcome_counts oc
            JOIN public.rag_result_groups rgr
              ON rgr.result_group_id = oc.result_group_id
            LEFT JOIN public.rag_canonical_groups cg
              ON cg.nct_id = oc.nct_id
             AND cg.group_idx = rgr.group_idx
            WHERE oc.outcome_id = o.id
              AND oc.nct_id = p_nct_id
          ),
          'analyses', (
            SELECT jsonb_agg(
              jsonb_build_object(
                'method', oa.method,
                'p_value', oa.p_value,
                'ci_percent', oa.ci_percent,
                'ci_lower', oa.ci_lower_limit,
                'ci_upper', oa.ci_upper_limit,
                'comparison_groups', (
                  SELECT jsonb_agg(
                    jsonb_build_object(
                      'group_idx', rgr.group_idx,
                      'group_title', cg.canonical_title
                    )
                  )
                  FROM public.ctgov_outcome_analysis_groups oag
                  JOIN public.rag_result_groups rgr
                    ON rgr.result_group_id = oag.result_group_id
                  LEFT JOIN public.rag_canonical_groups cg
                    ON cg.nct_id = oag.nct_id
                   AND cg.group_idx = rgr.group_idx
                  WHERE oag.outcome_analysis_id = oa.id
                    AND oag.nct_id = p_nct_id
                ),
                'description', COALESCE(oa.groups_description, oa.method_description)
              )
            )
            FROM public.ctgov_outcome_analyses oa
            WHERE oa.outcome_id = o.id
              AND oa.nct_id = p_nct_id
          )
        )
        ORDER BY o.outcome_type, o.id
      )
      FROM public.ctgov_outcomes o
      WHERE o.nct_id = p_nct_id
    ),
    -- 8) Adverse events
    'adverse_events', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'event_type', re.event_type,
          'term', re.adverse_event_term,
          'organ_system', re.organ_system,
          'group_idx', rgr.group_idx,
          'group_title', cg.canonical_title,
          'subjects_affected', re.subjects_affected,
          'subjects_at_risk', re.subjects_at_risk,
          'rate',
            CASE
              WHEN re.subjects_at_risk > 0
              THEN round(100.0 * re.subjects_affected::numeric / re.subjects_at_risk, 2)
              ELSE NULL
            END
        )
      )
      FROM public.ctgov_reported_events re
      JOIN public.rag_result_groups rgr
        ON rgr.result_group_id = re.result_group_id
      LEFT JOIN public.rag_canonical_groups cg
        ON cg.nct_id = re.nct_id
       AND cg.group_idx = rgr.group_idx
      WHERE re.nct_id = p_nct_id
    )
  )::jsonb AS study_json
FROM public.ctgov_studies s
LEFT JOIN public.ctgov_calculated_values cv
  ON cv.nct_id = s.nct_id
WHERE s.nct_id = p_nct_id
LIMIT 1;
$$;


-- One-time create; safe to re-run with OR REPLACE
CREATE OR REPLACE PROCEDURE public.upsert_rag_study_corpus_all(p_k int)
LANGUAGE plpgsql
AS $$
DECLARE
  k int;
  done bigint;
BEGIN
  IF p_k < 1 THEN
    RAISE EXCEPTION 'p_k (number of buckets) must be >= 1';
  END IF;
  FOR k IN 0..p_k-1 LOOP
    -- each bucket processed separately (faster + visible progress)
    -- Note: Cannot use START TRANSACTION inside PL/pgSQL procedure
    WITH ids AS (
      SELECT s.nct_id
      FROM public.ctgov_studies s
      WHERE (abs(hashtextextended(s.nct_id, 0)) % p_k) = k
      -- Uncomment to skip rows already present:
      -- AND NOT EXISTS (SELECT 1 FROM public.rag_study_corpus c WHERE c.nct_id = s.nct_id)
    )
    INSERT INTO public.rag_study_corpus (nct_id, study_json)
    SELECT nct_id, public.build_study_json(nct_id)
    FROM ids
    ON CONFLICT (nct_id) DO UPDATE
      SET study_json = EXCLUDED.study_json,
          updated_at = now();
    -- progress for this bucket
    SELECT count(*) INTO done
    FROM public.rag_study_corpus c
    WHERE (abs(hashtextextended(c.nct_id, 0)) % p_k) = k;
    RAISE NOTICE 'Bucket %/% committed. Rows in bucket now: %', k+1, p_k, done;
  END LOOP;
  RAISE NOTICE 'All % buckets completed.', p_k;
END
$$;

CALL public.upsert_rag_study_corpus_all(16);
DROP MATERIALIZED VIEW IF EXISTS public.rag_study_keys;

CREATE MATERIALIZED VIEW public.rag_study_keys AS
WITH base AS (
  SELECT
    c.nct_id,
    -- Safely explode CONDITIONS:
    jsonb_array_elements_text(
      CASE jsonb_typeof(c.study_json->'labels'->'conditions')
        WHEN 'array'  THEN c.study_json->'labels'->'conditions'
        WHEN 'string' THEN jsonb_build_array(c.study_json->'labels'->'conditions')
        ELSE '[]'::jsonb
      END
    ) AS condition_raw,

    -- Safely explode INTERVENTIONS:
    jsonb_array_elements_text(
      CASE jsonb_typeof(c.study_json->'labels'->'interventions')
        WHEN 'array'  THEN c.study_json->'labels'->'interventions'
        WHEN 'string' THEN jsonb_build_array(c.study_json->'labels'->'interventions')
        ELSE '[]'::jsonb
      END
    ) AS intervention_raw
  FROM public.rag_study_corpus c
),
aliases AS (
  SELECT s.nct_id, NULLIF(trim(s.acronym),'') AS alias_raw
  FROM public.ctgov_studies s
  UNION ALL
  SELECT ii.nct_id, NULLIF(trim(ii.id_value),'')
  FROM public.ctgov_id_information ii
  UNION ALL
  SELECT k.nct_id, NULLIF(trim(k.name),'')
  FROM public.ctgov_keywords k
  UNION ALL
  SELECT i.nct_id, NULLIF(trim(ion.name),'')
  FROM public.ctgov_interventions i
  JOIN public.ctgov_intervention_other_names ion ON ion.intervention_id = i.id
),
all_rows AS (
  SELECT nct_id, condition_raw, NULL::text AS intervention_raw, NULL::text AS alias_raw FROM base
  UNION ALL
  SELECT nct_id, NULL, intervention_raw, NULL FROM base
  UNION ALL
  SELECT nct_id, NULL, NULL, alias_raw FROM aliases
)
SELECT DISTINCT
  nct_id,
  NULLIF(trim(condition_raw), '')      AS condition_name,
  NULLIF(trim(intervention_raw), '')   AS intervention_name,
  NULLIF(trim(alias_raw), '')          AS alias_name,
  lower(unaccent(NULLIF(trim(condition_raw),     ''))) AS condition_norm,
  lower(unaccent(NULLIF(trim(intervention_raw),  ''))) AS intervention_norm,
  lower(unaccent(NULLIF(trim(alias_raw),         ''))) AS alias_norm
FROM all_rows
WHERE (condition_raw IS NOT NULL AND NULLIF(trim(condition_raw),'') IS NOT NULL)
   OR (intervention_raw IS NOT NULL AND NULLIF(trim(intervention_raw),'') IS NOT NULL)
   OR (alias_raw IS NOT NULL AND NULLIF(trim(alias_raw),'') IS NOT NULL)
WITH NO DATA;

-- First blocking populate (sets "populated" flag)
REFRESH MATERIALIZED VIEW public.rag_study_keys;

-- Unique + search indexes (idempotent)
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_keys_unique
  ON public.rag_study_keys (nct_id,
                            COALESCE(condition_name,''),
                            COALESCE(intervention_name,''),
                            COALESCE(alias_name,''));

CREATE INDEX IF NOT EXISTS idx_rag_keys_cond_norm_trgm  ON public.rag_study_keys USING gin (condition_norm    gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_rag_keys_intr_norm_trgm  ON public.rag_study_keys USING gin (intervention_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_rag_keys_alias_norm_trgm ON public.rag_study_keys USING gin (alias_norm        gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_rag_keys_nct             ON public.rag_study_keys (nct_id);
DROP FUNCTION IF EXISTS public.search_trials(text, text, integer);
CREATE OR REPLACE FUNCTION public.search_trials(
  p_kind  text,         -- 'nct' | 'condition' | 'intervention' | 'alias' | 'auto'
  p_query text,
  p_limit int DEFAULT 50
) RETURNS TABLE (
  nct_id     text,
  score      real,
  study_json jsonb
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
  IF p_kind = 'nct' THEN
    RETURN QUERY
      SELECT c.nct_id, 1.0::real, c.study_json
      FROM public.rag_study_corpus c
      WHERE c.nct_id = p_query
      LIMIT p_limit;

  ELSIF p_kind = 'condition' THEN
    RETURN QUERY
      WITH q AS (SELECT lower(unaccent(p_query)) AS q)
      SELECT k.nct_id, similarity(k.condition_norm, q.q), c.study_json
      FROM public.rag_study_keys k
      JOIN q ON TRUE
      JOIN public.rag_study_corpus c USING (nct_id)
      WHERE k.condition_norm % q.q
      ORDER BY 2 DESC, k.nct_id
      LIMIT p_limit;

  ELSIF p_kind = 'intervention' THEN
    RETURN QUERY
      WITH q AS (SELECT lower(unaccent(p_query)) AS q)
      SELECT k.nct_id, similarity(k.intervention_norm, q.q), c.study_json
      FROM public.rag_study_keys k
      JOIN q ON TRUE
      JOIN public.rag_study_corpus c USING (nct_id)
      WHERE k.intervention_norm % q.q
      ORDER BY 2 DESC, k.nct_id
      LIMIT p_limit;

  ELSIF p_kind = 'alias' THEN
    RETURN QUERY
      WITH q AS (SELECT lower(unaccent(p_query)) AS q)
      SELECT k.nct_id, similarity(k.alias_norm, q.q), c.study_json
      FROM public.rag_study_keys k
      JOIN q ON TRUE
      JOIN public.rag_study_corpus c USING (nct_id)
      WHERE k.alias_norm % q.q
      ORDER BY 2 DESC, k.nct_id
      LIMIT p_limit;

  ELSIF p_kind = 'auto' THEN
    RETURN QUERY
      SELECT c.nct_id, 1.0::real, c.study_json
      FROM public.rag_study_corpus c
      WHERE c.nct_id = p_query
      LIMIT p_limit;

    IF NOT FOUND THEN
      RETURN QUERY
        WITH q AS (SELECT lower(unaccent(p_query)) AS q),
        r AS (
          SELECT k.nct_id, similarity(k.alias_norm, q.q)        AS score FROM public.rag_study_keys k, q WHERE k.alias_norm % q.q
          UNION
          SELECT k.nct_id, similarity(k.intervention_norm, q.q) AS score FROM public.rag_study_keys k, q WHERE k.intervention_norm % q.q
          UNION
          SELECT k.nct_id, similarity(k.condition_norm, q.q)    AS score FROM public.rag_study_keys k, q WHERE k.condition_norm % q.q
        )
        SELECT r.nct_id, r.score, c.study_json
        FROM r JOIN public.rag_study_corpus c USING (nct_id)
        ORDER BY r.score DESC, r.nct_id
        LIMIT p_limit;
    END IF;

  ELSE
    RAISE EXCEPTION 'Unknown kind: %, expected nct|condition|intervention|alias|auto', p_kind;
  END IF;
END
$$;


DROP FUNCTION IF EXISTS public.search_trials_combo(text, text, int);

-- Replace the old version if present
DROP FUNCTION IF EXISTS public.search_trials_combo(text, text, int);

CREATE FUNCTION public.search_trials_combo(
  p_condition     text,          -- e.g., 'partial-onset seizures'
  p_intervention  text,          -- e.g., 'lamotrigine'
  p_limit         int DEFAULT 50
) RETURNS TABLE (
  nct_id     text,
  score      real,
  study_json jsonb
)
LANGUAGE sql
STABLE
AS $$
WITH
qc AS (
  SELECT CASE WHEN p_condition IS NULL OR btrim(p_condition) = '' THEN NULL
              ELSE lower(unaccent(p_condition)) END AS q
),
qi AS (
  SELECT CASE WHEN p_intervention IS NULL OR btrim(p_intervention) = '' THEN NULL
              ELSE lower(unaccent(p_intervention)) END AS q
),

-- condition hits
cond AS (
  SELECT k.nct_id,
         similarity(k.condition_norm, qc.q) AS s_c
  FROM public.rag_study_keys k, qc
  WHERE qc.q IS NOT NULL
    AND k.condition_norm % qc.q
),

-- intervention hits
intr AS (
  SELECT k.nct_id,
         similarity(k.intervention_norm, qi.q) AS s_i
  FROM public.rag_study_keys k, qi
  WHERE qi.q IS NOT NULL
    AND k.intervention_norm % qi.q
),

-- intersection & score (rename from "both" -> "both_hits" to avoid keyword)
both_hits AS (
  SELECT c.nct_id,
         ((c.s_c + i.s_i) / 2.0)::real AS score
  FROM cond c
  JOIN intr i USING (nct_id)
),

-- single-key fallbacks if one side missing
only_cond AS (
  SELECT c.nct_id, c.s_c::real AS score
  FROM cond c
  WHERE NOT EXISTS (SELECT 1 FROM intr)
),
only_intr AS (
  SELECT i.nct_id, i.s_i::real AS score
  FROM intr i
  WHERE NOT EXISTS (SELECT 1 FROM cond)
),

unioned AS (
  SELECT * FROM both_hits
  UNION ALL
  SELECT * FROM only_cond
  UNION ALL
  SELECT * FROM only_intr
)

SELECT u.nct_id, u.score, c.study_json
FROM unioned u
JOIN public.rag_study_corpus c USING (nct_id)
ORDER BY u.score DESC, u.nct_id
LIMIT p_limit;
$$;
