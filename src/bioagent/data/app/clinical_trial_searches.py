from __future__ import annotations

import json
from collections.abc import Iterable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Reuse your infra
from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


def _nz(x: Any, default: str = "—") -> str:
    """Safe string for None/empty."""
    return default if x in (None, "", [], {}) else str(x)


def _as_list(obj: Any) -> list[Any]:
    return obj if isinstance(obj, list) else []


def _fmt_bool(b: bool | None) -> str:
    if b is True:
        return "True"
    if b is False:
        return "False"
    return "Unknown"


def _fmt_ci(lower: Any, upper: Any, pct: Any) -> str | None:
    if lower is not None and upper is not None:
        return f"{_nz(pct, '')}% CI [{lower}, {upper}]".strip()
    return None


def _iter_outcome_measures(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for m in _as_list(o.get("measures")):
            yield {
                "outcome_type": o.get("type"),
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "group_title": m.get("group_title") or f"Group {m.get('group_idx')}",
                "classification": m.get("classification"),
                "param_type": m.get("param_type"),
                "value": m.get("value"),
                "value_num": m.get("value_num"),
                "units": m.get("units"),
                "dispersion_type": m.get("dispersion_type"),
                "dispersion": m.get("dispersion"),
                "lower": m.get("lower"),
                "upper": m.get("upper"),
            }


def _iter_outcome_counts(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for c in _as_list(o.get("counts")):
            yield {
                "outcome_type": o.get("type"),
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "group_title": c.get("group_title") or f"Group {c.get('group_idx')}",
                "scope": c.get("scope"),
                "units": c.get("units"),
                "count": c.get("count"),
            }


def _iter_outcome_analyses(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for a in _as_list(o.get("analyses")):
            comps = " vs ".join(g.get("group_title") for g in _as_list(a.get("comparison_groups")) if g.get("group_title"))
            yield {
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "method": a.get("method"),
                "param_type": a.get("param_type"),
                "param_value": a.get("param_value"),
                "p_value": a.get("p_value"),
                "ci_percent": a.get("ci_percent"),
                "ci_lower": a.get("ci_lower"),
                "ci_upper": a.get("ci_upper"),
                "comparison_groups": comps,
                "description": a.get("description"),
            }


def _iter_adverse_events(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for ae in _as_list(study_json.get("adverse_events")):
        yield {
            "term": ae.get("term"),
            "event_type": ae.get("event_type"),
            "group_title": ae.get("group_title") or f"Group {ae.get('group_idx')}",
            "affected": ae.get("subjects_affected"),
            "at_risk": ae.get("subjects_at_risk"),
            "rate_pct": ae.get("rate"),
        }


def _as_json_dict(v) -> dict:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return json.loads(bytes(v).decode("utf-8"))
        except Exception:
            return {}
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    # last resort: try mapping-like
    try:
        return dict(v)  # may raise
    except Exception:
        return {}


# ==============================
# 1) Models
# ==============================


class SearchKind(str, Enum):
    """Search scope choices wired to Postgres `public.search_trials`."""

    nct = "nct"
    condition = "condition"
    intervention = "intervention"
    alias = "alias"
    auto = "auto"
    combo = "combo"


class ClinicalTrialsSearchInput(BaseModel):
    kind: SearchKind = Field(description="nct | condition | intervention | alias | auto | combo")
    query: str = Field(default="", description="Used for single-key searches")
    limit: int = Field(default=25, ge=1, le=200)
    return_json: bool = Field(default=False)

    # NEW: for combo searches
    condition_query: str | None = None
    intervention_query: str | None = None
    
    # Output control flags - control what sections to display
    output_eligibility: bool = Field(default=True, description="Include eligibility criteria")
    output_groups: bool = Field(default=True, description="Include registry-defined groups & interventions")
    output_baseline_measurements: bool = Field(default=True, description="Include baseline measurements")
    output_results: bool = Field(default=True, description="Include outcome results")
    output_adverse_effects: bool = Field(default=True, description="Include adverse events")
    output_sponsors: bool = Field(default=True, description="Include sponsor information")
    output_countries: bool = Field(default=True, description="Include country information")

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        return (v or "").strip()


class TrialHit(BaseModel):
    """One search hit."""

    nct_id: str
    score: float
    summary: str
    study_json: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata about the match (e.g., matched MeSH terms, match source)"
    )


class ClinicalTrialsSearchOutput(BaseModel):
    status: str  # "success" | "not_found" | "error"
    hits: list[TrialHit] | None = None
    error: str | None = None


# ==============================
# 2) Rendering
# ==============================


def _list_from(obj: dict[str, Any], key: str) -> list[str]:
    v = obj.get(key)
    return v if isinstance(v, list) else []


def render_study_text_full(
    study_json: dict[str, Any],
    nct_id: str | None = None,
    output_eligibility: bool = True,
    output_groups: bool = True,
    output_baseline_measurements: bool = True,
    output_results: bool = True,
    output_adverse_effects: bool = True,
    output_sponsors: bool = True,
    output_countries: bool = True,
) -> str:
    """
    Render a comprehensive, ordered narrative for a single trial.

    Sections:
      1) Header & metadata (always included)
      2) Design (always included)
      3) Eligibility (controlled by output_eligibility)
      4) Labels: conditions (always), interventions (always), sponsors/countries (controlled by flags)
      5) Registry-defined groups (controlled by output_groups)
      6) Results groups with baseline measurements (controlled by output_baseline_measurements)
      7) Outcomes (controlled by output_results)
      8) Adverse events (controlled by output_adverse_effects)
    
    Args:
        study_json: The study JSON data
        nct_id: Optional NCT ID to include in header
        output_eligibility: Include eligibility criteria
        output_groups: Include registry-defined groups & interventions
        output_baseline_measurements: Include baseline measurements
        output_results: Include outcome results
        output_adverse_effects: Include adverse events
        output_sponsors: Include sponsor information
        output_countries: Include country information
    """
    md = study_json.get("metadata") or {}
    dsn = study_json.get("design") or {}
    elig = study_json.get("eligibility") or {}
    labels = study_json.get("labels") or {}
    reg_groups = _as_list(study_json.get("registry_groups"))
    res_groups = _as_list(study_json.get("results_groups"))
    outcomes = _as_list(study_json.get("outcomes"))
    aes = _as_list(study_json.get("adverse_events"))

    lines: list[str] = []

    # 1) Header & metadata
    header = _nz(md.get("brief_title"), "[No brief title]")
    if nct_id:
        header = f"{header} [{nct_id}]"
    lines.append(header)

    lines.append(
        "Type/Phase/Status: {stype}, {phase}, {status}.".format(
            stype=_nz(md.get("study_type")),
            phase=_nz(md.get("phase")),
            status=_nz(md.get("overall_status")),
        )
    )
    lines.append(
        "Enrollment: {enr} ({enr_t}). Dates — start: {start}; primary completion: {pc}; completion: {comp}.".format(
            enr=_nz(md.get("enrollment")),
            enr_t=_nz(md.get("enrollment_type")),
            start=_nz(md.get("start_date")),
            pc=_nz(md.get("primary_completion_date")),
            comp=_nz(md.get("completion_date")),
        )
    )
    if md.get("why_stopped"):
        lines.append(f"Why stopped: {_nz(md.get('why_stopped'))}.")
    if "has_results" in md:
        lines.append(f"Results posted: {_fmt_bool(md.get('has_results'))}.")

    # 2) Design
    lines.append(
        "Design — allocation: {alloc}; model: {model}; masking: {mask}; "
        "purpose: {purpose}; arms: {arms}; groups: {groups}.".format(
            alloc=_nz(dsn.get("allocation")),
            model=_nz(dsn.get("intervention_model")),
            mask=_nz(dsn.get("masking")),
            purpose=_nz(dsn.get("primary_purpose")),
            arms=_nz(md.get("number_of_arms")),
            groups=_nz(md.get("number_of_groups")),
        )
    )
    if dsn.get("masking_description"):
        lines.append("Masking description: " + _nz(dsn.get("masking_description")) + ".")

    # 3) Eligibility
    if output_eligibility:
        lines.append(
            "Eligibility — gender: {gender}; age: {min_age}–{max_age}; healthy volunteers: {hv}.".format(
                gender=_nz(elig.get("gender")),
                min_age=_nz(elig.get("minimum_age")),
                max_age=_nz(elig.get("maximum_age")),
                hv=_fmt_bool(elig.get("healthy_volunteers")),
            )
        )
        # Keep criteria short; call-site can print full criteria if desired.
        crit = elig.get("criteria") or ""
        if isinstance(crit, str) and crit.strip():
            first_line = crit.strip().splitlines()[0]
            lines.append("Eligibility criteria (first line): " + first_line)

    # 4) Labels
    conditions = labels.get("conditions") or []
    interventions = labels.get("interventions") or []
    sponsors = labels.get("sponsors") or []
    countries = labels.get("countries") or []
    lines.append("Conditions: " + (", ".join(map(str, conditions)) if conditions else "—") + ".")
    lines.append("Interventions (registry): " + (", ".join(map(str, interventions)) if interventions else "—") + ".")
    if output_sponsors and sponsors:
        lines.append("Lead sponsors: " + ", ".join(map(str, sponsors)) + ".")
    if output_countries and countries:
        lines.append("Countries: " + ", ".join(map(str, countries)) + ".")

    # 5) Registry-defined groups (with assigned interventions)
    if output_groups and reg_groups:
        lines.append("Registry-defined groups & assigned interventions:")
        for g in reg_groups:
            title = _nz(g.get("title"))
            desc = g.get("description")
            ivs = [i for i in _as_list(g.get("interventions")) if isinstance(i, dict)]
            if ivs:
                iv_strs = []
                for i in ivs:
                    bit = _nz(i.get("name"))
                    itype = i.get("type")
                    idesc = i.get("description")
                    if itype:
                        bit += f" [{itype}]"
                    if idesc:
                        bit += f" — {idesc}"
                    iv_strs.append(bit)
                lines.append(f"  • {title}: " + "; ".join(iv_strs))
            else:
                lines.append(f"  • {title}")
            if desc and not ivs:
                lines.append(f"    {desc}")

    # 6) Results groups, baselines & baseline measurements
    if output_baseline_measurements and res_groups:
         lines.append("Results groups, baseline counts & baseline measurements:")
         for g in res_groups:
             title = _nz(g.get("title"))
             lines.append(f"  • Group: {title}")
             # Baseline counts (often overall participants analyzed)
             for bc in _as_list(g.get("baselines")):
                 units = bc.get("units")
                 scope = bc.get("scope")
                 count = bc.get("count")
                 lines.append(f"    - Baseline count: {_nz(count)} {units or ''} (scope: {_nz(scope)})".rstrip())
             # Baseline measurements (e.g., Age mean ± SD)
             for bm in _as_list(g.get("baseline_measurements")):
                 parts = [
                     _nz(bm.get("title")),
                     (_nz(bm.get("param_type")).title() if bm.get("param_type") else ""),
                     _nz(bm.get("value")),
                     _nz(bm.get("units"), ""),
                 ]
                 disp = None
                 if bm.get("dispersion_type") and bm.get("dispersion") is not None:
                     disp = f"{str(bm['dispersion_type']).replace('_', ' ').title()} {_nz(bm['dispersion'])}"
                 if disp:
                     parts.append(f"({disp})")
                 lines.append("    - Baseline measure: " + " ".join([p for p in parts if p]).strip())

    # 7) Outcomes (measures, counts, analyses)
    if output_results and outcomes:
        lines.append("Outcomes:")
        
        # (a) numeric measures per group - group by header
        measures_by_header = {}
        for row in _iter_outcome_measures(study_json):
            header_key = (row['outcome_type'], row['outcome_title'], row['time_frame'])
            if header_key not in measures_by_header:
                measures_by_header[header_key] = []
            measures_by_header[header_key].append(row)
        
        for header_key, measures in measures_by_header.items():
            outcome_type, outcome_title, time_frame = header_key
            header = f"  • [{_nz(outcome_type).title()}] {_nz(outcome_title)} [{_nz(time_frame)}]"
            lines.append(header)
            for row in measures:
                parts = [
                    f"    - {row['group_title']}:",
                    (row.get("param_type") or ""),
                    _nz(row.get("value")),
                    _nz(row.get("units"), ""),
                ]
                if row.get("dispersion_type") and row.get("dispersion") is not None:
                    parts.append(f"({str(row['dispersion_type']).replace('_', ' ').title()} {row['dispersion']})")
                ci = _fmt_ci(row.get("lower"), row.get("upper"), None)  # ci_percent lives in analyses, not measures
                # Only add CI if present at measure level; most CIs live under analyses.
                if ci:
                    parts.append(ci)
                # classification (e.g., Baseline)
                if row.get("classification"):
                    parts.append(f"[{row['classification']}]")
                lines.append(" ".join([p for p in parts if p]).strip())

        # (b) count outcomes (e.g., number of participants) - group by header
        counts_by_header = {}
        for row in _iter_outcome_counts(study_json):
            header_key = (row['outcome_type'], row['outcome_title'], row['time_frame'])
            if header_key not in counts_by_header:
                counts_by_header[header_key] = []
            counts_by_header[header_key].append(row)
        
        for header_key, counts in counts_by_header.items():
            outcome_type, outcome_title, time_frame = header_key
            header = f"  • [{_nz(outcome_type).title()}] {_nz(outcome_title)} [{_nz(time_frame)}]"
            lines.append(header)
            for row in counts:
                lines.append(f"    - {row['group_title']}: {_nz(row['count'])} {_nz(row['units'])} (scope: {_nz(row['scope'])})")

        # (c) analyses (comparisons, p-values, CIs) - group by outcome
        analyses_by_outcome = {}
        for a in _iter_outcome_analyses(study_json):
            outcome_key = (a['outcome_title'], a['time_frame'])
            if outcome_key not in analyses_by_outcome:
                analyses_by_outcome[outcome_key] = []
            analyses_by_outcome[outcome_key].append(a)
        
        for outcome_key, analyses in analyses_by_outcome.items():
            # Analyses typically follow the measures/counts for the same outcome,
            # so we just print them without a header if they're already associated
            for a in analyses:
                bits = [a.get("method") or "", a.get("comparison_groups") or ""]
                
                # Add test statistic if available
                if a.get("param_type") and a.get("param_value") is not None:
                    bits.append(f"{a['param_type']}={a['param_value']}")
                
                # Add p-value
                if a.get("p_value") is not None:
                    bits.append(f"p={a['p_value']}")
                
                # Add confidence interval
                ci = _fmt_ci(a.get("ci_lower"), a.get("ci_upper"), a.get("ci_percent"))
                if ci:
                    bits.append(ci)
                
                desc = a.get("description")
                line = "    - Analysis: " + ", ".join([b for b in bits if b]).strip()
                lines.append(line)
                if desc:
                    lines.append(f"      {desc}")

    # 8) Adverse events - group by term and event type
    if output_adverse_effects and aes:
        lines.append("Adverse events (per group):")
        aes_by_term = {}
        for ae in _iter_adverse_events(study_json):
            ae_key = (ae['term'], ae['event_type'])
            if ae_key not in aes_by_term:
                aes_by_term[ae_key] = []
            aes_by_term[ae_key].append(ae)
        
        for ae_key, ae_list in aes_by_term.items():
            term, event_type = ae_key
            header = f"  • {_nz(term)} [{_nz(event_type)}]"
            lines.append(header)
            for ae in ae_list:
                line = f"    - {_nz(ae['group_title'])}: {_nz(ae['affected'])}/{_nz(ae['at_risk'])}"
                if ae.get("rate_pct") is not None:
                    line += f" ({_nz(ae['rate_pct'])}%)"
                lines.append(line)

    return "\n".join(lines)


# ==============================
# 3) Core DB call
# ==============================


async def _search_trials_combo_db(
    async_cfg: AsyncDatabaseConfig,
    condition_query: str | None,
    intervention_query: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    sql = """
        SELECT nct_id, score, (study_json)::jsonb AS study_json
        FROM public.search_trials_combo($1, $2, $3);
    """
    return await async_cfg.execute_query(sql, condition_query, intervention_query, limit)


async def _search_trials_db(
    async_cfg: AsyncDatabaseConfig,
    kind: SearchKind,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Call Postgres function `public.search_trials(kind, query, limit)`.
    Returns rows as dicts: nct_id, score, study_json
    """
    sql = """
        SELECT nct_id, score, study_json
        FROM public.search_trials($1, $2, $3);
    """
    rows: list[dict[str, Any]] = await async_cfg.execute_query(sql, kind.value, query, limit)
    return rows


# ==============================
# 4) Public API
# ==============================


async def clinical_trials_search_async(
    db_config: DatabaseConfig,
    search_input: ClinicalTrialsSearchInput,
) -> ClinicalTrialsSearchOutput:
    try:
        async_cfg = await get_async_connection(db_config)

        if search_input.kind == SearchKind("combo"):
            rows = await _search_trials_combo_db(
                async_cfg,
                (search_input.condition_query or "").strip() or None,
                (search_input.intervention_query or "").strip() or None,
                search_input.limit,
            )
        else:
            rows = await _search_trials_db(async_cfg, search_input.kind, search_input.query, search_input.limit)

        if not rows:
            return ClinicalTrialsSearchOutput(status="not_found", hits=[], error="No matches.")

        hits: list[TrialHit] = []
        for r in rows:
            sj = _as_json_dict(r.get("study_json"))
            summary = render_study_text_full(
                sj,
                nct_id=r["nct_id"],
                output_eligibility=search_input.output_eligibility,
                output_groups=search_input.output_groups,
                output_baseline_measurements=search_input.output_baseline_measurements,
                output_results=search_input.output_results,
                output_adverse_effects=search_input.output_adverse_effects,
                output_sponsors=search_input.output_sponsors,
                output_countries=search_input.output_countries,
            )
            hits.append(
                TrialHit(
                    nct_id=r["nct_id"],
                    score=float(r.get("score") or 0.0),
                    summary=summary,
                    study_json=sj if search_input.return_json else None,
                )
            )
        return ClinicalTrialsSearchOutput(status="success", hits=hits)
    except Exception as e:
        return ClinicalTrialsSearchOutput(status="error", error=f"{type(e).__name__}: {e}")


async def clinical_trials_search(
    db_config: DatabaseConfig,
    search_input: ClinicalTrialsSearchInput,
) -> ClinicalTrialsSearchOutput:
    """
    Sync convenience wrapper.
    """
    return await clinical_trials_search_async(db_config, search_input)
