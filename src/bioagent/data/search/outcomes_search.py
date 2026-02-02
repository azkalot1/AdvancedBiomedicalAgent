#!/usr/bin/env python3
"""
Clinical Trial Outcomes Search.

Queries outcome titles, measurements, and analyses from CT.gov tables.
Supports trial-centric outcome retrieval, outcome keyword searches across
trials, and drug-centric efficacy comparisons based on p-values.

Modes:
    - outcomes_for_trial: outcomes and analyses for a given NCT ID
    - trials_with_outcome: trials matching an outcome keyword
    - efficacy_comparison: analyses for trials involving a drug

Data Sources:
    - ctgov_outcomes
    - ctgov_outcome_measurements
    - ctgov_outcome_analyses
    - rag_study_search
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


class OutcomesSearchInput(BaseModel):
    mode: Literal["outcomes_for_trial", "trials_with_outcome", "efficacy_comparison"]
    nct_id: str | None = None
    outcome_keyword: str | None = None
    drug_name: str | None = None
    outcome_type: Literal["primary", "secondary", "all"] = "all"
    min_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    max_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("nct_id", "outcome_keyword", "drug_name", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode == "outcomes_for_trial" and not self.nct_id:
            errors.append("'nct_id' is required for outcomes_for_trial")
        if self.mode == "trials_with_outcome" and not self.outcome_keyword:
            errors.append("'outcome_keyword' is required for trials_with_outcome")
        if self.mode == "efficacy_comparison" and not self.drug_name:
            errors.append("'drug_name' is required for efficacy_comparison")
        return errors


class OutcomeHit(BaseModel):
    outcome_id: int | None = None
    nct_id: str
    outcome_type: str | None = None
    title: str | None = None
    description: str | None = None
    time_frame: str | None = None
    param_type: str | None = None
    units: str | None = None


class OutcomeMeasurementHit(BaseModel):
    outcome_id: int | None = None
    nct_id: str
    title: str | None = None
    param_type: str | None = None
    param_value_num: float | None = None
    units: str | None = None
    dispersion_type: str | None = None
    dispersion_value_num: float | None = None
    dispersion_lower_limit: float | None = None
    dispersion_upper_limit: float | None = None


class OutcomeAnalysisHit(BaseModel):
    outcome_id: int | None = None
    nct_id: str
    outcome_title: str | None = None
    outcome_type: str | None = None
    p_value: float | None = None
    method: str | None = None
    param_type: str | None = None
    param_value: float | None = None
    ci_percent: float | None = None
    ci_lower_limit: float | None = None
    ci_upper_limit: float | None = None


class TrialOutcomeMatchHit(BaseModel):
    nct_id: str
    brief_title: str | None = None
    phase: str | None = None
    status: str | None = None
    outcome_title: str | None = None
    outcome_type: str | None = None
    time_frame: str | None = None


class TrialOutcomesBundle(BaseModel):
    nct_id: str
    outcomes: list[OutcomeHit] = Field(default_factory=list)
    measurements: list[OutcomeMeasurementHit] = Field(default_factory=list)
    analyses: list[OutcomeAnalysisHit] = Field(default_factory=list)


class OutcomesSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    error: str | None = None
    query_summary: str = ""


def _outcome_type_filter(outcome_type: str, param_index: int) -> tuple[str, list[Any]]:
    if outcome_type == "all":
        return "", []
    return f"AND o.outcome_type = ${param_index}", [outcome_type.upper()]


async def outcomes_search_async(
    db_config: DatabaseConfig,
    search_input: OutcomesSearchInput,
) -> OutcomesSearchOutput:
    """
    Search trial outcomes, measurements, and analyses.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: OutcomesSearchInput with mode and filters.

    Returns:
        OutcomesSearchOutput with status, hits, and query summary.
    """
    errors = search_input.validate_for_mode()
    if errors:
        return OutcomesSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
            query_summary="invalid input",
        )

    try:
        async_config = await get_async_connection(db_config)

        if search_input.mode == "outcomes_for_trial":
            params: list[Any] = [search_input.nct_id]
            outcome_filter, outcome_params = _outcome_type_filter(search_input.outcome_type, 2)
            params.extend(outcome_params)
            params.extend([search_input.limit, search_input.offset])

            outcomes_sql = f"""
                SELECT o.id AS outcome_id, o.nct_id, o.outcome_type, o.title, o.description,
                       o.time_frame, o.param_type, o.units
                FROM ctgov_outcomes o
                WHERE o.nct_id = $1
                {outcome_filter}
                ORDER BY o.id
                LIMIT ${len(params) - 1} OFFSET ${len(params)}
            """
            outcome_rows = await async_config.execute_query(outcomes_sql, *params)
            outcomes = [OutcomeHit(**row) for row in outcome_rows]

            measurement_rows = await async_config.execute_query(
                """
                SELECT m.outcome_id, m.nct_id, m.title, m.param_type, m.param_value_num,
                       m.units, m.dispersion_type, m.dispersion_value_num,
                       m.dispersion_lower_limit, m.dispersion_upper_limit
                FROM ctgov_outcome_measurements m
                WHERE m.nct_id = $1
                ORDER BY m.outcome_id
                LIMIT $2 OFFSET $3
                """,
                search_input.nct_id,
                min(200, search_input.limit * 4),
                0,
            )
            measurements = [OutcomeMeasurementHit(**row) for row in measurement_rows]

            analysis_params: list[Any] = [search_input.nct_id]
            analysis_filters: list[str] = []
            if search_input.min_p_value is not None:
                analysis_params.append(search_input.min_p_value)
                analysis_filters.append(f"a.p_value >= ${len(analysis_params)}")
            if search_input.max_p_value is not None:
                analysis_params.append(search_input.max_p_value)
                analysis_filters.append(f"a.p_value <= ${len(analysis_params)}")
            analysis_where = " AND ".join(analysis_filters)
            analysis_where = f"AND {analysis_where}" if analysis_where else ""

            analysis_rows = await async_config.execute_query(
                f"""
                SELECT a.outcome_id, a.nct_id, o.title AS outcome_title, o.outcome_type,
                       a.p_value, a.method, a.param_type, a.param_value,
                       a.ci_percent, a.ci_lower_limit, a.ci_upper_limit
                FROM ctgov_outcome_analyses a
                LEFT JOIN ctgov_outcomes o ON a.outcome_id = o.id
                WHERE a.nct_id = $1
                {analysis_where}
                ORDER BY a.p_value ASC NULLS LAST
                LIMIT $2 OFFSET $3
                """,
                *analysis_params,
                min(200, search_input.limit * 4),
                0,
            )
            analyses = [OutcomeAnalysisHit(**row) for row in analysis_rows]

            hits = [TrialOutcomesBundle(
                nct_id=search_input.nct_id or "",
                outcomes=outcomes,
                measurements=measurements,
                analyses=analyses,
            )]
            status = "success" if (outcomes or measurements or analyses) else "not_found"
            return OutcomesSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.nct_id or "",
            )

        if search_input.mode == "trials_with_outcome":
            keyword = search_input.outcome_keyword or ""
            params: list[Any] = [f"%{keyword}%"]
            outcome_filter, outcome_params = _outcome_type_filter(search_input.outcome_type, 2)
            params.extend(outcome_params)
            params.extend([search_input.limit, search_input.offset])
            sql = f"""
                WITH matched_outcomes AS (
                    SELECT DISTINCT o.nct_id, o.title, o.outcome_type, o.time_frame
                    FROM ctgov_outcomes o
                    WHERE (o.title ILIKE $1 OR o.description ILIKE $1)
                    {outcome_filter}
                    UNION
                    SELECT DISTINCT m.nct_id, m.title, NULL::text AS outcome_type, NULL::text AS time_frame
                    FROM ctgov_outcome_measurements m
                    WHERE m.title ILIKE $1 OR m.description ILIKE $1
                )
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       mo.title AS outcome_title, mo.outcome_type, mo.time_frame
                FROM matched_outcomes mo
                JOIN rag_study_search rs ON rs.nct_id = mo.nct_id
                ORDER BY rs.nct_id
                LIMIT ${len(params) - 1} OFFSET ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [TrialOutcomeMatchHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                outcome_title=row.get("outcome_title"),
                outcome_type=row.get("outcome_type"),
                time_frame=row.get("time_frame"),
            ) for row in rows]
            status = "success" if hits else "not_found"
            return OutcomesSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=keyword,
            )

        if search_input.mode == "efficacy_comparison":
            drug_name = search_input.drug_name or ""
            drug_norm = drug_name.lower()
            tsquery = " & ".join(drug_name.split())
            params: list[Any] = [drug_norm, f"%{drug_name}%", tsquery]

            p_filters: list[str] = ["a.p_value IS NOT NULL"]
            if search_input.min_p_value is not None:
                params.append(search_input.min_p_value)
                p_filters.append(f"a.p_value >= ${len(params)}")
            if search_input.max_p_value is not None:
                params.append(search_input.max_p_value)
                p_filters.append(f"a.p_value <= ${len(params)}")
            p_filter_sql = " AND ".join(p_filters)

            params.extend([search_input.limit, search_input.offset])
            sql = f"""
                WITH matched_trials AS (
                    SELECT nct_id
                    FROM rag_study_search
                    WHERE interventions_norm % $1
                       OR interventions_norm ILIKE $2
                       OR terms_tsv @@ to_tsquery('english', $3)
                ),
                analyses AS (
                    SELECT a.nct_id, a.outcome_id, a.p_value, a.method, a.param_type, a.param_value,
                           a.ci_percent, a.ci_lower_limit, a.ci_upper_limit,
                           o.title AS outcome_title, o.outcome_type
                    FROM ctgov_outcome_analyses a
                    LEFT JOIN ctgov_outcomes o ON a.outcome_id = o.id
                    WHERE a.nct_id IN (SELECT nct_id FROM matched_trials)
                      AND {p_filter_sql}
                )
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       analyses.outcome_title, analyses.outcome_type, analyses.p_value,
                       analyses.method, analyses.param_type, analyses.param_value,
                       analyses.ci_percent, analyses.ci_lower_limit, analyses.ci_upper_limit,
                       analyses.outcome_id
                FROM analyses
                JOIN rag_study_search rs ON rs.nct_id = analyses.nct_id
                ORDER BY analyses.p_value ASC NULLS LAST
                LIMIT ${len(params) - 1} OFFSET ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [OutcomeAnalysisHit(
                outcome_id=row.get("outcome_id"),
                nct_id=row["nct_id"],
                outcome_title=row.get("outcome_title"),
                outcome_type=row.get("outcome_type"),
                p_value=row.get("p_value"),
                method=row.get("method"),
                param_type=row.get("param_type"),
                param_value=row.get("param_value"),
                ci_percent=row.get("ci_percent"),
                ci_lower_limit=row.get("ci_lower_limit"),
                ci_upper_limit=row.get("ci_upper_limit"),
            ) for row in rows]
            status = "success" if hits else "not_found"
            return OutcomesSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=drug_name,
            )

        return OutcomesSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="Unsupported mode. Supported: outcomes_for_trial, trials_with_outcome, efficacy_comparison",
            query_summary="invalid mode",
        )
    except Exception as exc:
        return OutcomesSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )
