#!/usr/bin/env python3
"""
Adverse Events Search for Clinical Trials (CT.gov).

Provides safety signal summaries and trial-level adverse event data for
drug- and event-centric queries. This module joins reported event tables
to the trial search index to support three primary modes:

- events_for_drug: summarize the most frequent adverse events for a drug
- drugs_with_event: find trials reporting a specific event term
- compare_safety: compare event profiles across multiple drugs

Data Sources:
    - ctgov_reported_events: reported adverse event summaries
    - rag_study_search: trial metadata and intervention matching
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


def _sanitize_tsquery(text: str) -> str:
    text = text or ""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " & ".join(cleaned.split())


class AdverseEventsSearchInput(BaseModel):
    mode: Literal["events_for_drug", "drugs_with_event", "compare_safety"]
    drug_name: str | None = None
    event_term: str | None = None
    drug_names: list[str] = Field(default_factory=list)
    severity: Literal["serious", "other", "all"] = "all"
    min_subjects_affected: int | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("drug_name", "event_term", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode == "events_for_drug" and not self.drug_name:
            errors.append("'drug_name' is required for events_for_drug")
        if self.mode == "drugs_with_event" and not self.event_term:
            errors.append("'event_term' is required for drugs_with_event")
        if self.mode == "compare_safety" and len(self.drug_names) < 2:
            errors.append("'drug_names' with at least 2 entries is required for compare_safety")
        return errors


class AdverseEventSummaryHit(BaseModel):
    adverse_event_term: str
    event_type: str | None = None
    subjects_affected: int | None = None
    subjects_at_risk: int | None = None
    n_trials: int | None = None


class AdverseEventTrialHit(BaseModel):
    nct_id: str
    brief_title: str | None = None
    phase: str | None = None
    status: str | None = None
    interventions: list[str] = Field(default_factory=list)
    adverse_event_term: str | None = None
    event_type: str | None = None
    subjects_affected: int | None = None
    subjects_at_risk: int | None = None


class AdverseEventDrugComparison(BaseModel):
    drug_name: str
    top_events: list[AdverseEventSummaryHit] = Field(default_factory=list)


class AdverseEventsSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    error: str | None = None
    query_summary: str = ""


def _severity_clause(severity: str) -> tuple[str, list[Any]]:
    if severity == "all":
        return "", []
    return "AND LOWER(re.event_type) = $1", [severity.lower()]


async def _fetch_events_for_drug(
    async_config: AsyncDatabaseConfig,
    drug_name: str,
    severity: str,
    min_subjects_affected: int | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    drug_norm = drug_name.lower()
    tsquery = _sanitize_tsquery(drug_name)
    params: list[Any] = [drug_norm, f"%{drug_name}%", tsquery]
    severity_clause, severity_params = _severity_clause(severity)
    params.extend(severity_params)

    having_clause = ""
    if min_subjects_affected is not None:
        params.append(min_subjects_affected)
        having_clause = f"HAVING SUM(re.subjects_affected) >= ${len(params)}"

    params.extend([limit, offset])
    sql = f"""
        WITH matched_trials AS (
            SELECT nct_id
            FROM rag_study_search
            WHERE interventions_norm % $1
               OR interventions_norm ILIKE $2
               OR terms_tsv @@ to_tsquery('english', $3)
        )
        SELECT re.adverse_event_term,
               re.event_type,
               SUM(re.subjects_affected)::int AS subjects_affected,
               SUM(re.subjects_at_risk)::int AS subjects_at_risk,
               COUNT(DISTINCT re.nct_id)::int AS n_trials
        FROM ctgov_reported_events re
        JOIN matched_trials mt ON mt.nct_id = re.nct_id
        WHERE re.adverse_event_term IS NOT NULL
        {severity_clause}
        GROUP BY re.adverse_event_term, re.event_type
        {having_clause}
        ORDER BY subjects_affected DESC NULLS LAST, n_trials DESC
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def _fetch_trials_with_event(
    async_config: AsyncDatabaseConfig,
    event_term: str,
    severity: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    params: list[Any] = [f"%{event_term}%"]
    severity_clause, severity_params = _severity_clause(severity)
    params.extend(severity_params)
    params.extend([limit, offset])
    sql = f"""
        SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status, rs.interventions,
               re.adverse_event_term, re.event_type, re.subjects_affected, re.subjects_at_risk
        FROM ctgov_reported_events re
        JOIN rag_study_search rs ON rs.nct_id = re.nct_id
        WHERE re.adverse_event_term ILIKE $1
        {severity_clause}
        ORDER BY re.subjects_affected DESC NULLS LAST, rs.nct_id
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def adverse_events_search_async(
    db_config: DatabaseConfig,
    search_input: AdverseEventsSearchInput,
) -> AdverseEventsSearchOutput:
    """
    Search CT.gov adverse events and safety signals.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: AdverseEventsSearchInput containing mode and filters.

    Returns:
        AdverseEventsSearchOutput with status, hits, and query summary.
        - events_for_drug: list of AdverseEventSummaryHit
        - drugs_with_event: list of AdverseEventTrialHit
        - compare_safety: list of AdverseEventDrugComparison
    """
    errors = search_input.validate_for_mode()
    if errors:
        return AdverseEventsSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
            query_summary="invalid input",
        )

    try:
        async_config = await get_async_connection(db_config)

        if search_input.mode == "events_for_drug":
            rows = await _fetch_events_for_drug(
                async_config,
                search_input.drug_name or "",
                search_input.severity,
                search_input.min_subjects_affected,
                search_input.limit,
                search_input.offset,
            )
            hits = [AdverseEventSummaryHit(**row) for row in rows]
            status = "success" if hits else "not_found"
            return AdverseEventsSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.drug_name or "",
            )

        if search_input.mode == "drugs_with_event":
            rows = await _fetch_trials_with_event(
                async_config,
                search_input.event_term or "",
                search_input.severity,
                search_input.limit,
                search_input.offset,
            )
            hits = [
                AdverseEventTrialHit(
                    nct_id=row["nct_id"],
                    brief_title=row.get("brief_title"),
                    phase=row.get("phase"),
                    status=row.get("overall_status"),
                    interventions=row.get("interventions") or [],
                    adverse_event_term=row.get("adverse_event_term"),
                    event_type=row.get("event_type"),
                    subjects_affected=row.get("subjects_affected"),
                    subjects_at_risk=row.get("subjects_at_risk"),
                )
                for row in rows
            ]
            status = "success" if hits else "not_found"
            return AdverseEventsSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.event_term or "",
            )

        if search_input.mode == "compare_safety":
            comparisons: list[AdverseEventDrugComparison] = []
            for name in search_input.drug_names:
                rows = await _fetch_events_for_drug(
                    async_config,
                    name,
                    search_input.severity,
                    search_input.min_subjects_affected,
                    limit=min(20, search_input.limit),
                    offset=0,
                )
                comparisons.append(
                    AdverseEventDrugComparison(
                        drug_name=name,
                        top_events=[AdverseEventSummaryHit(**row) for row in rows],
                    )
                )
            status = "success" if comparisons else "not_found"
            return AdverseEventsSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(comparisons),
                hits=comparisons,
                query_summary=", ".join(search_input.drug_names),
            )

        return AdverseEventsSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="Unsupported mode",
            query_summary="invalid mode",
        )
    except Exception as exc:
        return AdverseEventsSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )
