#!/usr/bin/env python3
"""
Molecule <-> Clinical Trials Connectivity Search.

Links small molecules to clinical trials using mapping tables and
concept identifiers. Supports molecule-centric, condition-centric,
and target-centric queries.

Modes:
    - trials_by_molecule: find trials linked to a molecule name or InChIKey
    - molecules_by_condition: find molecules associated with a condition
    - trials_by_target: find trials linked to a target gene

Data Sources:
    - map_ctgov_molecules
    - dm_molecule / dm_molecule_concept / dm_molecule_synonyms
    - rag_study_search
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


class MoleculeTrialSearchInput(BaseModel):
    mode: Literal["trials_by_molecule", "molecules_by_condition", "trials_by_target"]
    molecule_name: str | None = None
    inchi_key: str | None = None
    target_gene: str | None = None
    condition: str | None = None
    min_pchembl: float = Field(default=6.0, ge=0.0, le=15.0)
    phase: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("molecule_name", "inchi_key", "target_gene", "condition", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode == "trials_by_molecule":
            if not self.molecule_name and not self.inchi_key:
                errors.append("'molecule_name' or 'inchi_key' is required")
        elif self.mode == "molecules_by_condition":
            if not self.condition:
                errors.append("'condition' is required")
        elif self.mode == "trials_by_target":
            if not self.target_gene:
                errors.append("'target_gene' is required")
        return errors


class TrialByMoleculeHit(BaseModel):
    nct_id: str
    brief_title: str | None = None
    phase: str | None = None
    status: str | None = None
    match_type: str | None = None
    confidence: float | None = None
    mol_id: int | None = None
    concept_id: int | None = None
    concept_name: str | None = None
    molecule_name: str | None = None
    inchi_key: str | None = None
    canonical_smiles: str | None = None


class MoleculeByConditionHit(BaseModel):
    concept_id: int | None = None
    concept_name: str | None = None
    n_trials: int = 0


class TrialByTargetHit(BaseModel):
    nct_id: str
    brief_title: str | None = None
    phase: str | None = None
    status: str | None = None
    concept_id: int | None = None
    concept_name: str | None = None
    match_type: str | None = None
    confidence: float | None = None


class MoleculeTrialSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    error: str | None = None
    query_summary: str = ""


async def _find_molecule_ids(
    async_config: AsyncDatabaseConfig,
    molecule_name: str | None,
    inchi_key: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_parts: list[str] = []

    if inchi_key:
        params.append(inchi_key)
        where_parts.append("m.inchi_key = $1")

    if molecule_name:
        name_param = molecule_name.lower()
        params.append(name_param)
        name_param_idx = len(params)  # Index of name_param (1-based for SQL)
        params.append(f"%{molecule_name}%")
        pattern_param_idx = len(params)  # Index of pattern parameter (1-based for SQL)
        where_parts.append(
            f"""(
                m.pref_name ILIKE ${pattern_param_idx}
                OR mc.preferred_name ILIKE ${pattern_param_idx}
                OR EXISTS (
                    SELECT 1 FROM dm_molecule_synonyms s
                    WHERE s.mol_id = m.mol_id AND s.synonym_lower % ${name_param_idx}
                )
            )"""
        )

    if not where_parts:
        return []

    params.append(limit)
    sql = f"""
        SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
               m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
        FROM dm_molecule m
        LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
        WHERE {" OR ".join(where_parts)}
        ORDER BY m.mol_id
        LIMIT ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def molecule_trial_search_async(
    db_config: DatabaseConfig,
    search_input: MoleculeTrialSearchInput,
) -> MoleculeTrialSearchOutput:
    """
    Search trial connectivity for molecules, conditions, and targets.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: MoleculeTrialSearchInput with mode and filters.

    Returns:
        MoleculeTrialSearchOutput with status, hits, and query summary.
    """
    errors = search_input.validate_for_mode()
    if errors:
        return MoleculeTrialSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
            query_summary="invalid input",
        )

    try:
        async_config = await get_async_connection(db_config)

        if search_input.mode == "trials_by_molecule":
            mol_rows = await _find_molecule_ids(
                async_config,
                search_input.molecule_name,
                search_input.inchi_key,
                limit=min(200, search_input.limit * 5),
            )
            if not mol_rows:
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=search_input.molecule_name or search_input.inchi_key or "",
                )

            mol_ids = [row["mol_id"] for row in mol_rows if row.get("mol_id") is not None]
            concept_ids = [row["concept_id"] for row in mol_rows if row.get("concept_id") is not None]
            params: list[Any] = [mol_ids, concept_ids, search_input.limit, search_input.offset]

            sql = """
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.mol_id, map.concept_id,
                       mc.preferred_name AS concept_name, m.pref_name AS molecule_name,
                       m.inchi_key, m.canonical_smiles
                FROM map_ctgov_molecules map
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                LEFT JOIN dm_molecule m ON map.mol_id = m.mol_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE (map.mol_id = ANY($1) OR map.concept_id = ANY($2))
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT $3 OFFSET $4
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [TrialByMoleculeHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                match_type=row.get("match_type"),
                confidence=row.get("confidence"),
                mol_id=row.get("mol_id"),
                concept_id=row.get("concept_id"),
                concept_name=row.get("concept_name"),
                molecule_name=row.get("molecule_name"),
                inchi_key=row.get("inchi_key"),
                canonical_smiles=row.get("canonical_smiles"),
            ) for row in rows]
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.molecule_name or search_input.inchi_key or "",
            )

        if search_input.mode == "molecules_by_condition":
            condition = search_input.condition or ""
            condition_norm = condition.lower()
            tsquery = _sanitize_tsquery(condition)
            params: list[Any] = [condition_norm, f"%{condition}%", tsquery, search_input.limit, search_input.offset]
            sql = """
                WITH matched_trials AS (
                    SELECT nct_id
                    FROM rag_study_search
                    WHERE conditions_norm % $1
                       OR conditions_norm ILIKE $2
                       OR terms_tsv @@ to_tsquery('english', $3)
                )
                SELECT mc.concept_id, mc.preferred_name AS concept_name,
                       COUNT(DISTINCT map.nct_id)::int AS n_trials
                FROM map_ctgov_molecules map
                JOIN matched_trials mt ON mt.nct_id = map.nct_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                GROUP BY mc.concept_id, mc.preferred_name
                ORDER BY n_trials DESC, mc.preferred_name
                LIMIT $4 OFFSET $5
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [MoleculeByConditionHit(
                concept_id=row.get("concept_id"),
                concept_name=row.get("concept_name"),
                n_trials=row.get("n_trials") or 0,
            ) for row in rows]
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.condition or "",
            )

        if search_input.mode == "trials_by_target":
            target_gene = (search_input.target_gene or "").upper()
            params: list[Any] = [target_gene, search_input.min_pchembl]
            phase_filter = ""
            if search_input.phase:
                params.append(search_input.phase)
                phase_filter = f"AND rs.phase = ANY(${len(params)}::text[])"

            params.extend([search_input.limit, search_input.offset])
            sql = f"""
                WITH target_concepts AS (
                    SELECT DISTINCT concept_id
                    FROM dm_compound_target_activity
                    WHERE gene_symbol = $1
                      AND (pchembl_value IS NULL OR pchembl_value >= $2)
                      AND concept_id IS NOT NULL
                )
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.concept_id,
                       mc.preferred_name AS concept_name
                FROM target_concepts tc
                JOIN map_ctgov_molecules map ON map.concept_id = tc.concept_id
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE 1=1 {phase_filter}
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT ${len(params)-1} OFFSET ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [TrialByTargetHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                match_type=row.get("match_type"),
                confidence=row.get("confidence"),
                concept_id=row.get("concept_id"),
                concept_name=row.get("concept_name"),
            ) for row in rows]
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.target_gene or "",
            )

        return MoleculeTrialSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="Unsupported mode",
            query_summary="invalid mode",
        )
    except Exception as exc:
        return MoleculeTrialSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )
