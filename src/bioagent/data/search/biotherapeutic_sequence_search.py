#!/usr/bin/env python3
"""
Biotherapeutic sequence search.

Supports sequence motif lookup and target-gene mapping for biologics
using UniProt accessions. Intended for antibody/enzyme biologics with
sequence motifs or target-linked annotations.

Modes:
    - by_sequence: match biotherapeutic components by sequence motif
    - similar_biologics: motif-based similarity lookup
    - by_target: map target gene to biotherapeutic components

Data Sources:
    - dm_biotherapeutic
    - dm_biotherapeutic_component
    - dm_target / dm_target_uniprot_mappings
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


class BiotherapeuticSearchInput(BaseModel):
    mode: Literal["by_sequence", "by_target", "similar_biologics"]
    sequence: str | None = None
    target_gene: str | None = None
    biotherapeutic_type: Literal["antibody", "enzyme", "all"] = "all"
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("sequence", "target_gene", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode in ("by_sequence", "similar_biologics") and not self.sequence:
            errors.append("'sequence' is required for sequence-based search")
        if self.mode == "by_target" and not self.target_gene:
            errors.append("'target_gene' is required for by_target")
        return errors


class BiotherapeuticComponentHit(BaseModel):
    component_id: int | None = None
    component_type: str | None = None
    description: str | None = None
    sequence_length: int | None = None
    uniprot_accession: str | None = None


class BiotherapeuticHit(BaseModel):
    bio_id: int | None = None
    concept_id: int | None = None
    chembl_id: str | None = None
    pref_name: str | None = None
    molecule_type: str | None = None
    biotherapeutic_type: str | None = None
    organism: str | None = None
    components: list[BiotherapeuticComponentHit] = Field(default_factory=list)


class BiotherapeuticSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    error: str | None = None
    query_summary: str = ""


def _bio_type_filter(bio_type: str, param_idx: int) -> tuple[str, list[Any]]:
    if bio_type == "all":
        return "", []
    return f"AND LOWER(b.biotherapeutic_type) = ${param_idx}", [bio_type.lower()]


async def _fetch_by_sequence(
    async_config: AsyncDatabaseConfig,
    sequence: str,
    bio_type: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    seq_pattern = f"%{sequence}%"
    params: list[Any] = [seq_pattern]
    type_clause, type_params = _bio_type_filter(bio_type, 2)
    params.extend(type_params)
    params.extend([limit, offset])
    sql = f"""
        SELECT b.bio_id, b.concept_id, b.chembl_id, b.pref_name, b.molecule_type, b.biotherapeutic_type, b.organism,
               c.component_id, c.component_type, c.description, c.sequence_length, c.uniprot_accession
        FROM dm_biotherapeutic b
        JOIN dm_biotherapeutic_component c ON b.bio_id = c.bio_id
        WHERE c.sequence ILIKE $1
        {type_clause}
        ORDER BY b.pref_name
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def _fetch_by_target(
    async_config: AsyncDatabaseConfig,
    target_gene: str,
    bio_type: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    params: list[Any] = [target_gene.upper()]
    type_clause, type_params = _bio_type_filter(bio_type, 2)
    params.extend(type_params)
    params.extend([limit, offset])
    sql = f"""
        SELECT b.bio_id, b.concept_id, b.chembl_id, b.pref_name, b.molecule_type, b.biotherapeutic_type, b.organism,
               c.component_id, c.component_type, c.description, c.sequence_length, c.uniprot_accession
        FROM dm_target t
        JOIN dm_target_uniprot_mappings um ON um.target_id = t.target_id
        JOIN dm_biotherapeutic_component c ON c.uniprot_accession = um.uniprot_accession
        JOIN dm_biotherapeutic b ON b.bio_id = c.bio_id
        WHERE t.gene_symbol = $1
        {type_clause}
        ORDER BY b.pref_name
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def biotherapeutic_sequence_search_async(
    db_config: DatabaseConfig,
    search_input: BiotherapeuticSearchInput,
) -> BiotherapeuticSearchOutput:
    """
    Search biotherapeutics by sequence motif or target gene.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: BiotherapeuticSearchInput with mode and filters.

    Returns:
        BiotherapeuticSearchOutput with status, hits, and query summary.
    """
    errors = search_input.validate_for_mode()
    if errors:
        return BiotherapeuticSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
            query_summary="invalid input",
        )

    try:
        async_config = await get_async_connection(db_config)

        if search_input.mode in ("by_sequence", "similar_biologics"):
            rows = await _fetch_by_sequence(
                async_config,
                search_input.sequence or "",
                search_input.biotherapeutic_type,
                search_input.limit,
                search_input.offset,
            )
        else:
            rows = await _fetch_by_target(
                async_config,
                search_input.target_gene or "",
                search_input.biotherapeutic_type,
                search_input.limit,
                search_input.offset,
            )

        if not rows:
            return BiotherapeuticSearchOutput(
                status="not_found",
                mode=search_input.mode,
                query_summary=search_input.sequence or search_input.target_gene or "",
            )

        grouped: dict[int, BiotherapeuticHit] = {}
        for row in rows:
            bio_id = row.get("bio_id")
            if bio_id is None:
                continue
            if bio_id not in grouped:
                grouped[bio_id] = BiotherapeuticHit(
                    bio_id=bio_id,
                    concept_id=row.get("concept_id"),
                    chembl_id=row.get("chembl_id"),
                    pref_name=row.get("pref_name"),
                    molecule_type=row.get("molecule_type"),
                    biotherapeutic_type=row.get("biotherapeutic_type"),
                    organism=row.get("organism"),
                )
            grouped[bio_id].components.append(
                BiotherapeuticComponentHit(
                    component_id=row.get("component_id"),
                    component_type=row.get("component_type"),
                    description=row.get("description"),
                    sequence_length=row.get("sequence_length"),
                    uniprot_accession=row.get("uniprot_accession"),
                )
            )

        hits = list(grouped.values())
        return BiotherapeuticSearchOutput(
            status="success",
            mode=search_input.mode,
            total_hits=len(hits),
            hits=hits,
            query_summary=search_input.sequence or search_input.target_gene or "",
        )
    except Exception as exc:
        return BiotherapeuticSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )
