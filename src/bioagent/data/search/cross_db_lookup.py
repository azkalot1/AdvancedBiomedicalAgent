#!/usr/bin/env python3
"""
Cross-database lookup for unified drug identifiers.

Resolves a name or identifier (e.g., ChEMBL ID, InChIKey, NDC, RxCUI)
to molecule records and optionally expands to related labels, trials,
and targets.

Data Sources:
    - dm_molecule / dm_molecule_concept / dm_molecule_synonyms
    - map_product_molecules / labels_meta / dailymed_products
    - map_ctgov_molecules / rag_study_search
    - dm_compound_target_activity
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CHEMBL_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
NCT_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
RXCUI_RE = re.compile(r"^\d{1,10}$")
NDC_RE = re.compile(r"^[0-9-]{4,}$")
UNII_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


class CrossDatabaseLookupInput(BaseModel):
    identifier: str
    identifier_type: Literal["name", "chembl", "inchikey", "ndc", "rxcui", "unii", "auto"] = "auto"
    include_labels: bool = True
    include_trials: bool = True
    include_targets: bool = True
    limit: int = Field(default=20, ge=1, le=200)

    @field_validator("identifier", mode="before")
    @classmethod
    def strip_identifier(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class MoleculeIdentifierHit(BaseModel):
    mol_id: int | None = None
    concept_id: int | None = None
    pref_name: str | None = None
    concept_name: str | None = None
    inchi_key: str | None = None
    canonical_smiles: str | None = None
    chembl_id: str | None = None
    drugcentral_id: int | None = None
    pubchem_cid: int | None = None
    sources: list[str] = Field(default_factory=list)


class LabelHit(BaseModel):
    set_id: str
    source: str
    title: str | None = None


class TrialHit(BaseModel):
    nct_id: str
    brief_title: str | None = None
    phase: str | None = None
    status: str | None = None
    match_type: str | None = None
    confidence: float | None = None


class TargetHit(BaseModel):
    gene_symbol: str
    best_pchembl: float | None = None
    n_measurements: int | None = None


class CrossDatabaseLookupOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    identifier: str
    identifier_type: str
    molecules: list[MoleculeIdentifierHit] = Field(default_factory=list)
    labels: list[LabelHit] = Field(default_factory=list)
    trials: list[TrialHit] = Field(default_factory=list)
    targets: list[TargetHit] = Field(default_factory=list)
    error: str | None = None


def _infer_identifier_type(identifier: str) -> str:
    if INCHIKEY_RE.match(identifier.upper()):
        return "inchikey"
    if identifier.upper().startswith("CHEMBL"):
        return "chembl"
    return "name"


async def _fetch_molecules(
    async_config: AsyncDatabaseConfig,
    identifier: str,
    identifier_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    # Direct identifier lookups
    if identifier_type == "chembl":
        sql = """
            SELECT
                m.mol_id,
                m.concept_id,
                m.pref_name,
                m.inchi_key,
                m.canonical_smiles,
                m.chembl_id,
                m.drugcentral_id,
                m.pubchem_cid,
                m.sources,
                mc.preferred_name AS concept_name,
                100 AS match_score
            FROM dm_molecule m
            LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
            WHERE m.chembl_id = $1

            UNION ALL

            SELECT
                NULL::bigint AS mol_id,
                b.concept_id,
                b.pref_name,
                NULL::text AS inchi_key,
                NULL::text AS canonical_smiles,
                b.chembl_id,
                b.drugcentral_id,
                NULL::integer AS pubchem_cid,
                b.sources,
                mc.preferred_name AS concept_name,
                100 AS match_score
            FROM dm_biotherapeutic b
            LEFT JOIN dm_molecule_concept mc ON b.concept_id = mc.concept_id
            WHERE b.chembl_id = $1

            ORDER BY match_score DESC, concept_name NULLS LAST, pref_name NULLS LAST
            LIMIT $2
        """
        return await async_config.execute_query(sql, identifier.upper(), limit)

    if identifier_type == "inchikey":
        sql = """
            SELECT
                m.mol_id,
                m.concept_id,
                m.pref_name,
                m.inchi_key,
                m.canonical_smiles,
                m.chembl_id,
                m.drugcentral_id,
                m.pubchem_cid,
                m.sources,
                mc.preferred_name AS concept_name,
                100 AS match_score
            FROM dm_molecule m
            LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
            WHERE m.inchi_key = $1
            ORDER BY m.mol_id
            LIMIT $2
        """
        return await async_config.execute_query(sql, identifier.upper(), limit)

    # Name-like lookups: query synonyms/tables directly, then resolve to records.
    # This avoids correlated EXISTS scans and includes biotherapeutics.
    sql = """
        WITH q AS (
            SELECT LOWER($1) AS name_exact, $2::text AS name_pattern
        ),
        name_candidates AS (
            -- Small molecule synonyms (preferred names are also indexed here).
            SELECT s.mol_id, s.concept_id, NULL::bigint AS bio_id,
                   CASE
                       WHEN s.synonym_lower = (SELECT name_exact FROM q) THEN 95
                       WHEN s.synonym ILIKE (SELECT name_pattern FROM q) THEN 80
                       ELSE 70
                   END AS score
            FROM dm_molecule_synonyms s
            WHERE s.synonym_lower = (SELECT name_exact FROM q)
               OR s.synonym ILIKE (SELECT name_pattern FROM q)
               OR s.synonym_lower % (SELECT name_exact FROM q)

            UNION ALL

            -- Biotherapeutic synonyms
            SELECT NULL::bigint AS mol_id, s.concept_id, s.bio_id,
                   CASE
                       WHEN s.synonym_lower = (SELECT name_exact FROM q) THEN 95
                       WHEN s.synonym ILIKE (SELECT name_pattern FROM q) THEN 80
                       ELSE 70
                   END AS score
            FROM dm_biotherapeutic_synonyms s
            WHERE s.synonym_lower = (SELECT name_exact FROM q)
               OR s.synonym ILIKE (SELECT name_pattern FROM q)
               OR s.synonym_lower % (SELECT name_exact FROM q)

            UNION ALL

            -- Lightweight fallback directly on biotherapeutic preferred names.
            -- dm_biotherapeutic is comparatively small, so this avoids misses if
            -- synonym ingestion is incomplete while remaining fast.
            SELECT NULL::bigint AS mol_id, b.concept_id, b.bio_id,
                   CASE
                       WHEN LOWER(b.pref_name) = (SELECT name_exact FROM q) THEN 90
                       WHEN b.pref_name ILIKE (SELECT name_pattern FROM q) THEN 78
                       ELSE 68
                   END AS score
            FROM dm_biotherapeutic b
            WHERE LOWER(b.pref_name) = (SELECT name_exact FROM q)
               OR b.pref_name ILIKE (SELECT name_pattern FROM q)
               OR b.pref_name % (SELECT name_exact FROM q)
        ),
        dedup AS (
            SELECT
                mol_id,
                concept_id,
                bio_id,
                MAX(score) AS best_score
            FROM name_candidates
            GROUP BY mol_id, concept_id, bio_id
            ORDER BY MAX(score) DESC
            LIMIT 600
        ),
        score_stats AS (
            SELECT COALESCE(MAX(best_score), 0) AS max_score
            FROM dedup
        ),
        filtered AS (
            -- If we have a strong exact/near-exact match, suppress weak fuzzy neighbors.
            -- This prevents broad '-zumab' trigram cohorts from outranking true hits.
            SELECT d.*
            FROM dedup d
            CROSS JOIN score_stats s
            WHERE d.best_score >= CASE WHEN s.max_score >= 90 THEN 90 ELSE 70 END
        ),
        resolved AS (
            SELECT
                d.best_score AS match_score,
                m.mol_id,
                m.concept_id,
                m.pref_name,
                m.inchi_key,
                m.canonical_smiles,
                m.chembl_id,
                m.drugcentral_id,
                m.pubchem_cid,
                m.sources,
                mc.preferred_name AS concept_name
            FROM filtered d
            JOIN dm_molecule m ON d.mol_id = m.mol_id
            LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id

            UNION ALL

            SELECT
                d.best_score AS match_score,
                NULL::bigint AS mol_id,
                b.concept_id,
                b.pref_name,
                NULL::text AS inchi_key,
                NULL::text AS canonical_smiles,
                b.chembl_id,
                b.drugcentral_id,
                NULL::integer AS pubchem_cid,
                b.sources,
                mc.preferred_name AS concept_name
            FROM filtered d
            JOIN dm_biotherapeutic b ON d.bio_id = b.bio_id
            LEFT JOIN dm_molecule_concept mc ON b.concept_id = mc.concept_id
        ),
        dedup_resolved AS (
            SELECT DISTINCT ON (
                COALESCE(concept_id, -1),
                COALESCE(chembl_id, ''),
                COALESCE(pref_name, '')
            )
                mol_id,
                concept_id,
                pref_name,
                inchi_key,
                canonical_smiles,
                chembl_id,
                drugcentral_id,
                pubchem_cid,
                sources,
                concept_name,
                match_score
            FROM resolved
            ORDER BY
                COALESCE(concept_id, -1),
                COALESCE(chembl_id, ''),
                COALESCE(pref_name, ''),
                match_score DESC
        )
        SELECT
            mol_id, concept_id, pref_name, inchi_key, canonical_smiles,
            chembl_id, drugcentral_id, pubchem_cid, sources, concept_name
        FROM dedup_resolved
        ORDER BY
            match_score DESC,
            concept_name NULLS LAST,
            pref_name NULLS LAST
        LIMIT $3
    """
    return await async_config.execute_query(sql, identifier, f"%{identifier}%", limit)


async def cross_database_lookup_async(
    db_config: DatabaseConfig,
    search_input: CrossDatabaseLookupInput,
) -> CrossDatabaseLookupOutput:
    """
    Resolve a drug identifier across internal databases.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: CrossDatabaseLookupInput specifying identifier and options.

    Returns:
        CrossDatabaseLookupOutput with molecules, labels, trials, and targets.
    """
    if not search_input.identifier:
        return CrossDatabaseLookupOutput(
            status="invalid_input",
            identifier="",
            identifier_type=search_input.identifier_type,
            error="identifier is required",
        )

    try:
        async_config = await get_async_connection(db_config)
        identifier_type = search_input.identifier_type
        if identifier_type == "auto":
            identifier_type = _infer_identifier_type(search_input.identifier)

        identifier_value = search_input.identifier.strip()
        if identifier_type == "chembl" and not CHEMBL_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="identifier_type='chembl' requires a CHEMBL ID (e.g., CHEMBL941)",
            )
        if identifier_type == "inchikey" and not INCHIKEY_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="identifier_type='inchikey' requires a valid InChIKey (e.g., BSYNRYMUTXBXSQ-UHFFFAOYSA-N)",
            )
        if identifier_type == "rxcui" and not RXCUI_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="identifier_type='rxcui' requires a numeric RxCUI",
            )
        if identifier_type == "ndc" and not NDC_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="identifier_type='ndc' requires a valid NDC format (digits and hyphens)",
            )
        if identifier_type == "unii" and not UNII_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="identifier_type='unii' requires a 10-character UNII code",
            )
        if identifier_type == "name" and identifier_value.upper().startswith("NCT") and not NCT_RE.match(identifier_value):
            return CrossDatabaseLookupOutput(
                status="invalid_input",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
                error="NCT identifiers must match NCT######## format (e.g., NCT01295827)",
            )

        molecule_rows = await _fetch_molecules(
            async_config,
            search_input.identifier,
            identifier_type,
            search_input.limit,
        )
        molecules = [MoleculeIdentifierHit(
            mol_id=row.get("mol_id"),
            concept_id=row.get("concept_id"),
            pref_name=row.get("pref_name"),
            concept_name=row.get("concept_name"),
            inchi_key=row.get("inchi_key"),
            canonical_smiles=row.get("canonical_smiles"),
            chembl_id=row.get("chembl_id"),
            drugcentral_id=row.get("drugcentral_id"),
            pubchem_cid=row.get("pubchem_cid"),
            sources=row.get("sources") or [],
        ) for row in molecule_rows]

        if not molecules:
            return CrossDatabaseLookupOutput(
                status="not_found",
                identifier=search_input.identifier,
                identifier_type=identifier_type,
            )

        mol_ids = [m.mol_id for m in molecules if m.mol_id is not None]
        concept_ids = [m.concept_id for m in molecules if m.concept_id is not None]

        labels: list[LabelHit] = []
        if search_input.include_labels and (mol_ids or concept_ids):
            label_rows = await async_config.execute_query(
                """
                SELECT map.set_id, map.source_table,
                       COALESCE(lm.title, dp.product_name) AS title
                FROM map_product_molecules map
                LEFT JOIN set_ids si ON si.set_id = map.set_id
                LEFT JOIN labels_meta lm ON lm.set_id_id = si.id
                LEFT JOIN dailymed_products dp ON dp.set_id = map.set_id
                WHERE map.mol_id = ANY($1) OR map.concept_id = ANY($2)
                LIMIT 200
                """,
                mol_ids or [0],
                concept_ids or [0],
            )
            labels = [LabelHit(
                set_id=row["set_id"],
                source=row.get("source_table") or "unknown",
                title=row.get("title"),
            ) for row in label_rows]

        trials: list[TrialHit] = []
        if search_input.include_trials and (mol_ids or concept_ids):
            trial_rows = await async_config.execute_query(
                """
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence
                FROM map_ctgov_molecules map
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                WHERE map.mol_id = ANY($1) OR map.concept_id = ANY($2)
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT 200
                """,
                mol_ids or [0],
                concept_ids or [0],
            )
            trials = [TrialHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                match_type=row.get("match_type"),
                confidence=row.get("confidence"),
            ) for row in trial_rows]

        targets: list[TargetHit] = []
        if search_input.include_targets and concept_ids:
            target_rows = await async_config.execute_query(
                """
                SELECT gene_symbol,
                       MAX(pchembl_value) AS best_pchembl,
                       COUNT(*)::int AS n_measurements
                FROM dm_compound_target_activity
                WHERE concept_id = ANY($1)
                  AND gene_symbol IS NOT NULL
                GROUP BY gene_symbol
                ORDER BY best_pchembl DESC NULLS LAST
                LIMIT 200
                """,
                concept_ids,
            )
            targets = [TargetHit(
                gene_symbol=row["gene_symbol"],
                best_pchembl=row.get("best_pchembl"),
                n_measurements=row.get("n_measurements"),
            ) for row in target_rows]

        return CrossDatabaseLookupOutput(
            status="success",
            identifier=search_input.identifier,
            identifier_type=identifier_type,
            molecules=molecules,
            labels=labels,
            trials=trials,
            targets=targets,
        )
    except Exception as exc:
        return CrossDatabaseLookupOutput(
            status="error",
            identifier=search_input.identifier,
            identifier_type=search_input.identifier_type,
            error=f"{type(exc).__name__}: {exc}",
        )
