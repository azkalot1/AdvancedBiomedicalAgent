#!/usr/bin/env python3
"""
STRING DB protein-protein interaction search (human, gene symbols).

Data sources:
    - string_ppi, string_protein_info
    - dm_target, dm_target_gene_synonyms (gene resolution / optional enrichment)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig

EVIDENCE_COLUMNS = {
    "neighborhood": "neighborhood",
    "fusion": "fusion",
    "cooccurence": "cooccurence",
    "cooccurrence": "cooccurence",
    "coexpression": "coexpression",
    "experimental": "experimental",
    "database": "database",
    "textmining": "textmining",
}


class PpiSearchInput(BaseModel):
    mode: Literal["interactions", "pair_detail", "shared_partners", "network"]
    gene: str | None = None
    gene_1: str | None = None
    gene_2: str | None = None
    min_score: int = Field(default=700, ge=0, le=1000)
    evidence_type: str | None = None
    include_target_info: bool = False
    limit: int = Field(default=30, ge=1, le=500)

    @field_validator("gene", "gene_1", "gene_2", "evidence_type", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode in {"interactions", "network"}:
            if not self.gene:
                errors.append(f"{self.mode} requires gene")
        elif self.mode == "pair_detail":
            if not self.gene_1 or not self.gene_2:
                errors.append("pair_detail requires gene_1 and gene_2")
        elif self.mode == "shared_partners":
            if not self.gene_1 or not self.gene_2:
                errors.append("shared_partners requires gene_1 and gene_2")
        if self.evidence_type:
            key = self.evidence_type.lower().strip()
            if key not in EVIDENCE_COLUMNS:
                errors.append(
                    f"evidence_type must be one of: {', '.join(sorted(set(EVIDENCE_COLUMNS.keys())))}"
                )
        return errors


class PpiSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_results: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    query_gene: str | None = None
    resolved_gene_symbol: str | None = None
    resolved_gene_1: str | None = None
    resolved_gene_2: str | None = None
    messages: list[str] = Field(default_factory=list)
    error: str | None = None


def _evidence_filter_sql(evidence_type: str | None, table_alias: str = "p") -> str:
    if not evidence_type:
        return ""
    col = EVIDENCE_COLUMNS.get(evidence_type.lower().strip())
    if not col:
        return ""
    return f" AND {table_alias}.{col} > 0 "


def _target_lateral_sql(include: bool) -> str:
    if not include:
        return ""
    return """
        LEFT JOIN LATERAL (
            SELECT uniprot_id, protein_name, target_id
            FROM dm_target
            WHERE gene_symbol = partner_gene
              AND (organism IS NULL OR organism = 'Homo sapiens')
            ORDER BY target_id
            LIMIT 1
        ) dt ON TRUE
    """


def _target_select(include: bool) -> str:
    if not include:
        return ""
    return ", dt.uniprot_id AS dm_uniprot_id, dt.protein_name AS dm_protein_name, dt.target_id AS dm_target_id"


async def _resolve_ensembl_to_symbol(async_config: AsyncDatabaseConfig, ensembl_id: str) -> str | None:
    row = await async_config.execute_one(
        """
        SELECT gene_symbol FROM dm_target
        WHERE ensembl_gene_id = $1
        LIMIT 1
        """,
        ensembl_id.strip(),
    )
    return row["gene_symbol"] if row else None


async def _resolve_gene_symbol(
    async_config: AsyncDatabaseConfig,
    query: str,
) -> tuple[str | None, list[str]]:
    messages: list[str] = []
    q = query.strip()
    if not q:
        return None, messages

    if q.upper().startswith("ENSG"):
        sym = await _resolve_ensembl_to_symbol(async_config, q)
        if sym:
            messages.append(f"Resolved Ensembl gene '{q}' to symbol '{sym}' via dm_target.")
            return sym, messages
        return None, messages

    row = await async_config.execute_one(
        """
        SELECT gene_symbol FROM string_protein_info WHERE gene_symbol = $1 LIMIT 1
        """,
        q,
    )
    if row:
        return row["gene_symbol"], messages

    rows = await async_config.execute_query(
        """
        SELECT DISTINCT t.gene_symbol
        FROM dm_target_gene_synonyms s
        JOIN dm_target t ON t.target_id = s.target_id
        WHERE UPPER(s.gene_symbol) = UPPER($1)
        LIMIT 1
        """,
        q,
    )
    if rows:
        sym = rows[0]["gene_symbol"]
        messages.append(f"Resolved synonym '{q}' to '{sym}' via dm_target_gene_synonyms.")
        return sym, messages

    row = await async_config.execute_one(
        """
        SELECT gene_symbol FROM string_protein_info
        WHERE gene_symbol ILIKE $1
        ORDER BY gene_symbol
        LIMIT 1
        """,
        f"%{q}%",
    )
    if row:
        sym = row["gene_symbol"]
        messages.append(f"Resolved '{q}' via ILIKE to '{sym}'.")
        return sym, messages

    return None, messages


async def ppi_search_async(
    db_config: DatabaseConfig,
    search_input: PpiSearchInput,
) -> PpiSearchOutput:
    errors = search_input.validate_for_mode()
    if errors:
        return PpiSearchOutput(status="invalid_input", mode=search_input.mode, error="; ".join(errors))

    try:
        async_config = await get_async_connection(db_config)
    except Exception as exc:
        return PpiSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
        )

    mode = search_input.mode
    ev_inner = _evidence_filter_sql(search_input.evidence_type, "p0")
    lat = _target_lateral_sql(search_input.include_target_info)
    tsel = _target_select(search_input.include_target_info)
    messages: list[str] = []

    try:
        if mode in {"interactions", "network"}:
            assert search_input.gene
            g_raw = search_input.gene.strip()
            resolved, msgs = await _resolve_gene_symbol(async_config, g_raw)
            messages.extend(msgs)
            if not resolved:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_gene=g_raw,
                    messages=messages + [f"No gene symbol match for '{g_raw}'."],
                )

            # One row per partner gene: STRING is protein-level; multiple ENSP pairs can map to
            # the same symbols. DISTINCT ON keeps the edge with highest combined_score per partner.
            sql = f"""
                SELECT
                    d.partner_gene,
                    d.combined_score,
                    d.neighborhood, d.fusion, d.cooccurence, d.coexpression,
                    d.experimental, d.database, d.textmining,
                    d.ensp_id_1, d.ensp_id_2, d.gene_symbol_1, d.gene_symbol_2
                    {tsel}
                FROM (
                    SELECT DISTINCT ON (p.partner_gene)
                        p.partner_gene,
                        p.combined_score,
                        p.neighborhood, p.fusion, p.cooccurence, p.coexpression,
                        p.experimental, p.database, p.textmining,
                        p.ensp_id_1, p.ensp_id_2, p.gene_symbol_1, p.gene_symbol_2
                    FROM (
                        SELECT
                            CASE
                                WHEN p0.gene_symbol_1 = $1 THEN p0.gene_symbol_2
                                ELSE p0.gene_symbol_1
                            END AS partner_gene,
                            p0.combined_score,
                            p0.neighborhood, p0.fusion, p0.cooccurence, p0.coexpression,
                            p0.experimental, p0.database, p0.textmining,
                            p0.ensp_id_1, p0.ensp_id_2, p0.gene_symbol_1, p0.gene_symbol_2
                        FROM string_ppi p0
                        WHERE (p0.gene_symbol_1 = $1 OR p0.gene_symbol_2 = $1)
                          AND p0.combined_score >= $2
                          {ev_inner}
                    ) p
                    ORDER BY
                        p.partner_gene,
                        p.combined_score DESC NULLS LAST,
                        p.ensp_id_1,
                        p.ensp_id_2
                ) d
                {lat}
                ORDER BY d.combined_score DESC NULLS LAST, d.partner_gene
                LIMIT $3
            """
            rows = await async_config.execute_query(
                sql, resolved, search_input.min_score, search_input.limit
            )
            if not rows:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_gene=g_raw,
                    resolved_gene_symbol=resolved,
                    messages=messages,
                )
            return PpiSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                query_gene=g_raw,
                resolved_gene_symbol=resolved,
                messages=messages,
            )

        if mode == "pair_detail":
            assert search_input.gene_1 and search_input.gene_2
            r1, m1 = await _resolve_gene_symbol(async_config, search_input.gene_1)
            r2, m2 = await _resolve_gene_symbol(async_config, search_input.gene_2)
            messages.extend(m1 + m2)
            if not r1 or not r2:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=messages + ["Could not resolve one or both genes."],
                )

            ev_p = _evidence_filter_sql(search_input.evidence_type, "p")
            row = await async_config.execute_one(
                f"""
                SELECT
                    p.gene_symbol_1, p.gene_symbol_2, p.ensp_id_1, p.ensp_id_2,
                    p.neighborhood, p.fusion, p.cooccurence, p.coexpression,
                    p.experimental, p.database, p.textmining, p.combined_score
                FROM string_ppi p
                WHERE (
                    (p.gene_symbol_1 = $1 AND p.gene_symbol_2 = $2)
                    OR (p.gene_symbol_1 = $2 AND p.gene_symbol_2 = $1)
                )
                  AND p.combined_score >= $3
                  {ev_p}
                LIMIT 1
                """,
                r1,
                r2,
                search_input.min_score,
            )
            if not row:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    resolved_gene_1=r1,
                    resolved_gene_2=r2,
                    messages=messages,
                )
            if search_input.include_target_info:
                row = dict(row)
                for label, sym in [("a", r1), ("b", r2)]:
                    dt = await async_config.execute_one(
                        """
                        SELECT uniprot_id, protein_name, target_id
                        FROM dm_target
                        WHERE gene_symbol = $1
                          AND (organism IS NULL OR organism = 'Homo sapiens')
                        ORDER BY target_id LIMIT 1
                        """,
                        sym,
                    )
                    if dt:
                        row[f"dm_uniprot_{label}"] = dt.get("uniprot_id")
                        row[f"dm_protein_name_{label}"] = dt.get("protein_name")
            return PpiSearchOutput(
                status="success",
                mode=mode,
                total_results=1,
                results=[dict(row)],
                resolved_gene_1=r1,
                resolved_gene_2=r2,
                messages=messages,
            )

        if mode == "shared_partners":
            assert search_input.gene_1 and search_input.gene_2
            r1, m1 = await _resolve_gene_symbol(async_config, search_input.gene_1)
            r2, m2 = await _resolve_gene_symbol(async_config, search_input.gene_2)
            messages.extend(m1 + m2)
            if not r1 or not r2:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=messages + ["Could not resolve one or both genes."],
                )

            ev_p = _evidence_filter_sql(search_input.evidence_type, "p")
            sql = f"""
                WITH partners1 AS (
                    SELECT DISTINCT
                        CASE WHEN p.gene_symbol_1 = $1 THEN p.gene_symbol_2 ELSE p.gene_symbol_1 END
                        AS partner
                    FROM string_ppi p
                    WHERE (p.gene_symbol_1 = $1 OR p.gene_symbol_2 = $1)
                      AND p.combined_score >= $3
                      {ev_p}
                ),
                partners2 AS (
                    SELECT DISTINCT
                        CASE WHEN p.gene_symbol_1 = $2 THEN p.gene_symbol_2 ELSE p.gene_symbol_1 END
                        AS partner
                    FROM string_ppi p
                    WHERE (p.gene_symbol_1 = $2 OR p.gene_symbol_2 = $2)
                      AND p.combined_score >= $3
                      {ev_p}
                )
                SELECT p1.partner AS partner_gene
                FROM partners1 p1
                INNER JOIN partners2 p2 ON p1.partner = p2.partner
                ORDER BY p1.partner
                LIMIT $4
            """
            rows = await async_config.execute_query(
                sql, r1, r2, search_input.min_score, search_input.limit
            )
            if not rows:
                return PpiSearchOutput(
                    status="not_found",
                    mode=mode,
                    resolved_gene_1=r1,
                    resolved_gene_2=r2,
                    messages=messages,
                )
            return PpiSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                resolved_gene_1=r1,
                resolved_gene_2=r2,
                messages=messages,
            )

        return PpiSearchOutput(status="error", mode=mode, error=f"Unhandled mode: {mode}")

    except Exception as exc:
        return PpiSearchOutput(
            status="error",
            mode=mode,
            error=f"{type(exc).__name__}: {exc}",
        )
