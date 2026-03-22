#!/usr/bin/env python3
"""
Human Protein Atlas single-cell RNA expression search (by gene and cell type).

Data sources:
    - hpa_rna_cell_type_expression
    - hpa_cell_type_summary
    - dm_target, dm_target_gene_synonyms (optional enrichment / gene resolution)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


class HpaExpressionSearchInput(BaseModel):
    mode: Literal["gene_expression", "cell_type_genes", "list_cell_types", "compare_expression"]
    gene_symbol: str | None = None
    ensembl_gene_id: str | None = None
    cell_type: str | None = None
    cell_types: list[str] | None = None
    min_ncpm: float | None = None
    include_target_info: bool = False
    limit: int = Field(default=50, ge=1, le=500)

    @field_validator("gene_symbol", "ensembl_gene_id", "cell_type", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if self.mode == "gene_expression":
            if not self.gene_symbol and not self.ensembl_gene_id:
                errors.append("gene_expression requires gene_symbol or ensembl_gene_id")
        elif self.mode == "cell_type_genes":
            if not self.cell_type:
                errors.append("cell_type_genes requires cell_type")
        elif self.mode == "list_cell_types":
            pass
        elif self.mode == "compare_expression":
            if not self.gene_symbol and not self.ensembl_gene_id:
                errors.append("compare_expression requires gene_symbol or ensembl_gene_id")
        return errors


class HpaExpressionSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_results: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    query_gene: str | None = None
    query_cell_type: str | None = None
    resolved_gene_symbol: str | None = None
    resolved_ensembl_gene_id: str | None = None
    messages: list[str] = Field(default_factory=list)
    error: str | None = None


async def _resolve_gene_symbol(
    async_config: AsyncDatabaseConfig,
    query: str,
) -> tuple[str | None, str | None, list[str]]:
    """
    Returns (canonical_gene_symbol, ensembl_from_hpa_if_any, messages).
    Resolution order: exact HPA symbol, synonym -> dm_target, ILIKE prefix on HPA.
    """
    messages: list[str] = []
    q = query.strip()
    if not q:
        return None, None, messages

    rows = await async_config.execute_query(
        """
        SELECT DISTINCT gene_symbol, ensembl_gene_id
        FROM hpa_rna_cell_type_expression
        WHERE gene_symbol = $1
        LIMIT 1
        """,
        q,
    )
    if rows:
        return rows[0]["gene_symbol"], rows[0].get("ensembl_gene_id"), messages

    rows = await async_config.execute_query(
        """
        SELECT DISTINCT t.gene_symbol,
               (SELECT e.ensembl_gene_id FROM hpa_rna_cell_type_expression e
                WHERE e.gene_symbol = t.gene_symbol LIMIT 1) AS ensembl_gene_id
        FROM dm_target_gene_synonyms s
        JOIN dm_target t ON t.target_id = s.target_id
        WHERE UPPER(s.gene_symbol) = UPPER($1)
        LIMIT 1
        """,
        q,
    )
    if rows:
        sym = rows[0]["gene_symbol"]
        ens = rows[0].get("ensembl_gene_id")
        messages.append(f"Resolved synonym '{q}' to canonical gene_symbol '{sym}' via dm_target_gene_synonyms.")
        return sym, ens, messages

    rows = await async_config.execute_query(
        """
        SELECT DISTINCT gene_symbol, ensembl_gene_id
        FROM hpa_rna_cell_type_expression
        WHERE gene_symbol ILIKE $1
        ORDER BY gene_symbol
        LIMIT 1
        """,
        f"%{q}%",
    )
    if rows:
        sym = rows[0]["gene_symbol"]
        messages.append(f"Resolved '{q}' via ILIKE match to gene_symbol '{sym}'.")
        return sym, rows[0].get("ensembl_gene_id"), messages

    return None, None, messages


async def _resolve_ensembl_to_symbol(
    async_config: AsyncDatabaseConfig,
    ensembl_id: str,
) -> str | None:
    row = await async_config.execute_one(
        """
        SELECT gene_symbol FROM hpa_rna_cell_type_expression
        WHERE ensembl_gene_id = $1
        LIMIT 1
        """,
        ensembl_id.strip(),
    )
    return row["gene_symbol"] if row else None


async def _resolve_cell_type(
    async_config: AsyncDatabaseConfig,
    query: str,
) -> tuple[str | None, list[str]]:
    messages: list[str] = []
    q = query.strip()
    if not q:
        return None, messages

    row = await async_config.execute_one(
        "SELECT cell_type FROM hpa_cell_type_summary WHERE cell_type = $1",
        q,
    )
    if row:
        return row["cell_type"], messages

    row = await async_config.execute_one(
        """
        SELECT cell_type
        FROM hpa_cell_type_summary
        WHERE cell_type ILIKE $1
        ORDER BY similarity(cell_type, $2) DESC NULLS LAST
        LIMIT 1
        """,
        f"%{q}%",
        q,
    )
    if row:
        ct = row["cell_type"]
        messages.append(f"Resolved cell type '{q}' to '{ct}' (fuzzy/ILIKE).")
        return ct, messages

    row = await async_config.execute_one(
        """
        SELECT cell_type
        FROM hpa_cell_type_summary
        WHERE similarity(cell_type, $1) > 0.15
        ORDER BY similarity(cell_type, $1) DESC NULLS LAST
        LIMIT 1
        """,
        q,
    )
    if row and row.get("cell_type"):
        ct = row["cell_type"]
        messages.append(f"Best-effort cell type match for '{q}': '{ct}'.")
        return ct, messages

    return None, messages


def _target_lateral_join_sql(include_target_info: bool) -> str:
    if not include_target_info:
        return ""
    return """
        LEFT JOIN LATERAL (
            SELECT uniprot_id, protein_name, target_id
            FROM dm_target
            WHERE gene_symbol = h.gene_symbol
              AND (organism IS NULL OR organism = 'Homo sapiens')
            ORDER BY target_id
            LIMIT 1
        ) dt ON TRUE
    """


def _target_select_extra(include_target_info: bool) -> str:
    if not include_target_info:
        return ""
    return ", dt.uniprot_id AS dm_uniprot_id, dt.protein_name AS dm_protein_name, dt.target_id AS dm_target_id"


async def hpa_expression_search_async(
    db_config: DatabaseConfig,
    search_input: HpaExpressionSearchInput,
) -> HpaExpressionSearchOutput:
    errors = search_input.validate_for_mode()
    if errors:
        return HpaExpressionSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
        )

    try:
        async_config = await get_async_connection(db_config)
    except Exception as exc:
        return HpaExpressionSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
        )

    mode = search_input.mode
    messages: list[str] = []
    resolved_sym: str | None = None
    resolved_ens: str | None = None

    try:
        if mode == "list_cell_types":
            rows = await async_config.execute_query(
                """
                SELECT cell_type, gene_count, avg_ncpm, max_ncpm, top_gene_symbol
                FROM hpa_cell_type_summary
                ORDER BY cell_type
                """
            )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    total_results=0,
                    messages=["hpa_cell_type_summary is empty; run HPA RNA ingestion."],
                )
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
            )

        if mode in ("gene_expression", "compare_expression"):
            if search_input.ensembl_gene_id:
                resolved_ens = search_input.ensembl_gene_id.strip()
                resolved_sym = await _resolve_ensembl_to_symbol(async_config, resolved_ens)
                if not resolved_sym:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_gene=resolved_ens,
                        resolved_ensembl_gene_id=resolved_ens,
                        messages=[f"No HPA rows for ensembl_gene_id '{resolved_ens}'."],
                    )
            elif search_input.gene_symbol:
                resolved_sym, resolved_ens, msgs = await _resolve_gene_symbol(
                    async_config, search_input.gene_symbol
                )
                messages.extend(msgs)
                if not resolved_sym:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_gene=search_input.gene_symbol,
                        messages=messages + [f"No HPA or synonym match for gene '{search_input.gene_symbol}'."],
                    )

        join_sql = _target_lateral_join_sql(search_input.include_target_info)
        extra_cols = _target_select_extra(search_input.include_target_info)

        if mode == "gene_expression":
            assert resolved_sym is not None
            if search_input.min_ncpm is not None:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1 AND h.ncpm >= $2
                    ORDER BY h.ncpm DESC NULLS LAST, h.cell_type
                    LIMIT $3
                    """,
                    resolved_sym,
                    search_input.min_ncpm,
                    search_input.limit,
                )
            else:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1
                    ORDER BY h.ncpm DESC NULLS LAST, h.cell_type
                    LIMIT $2
                    """,
                    resolved_sym,
                    search_input.limit,
                )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_gene=search_input.gene_symbol or search_input.ensembl_gene_id,
                    resolved_gene_symbol=resolved_sym,
                    resolved_ensembl_gene_id=resolved_ens,
                    messages=messages,
                )
            ens_out = resolved_ens or rows[0].get("ensembl_gene_id")
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                query_gene=search_input.gene_symbol or search_input.ensembl_gene_id,
                resolved_gene_symbol=resolved_sym,
                resolved_ensembl_gene_id=ens_out,
                messages=messages,
            )

        if mode == "cell_type_genes":
            assert search_input.cell_type
            ct, ct_msgs = await _resolve_cell_type(async_config, search_input.cell_type)
            messages.extend(ct_msgs)
            if not ct:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_cell_type=search_input.cell_type,
                    messages=messages + ["Could not resolve cell type."],
                )
            if search_input.min_ncpm is not None:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.gene_symbol, h.ensembl_gene_id, h.ncpm, h.cell_type
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.cell_type = $1 AND h.ncpm >= $2
                    ORDER BY h.ncpm DESC NULLS LAST, h.gene_symbol
                    LIMIT $3
                    """,
                    ct,
                    search_input.min_ncpm,
                    search_input.limit,
                )
            else:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.gene_symbol, h.ensembl_gene_id, h.ncpm, h.cell_type
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.cell_type = $1
                    ORDER BY h.ncpm DESC NULLS LAST, h.gene_symbol
                    LIMIT $2
                    """,
                    ct,
                    search_input.limit,
                )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_cell_type=ct,
                    messages=messages,
                )
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                query_cell_type=ct,
                messages=messages,
            )

        if mode == "compare_expression":
            assert resolved_sym is not None
            cleaned_types = (
                [c.strip() for c in (search_input.cell_types or []) if c and str(c).strip()]
            )
            has_types = bool(cleaned_types)
            has_min = search_input.min_ncpm is not None

            if has_types and has_min:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1
                      AND h.cell_type = ANY($2::text[])
                      AND h.ncpm >= $3
                    ORDER BY h.cell_type
                    LIMIT $4
                    """,
                    resolved_sym,
                    cleaned_types,
                    search_input.min_ncpm,
                    search_input.limit,
                )
            elif has_types:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1
                      AND h.cell_type = ANY($2::text[])
                    ORDER BY h.cell_type
                    LIMIT $3
                    """,
                    resolved_sym,
                    cleaned_types,
                    search_input.limit,
                )
            elif has_min:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1 AND h.ncpm >= $2
                    ORDER BY h.cell_type
                    LIMIT $3
                    """,
                    resolved_sym,
                    search_input.min_ncpm,
                    search_input.limit,
                )
            else:
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    WHERE h.gene_symbol = $1
                    ORDER BY h.cell_type
                    LIMIT $2
                    """,
                    resolved_sym,
                    search_input.limit,
                )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    query_gene=search_input.gene_symbol or search_input.ensembl_gene_id,
                    resolved_gene_symbol=resolved_sym,
                    messages=messages,
                )
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                query_gene=search_input.gene_symbol or search_input.ensembl_gene_id,
                resolved_gene_symbol=resolved_sym,
                resolved_ensembl_gene_id=resolved_ens or rows[0].get("ensembl_gene_id"),
                messages=messages,
            )

        return HpaExpressionSearchOutput(
            status="error",
            mode=mode,
            error=f"Unhandled mode: {mode}",
        )

    except Exception as exc:
        return HpaExpressionSearchOutput(
            status="error",
            mode=mode,
            error=f"{type(exc).__name__}: {exc}",
        )
