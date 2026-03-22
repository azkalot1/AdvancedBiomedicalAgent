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
    mode: Literal[
        "gene_expression",
        "cell_type_genes",
        "list_cell_types",
        "compare_expression",
        "compare_genes",
        "compare_cell_types",
    ]
    gene_symbol: str | None = None
    ensembl_gene_id: str | None = None
    gene_symbols: list[str] | None = None
    cell_type: str | None = None
    cell_types: list[str] | None = None
    gene_1: str | None = None
    gene_2: str | None = None
    cell_type_1: str | None = None
    cell_type_2: str | None = None
    min_ncpm: float | None = None
    include_target_info: bool = False
    limit: int = Field(default=50, ge=1, le=500)

    @field_validator("gene_symbol", "ensembl_gene_id", "cell_type", "gene_1", "gene_2", "cell_type_1", "cell_type_2", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    @field_validator("gene_symbols", "cell_types", mode="before")
    @classmethod
    def strip_string_lists(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out or None

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        has_single_gene = bool(self.gene_symbol or self.ensembl_gene_id)
        has_gene_batch = bool(self.gene_symbols)
        if self.mode == "gene_expression":
            if not has_single_gene and not has_gene_batch:
                errors.append(
                    "gene_expression requires gene_symbol, ensembl_gene_id, or non-empty gene_symbols"
                )
        elif self.mode == "cell_type_genes":
            if not self.cell_type and not self.cell_types:
                errors.append("cell_type_genes requires cell_type or non-empty cell_types")
        elif self.mode == "list_cell_types":
            pass
        elif self.mode == "compare_expression":
            if not has_single_gene and not has_gene_batch:
                errors.append(
                    "compare_expression requires gene_symbol, ensembl_gene_id, or non-empty gene_symbols"
                )
        elif self.mode == "compare_genes":
            if not self.gene_1 or not self.gene_2:
                errors.append("compare_genes requires gene_1 and gene_2")
        elif self.mode == "compare_cell_types":
            if not self.cell_type_1 or not self.cell_type_2:
                errors.append("compare_cell_types requires cell_type_1 and cell_type_2")
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
    resolved_genes: list[str] | None = None
    query_genes: list[str] | None = None
    resolved_cell_types: list[str] | None = None
    compare_gene_1: str | None = None
    compare_gene_2: str | None = None
    compare_cell_type_1: str | None = None
    compare_cell_type_2: str | None = None
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


def _ncpm_fold_change(a: float | None, b: float | None) -> float:
    """max(a,b) / max(min(a,b), 0.01) for display; handles missing as 0."""
    x = float(a) if a is not None else 0.0
    y = float(b) if b is not None else 0.0
    hi = max(x, y)
    lo = max(min(x, y), 0.01)
    return hi / lo


async def _resolve_one_gene_token(
    async_config: AsyncDatabaseConfig,
    token: str,
) -> tuple[str | None, str | None, list[str]]:
    """Resolve a single user token (ENSG or symbol) to (gene_symbol, ensembl_if_any, messages)."""
    t = token.strip()
    if not t:
        return None, None, []
    if t.upper().startswith("ENSG"):
        sym = await _resolve_ensembl_to_symbol(async_config, t)
        if sym:
            return sym, t, []
        return None, None, [f"No HPA rows for ensembl_gene_id '{t}'."]
    sym, ens, msgs = await _resolve_gene_symbol(async_config, t)
    return sym, ens, msgs


async def _resolve_gene_list(
    async_config: AsyncDatabaseConfig,
    raw_genes: list[str],
) -> tuple[list[str], list[str]]:
    """
    Resolve a list of gene tokens to canonical HPA gene_symbol list (deduped, order preserved).
    Returns (resolved_symbols, all_messages).
    """
    messages: list[str] = []
    seen: set[str] = set()
    resolved: list[str] = []
    for raw in raw_genes:
        sym, _ens, msgs = await _resolve_one_gene_token(async_config, raw)
        messages.extend(msgs)
        if sym and sym not in seen:
            seen.add(sym)
            resolved.append(sym)
        elif not sym:
            messages.append(f"Could not resolve gene token '{raw}'.")
    return resolved, messages


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
    resolved_gene_list: list[str] | None = None

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

        join_sql = _target_lateral_join_sql(search_input.include_target_info)
        extra_cols = _target_select_extra(search_input.include_target_info)

        if mode == "compare_genes":
            s1, _e1, m1 = await _resolve_one_gene_token(async_config, search_input.gene_1 or "")
            messages.extend(m1)
            s2, _e2, m2 = await _resolve_one_gene_token(async_config, search_input.gene_2 or "")
            messages.extend(m2)
            if not s1 or not s2:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=messages + ["Could not resolve one or both genes for compare_genes."],
                )
            cleaned_types = list(search_input.cell_types or [])
            type_filter_sql = ""
            params: list[Any] = [s1, s2, search_input.limit]
            if cleaned_types:
                resolved_cts: list[str] = []
                for ct_raw in cleaned_types:
                    ct, cm = await _resolve_cell_type(async_config, ct_raw)
                    messages.extend(cm)
                    if ct:
                        resolved_cts.append(ct)
                if resolved_cts:
                    type_filter_sql = (
                        "AND COALESCE(g1.cell_type, g2.cell_type) = ANY($3::text[])"
                    )
                    params = [s1, s2, resolved_cts, search_input.limit]
            sql = f"""
                WITH g1 AS (
                    SELECT cell_type, ncpm FROM hpa_rna_cell_type_expression
                    WHERE gene_symbol = $1
                ),
                g2 AS (
                    SELECT cell_type, ncpm FROM hpa_rna_cell_type_expression
                    WHERE gene_symbol = $2
                )
                SELECT
                    COALESCE(g1.cell_type, g2.cell_type) AS cell_type,
                    g1.ncpm AS gene_1_ncpm,
                    g2.ncpm AS gene_2_ncpm
                FROM g1
                FULL OUTER JOIN g2 ON g1.cell_type = g2.cell_type
                WHERE 1=1
                {type_filter_sql}
                ORDER BY ABS(COALESCE(g1.ncpm, 0) - COALESCE(g2.ncpm, 0)) DESC NULLS LAST
                LIMIT ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            for row in rows:
                a = row.get("gene_1_ncpm")
                b = row.get("gene_2_ncpm")
                row["delta_ncpm"] = (float(a) if a is not None else 0.0) - (
                    float(b) if b is not None else 0.0
                )
                row["fold_change"] = _ncpm_fold_change(
                    float(a) if a is not None else None,
                    float(b) if b is not None else None,
                )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    compare_gene_1=s1,
                    compare_gene_2=s2,
                    messages=messages,
                )
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                compare_gene_1=s1,
                compare_gene_2=s2,
                messages=messages,
            )

        if mode == "compare_cell_types":
            ct1, m1 = await _resolve_cell_type(async_config, search_input.cell_type_1 or "")
            messages.extend(m1)
            ct2, m2 = await _resolve_cell_type(async_config, search_input.cell_type_2 or "")
            messages.extend(m2)
            if not ct1 or not ct2:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=messages + ["Could not resolve one or both cell types."],
                )

            glist: list[str] | None = None
            if search_input.gene_symbols:
                rg, gm = await _resolve_gene_list(async_config, search_input.gene_symbols)
                messages.extend(gm)
                if rg:
                    glist = rg

            params: list[Any] = [ct1, ct2]
            gene_filter_sql = ""
            if glist:
                gene_filter_sql = (
                    "AND COALESCE(c1.gene_symbol, c2.gene_symbol) = ANY($3::text[])"
                )
                params.append(glist)

            min_filter_sql = ""
            if search_input.min_ncpm is not None:
                pidx = len(params) + 1
                min_filter_sql = (
                    f"AND (COALESCE(c1.ncpm, 0) >= ${pidx} "
                    f"OR COALESCE(c2.ncpm, 0) >= ${pidx})"
                )
                params.append(search_input.min_ncpm)

            params.append(search_input.limit)
            lim_idx = len(params)

            sql = f"""
                WITH c1 AS (
                    SELECT gene_symbol, ncpm FROM hpa_rna_cell_type_expression
                    WHERE cell_type = $1
                ),
                c2 AS (
                    SELECT gene_symbol, ncpm FROM hpa_rna_cell_type_expression
                    WHERE cell_type = $2
                )
                SELECT
                    COALESCE(c1.gene_symbol, c2.gene_symbol) AS gene_symbol,
                    c1.ncpm AS cell_type_1_ncpm,
                    c2.ncpm AS cell_type_2_ncpm
                FROM c1
                FULL OUTER JOIN c2 ON c1.gene_symbol = c2.gene_symbol
                WHERE 1=1
                {gene_filter_sql}
                {min_filter_sql}
                ORDER BY ABS(COALESCE(c1.ncpm, 0) - COALESCE(c2.ncpm, 0)) DESC NULLS LAST
                LIMIT ${lim_idx}
            """
            rows = await async_config.execute_query(sql, *params)
            for row in rows:
                a = row.get("cell_type_1_ncpm")
                b = row.get("cell_type_2_ncpm")
                row["delta_ncpm"] = (float(a) if a is not None else 0.0) - (
                    float(b) if b is not None else 0.0
                )
                row["fold_change"] = _ncpm_fold_change(
                    float(a) if a is not None else None,
                    float(b) if b is not None else None,
                )
            if not rows:
                return HpaExpressionSearchOutput(
                    status="not_found",
                    mode=mode,
                    compare_cell_type_1=ct1,
                    compare_cell_type_2=ct2,
                    messages=messages,
                )
            return HpaExpressionSearchOutput(
                status="success",
                mode=mode,
                total_results=len(rows),
                results=rows,
                compare_cell_type_1=ct1,
                compare_cell_type_2=ct2,
                resolved_genes=glist,
                query_genes=list(search_input.gene_symbols) if search_input.gene_symbols else None,
                messages=messages,
            )

        if mode in ("gene_expression", "compare_expression"):
            if search_input.gene_symbols:
                resolved_gene_list, gl_msgs = await _resolve_gene_list(
                    async_config, search_input.gene_symbols
                )
                messages.extend(gl_msgs)
                if not resolved_gene_list:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_genes=list(search_input.gene_symbols),
                        messages=messages + ["No genes resolved from gene_symbols list."],
                    )
            elif search_input.ensembl_gene_id:
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
                        messages=messages
                        + [f"No HPA or synonym match for gene '{search_input.gene_symbol}'."],
                    )

        if mode == "gene_expression":
            if resolved_gene_list:
                if search_input.min_ncpm is not None:
                    rows = await async_config.execute_query(
                        f"""
                        SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                        {extra_cols}
                        FROM hpa_rna_cell_type_expression h
                        {join_sql}
                        WHERE h.gene_symbol = ANY($1::text[]) AND h.ncpm >= $2
                        ORDER BY h.gene_symbol, h.ncpm DESC NULLS LAST, h.cell_type
                        LIMIT $3
                        """,
                        resolved_gene_list,
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
                        WHERE h.gene_symbol = ANY($1::text[])
                        ORDER BY h.gene_symbol, h.ncpm DESC NULLS LAST, h.cell_type
                        LIMIT $2
                        """,
                        resolved_gene_list,
                        search_input.limit,
                    )
                if not rows:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_genes=search_input.gene_symbols,
                        resolved_genes=resolved_gene_list,
                        messages=messages,
                    )
                return HpaExpressionSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=len(rows),
                    results=rows,
                    query_genes=list(search_input.gene_symbols or []),
                    resolved_genes=resolved_gene_list,
                    messages=messages,
                )

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
            if search_input.cell_types:
                resolved_cts: list[str] = []
                for ct_raw in search_input.cell_types:
                    ct, ct_msgs = await _resolve_cell_type(async_config, ct_raw)
                    messages.extend(ct_msgs)
                    if ct:
                        resolved_cts.append(ct)
                if not resolved_cts:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_cell_type=str(search_input.cell_types),
                        messages=messages + ["Could not resolve any cell types from list."],
                    )
                if search_input.include_target_info:
                    messages.append(
                        "Note: include_target_info is not applied for multi cell_types batch queries."
                    )
                n_ct = len(resolved_cts)
                per_type = max(1, min(200, search_input.limit // n_ct))
                if search_input.min_ncpm is not None:
                    rows = await async_config.execute_query(
                        f"""
                        WITH ranked AS (
                            SELECT h.gene_symbol, h.ensembl_gene_id, h.ncpm, h.cell_type,
                                ROW_NUMBER() OVER (
                                    PARTITION BY h.cell_type
                                    ORDER BY h.ncpm DESC NULLS LAST
                                ) AS rn
                            FROM hpa_rna_cell_type_expression h
                            WHERE h.cell_type = ANY($1::text[])
                              AND h.ncpm >= $4
                        )
                        SELECT gene_symbol, ensembl_gene_id, ncpm, cell_type
                        FROM ranked
                        WHERE rn <= $2
                        ORDER BY cell_type, ncpm DESC NULLS LAST, gene_symbol
                        LIMIT $3
                        """,
                        resolved_cts,
                        per_type,
                        search_input.limit,
                        search_input.min_ncpm,
                    )
                else:
                    rows = await async_config.execute_query(
                        f"""
                        WITH ranked AS (
                            SELECT h.gene_symbol, h.ensembl_gene_id, h.ncpm, h.cell_type,
                                ROW_NUMBER() OVER (
                                    PARTITION BY h.cell_type
                                    ORDER BY h.ncpm DESC NULLS LAST
                                ) AS rn
                            FROM hpa_rna_cell_type_expression h
                            WHERE h.cell_type = ANY($1::text[])
                        )
                        SELECT gene_symbol, ensembl_gene_id, ncpm, cell_type
                        FROM ranked
                        WHERE rn <= $2
                        ORDER BY cell_type, ncpm DESC NULLS LAST, gene_symbol
                        LIMIT $3
                        """,
                        resolved_cts,
                        per_type,
                        search_input.limit,
                    )
                if not rows:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        resolved_cell_types=resolved_cts,
                        messages=messages,
                    )
                return HpaExpressionSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=len(rows),
                    results=rows,
                    resolved_cell_types=resolved_cts,
                    messages=messages,
                )

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
            if resolved_gene_list:
                cleaned_types = list(search_input.cell_types or [])
                has_types = bool(cleaned_types)
                has_min = search_input.min_ncpm is not None
                resolved_filter_cts: list[str] = []
                if has_types:
                    for ct_raw in cleaned_types:
                        rct, cm = await _resolve_cell_type(async_config, ct_raw)
                        messages.extend(cm)
                        if rct:
                            resolved_filter_cts.append(rct)
                    if not resolved_filter_cts:
                        messages.append(
                            "No cell types resolved from cell_types filter; using all cell types."
                        )

                def _batch_params() -> tuple[str, list[Any]]:
                    if resolved_filter_cts and has_min:
                        return (
                            """
                            WHERE h.gene_symbol = ANY($1::text[])
                              AND h.cell_type = ANY($2::text[])
                              AND h.ncpm >= $3
                            ORDER BY h.gene_symbol, h.cell_type
                            """,
                            [
                                resolved_gene_list,
                                resolved_filter_cts,
                                search_input.min_ncpm,
                                search_input.limit,
                            ],
                        )
                    if resolved_filter_cts:
                        return (
                            """
                            WHERE h.gene_symbol = ANY($1::text[])
                              AND h.cell_type = ANY($2::text[])
                            ORDER BY h.gene_symbol, h.cell_type
                            """,
                            [resolved_gene_list, resolved_filter_cts, search_input.limit],
                        )
                    if has_min:
                        return (
                            """
                            WHERE h.gene_symbol = ANY($1::text[]) AND h.ncpm >= $2
                            ORDER BY h.gene_symbol, h.cell_type
                            """,
                            [resolved_gene_list, search_input.min_ncpm, search_input.limit],
                        )
                    return (
                        """
                        WHERE h.gene_symbol = ANY($1::text[])
                        ORDER BY h.gene_symbol, h.cell_type
                        """,
                        [resolved_gene_list, search_input.limit],
                    )

                where_sql, qargs = _batch_params()
                rows = await async_config.execute_query(
                    f"""
                    SELECT h.cell_type, h.ncpm, h.ensembl_gene_id, h.gene_symbol
                    {extra_cols}
                    FROM hpa_rna_cell_type_expression h
                    {join_sql}
                    {where_sql}
                    LIMIT ${len(qargs)}
                    """,
                    *qargs,
                )
                if not rows:
                    return HpaExpressionSearchOutput(
                        status="not_found",
                        mode=mode,
                        query_genes=search_input.gene_symbols,
                        resolved_genes=resolved_gene_list,
                        messages=messages,
                    )
                return HpaExpressionSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=len(rows),
                    results=rows,
                    query_genes=list(search_input.gene_symbols or []),
                    resolved_genes=resolved_gene_list,
                    messages=messages,
                )

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
