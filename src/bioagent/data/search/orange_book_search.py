#!/usr/bin/env python3
"""
Orange Book Search.

Supports therapeutic equivalence (TE) codes, patent records, and exclusivity
lookups for FDA-approved products. Queries can be filtered by trade name,
ingredient, or NDA number.

Modes:
    - te_codes: TE code lookups with product summaries
    - patents: patent listings for matching products
    - exclusivity: exclusivity listings for matching products
    - generics: combined product, patent, and exclusivity views

Data Sources:
    - orange_book_products
    - orange_book_patents
    - orange_book_exclusivity
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


class OrangeBookSearchInput(BaseModel):
    mode: Literal["te_codes", "patents", "generics", "exclusivity"]
    drug_name: str | None = None
    nda_number: str | None = None
    ingredient: str | None = None
    include_patents: bool = True
    include_exclusivity: bool = True
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("drug_name", "nda_number", "ingredient", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def validate_for_mode(self) -> list[str]:
        errors: list[str] = []
        if not (self.drug_name or self.nda_number or self.ingredient):
            errors.append("Provide at least one of: drug_name, ingredient, nda_number")
        return errors


class OrangeBookPatentHit(BaseModel):
    patent_no: str | None = None
    patent_expiration_date: str | None = None
    drug_substance_flag: str | None = None
    drug_product_flag: str | None = None
    patent_use_code: str | None = None
    patent_use_code_description: str | None = None


class OrangeBookExclusivityHit(BaseModel):
    exclusivity_code: str | None = None
    exclusivity_date: str | None = None


class OrangeBookProductHit(BaseModel):
    appl_no: str | None = None
    product_no: str | None = None
    trade_name: str | None = None
    ingredient: str | None = None
    dosage_form: str | None = None
    route: str | None = None
    strength: str | None = None
    te_code: str | None = None
    applicant: str | None = None
    approval_date: str | None = None
    drug_type: str | None = None
    appl_type: str | None = None
    rld: str | None = None
    rs: str | None = None
    applicant_full_name: str | None = None
    patents: list[OrangeBookPatentHit] = Field(default_factory=list)
    exclusivity: list[OrangeBookExclusivityHit] = Field(default_factory=list)


class OrangeBookSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    error: str | None = None
    query_summary: str = ""


async def _fetch_products(
    async_config: AsyncDatabaseConfig,
    drug_name: str | None,
    ingredient: str | None,
    nda_number: str | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    filters: list[str] = []

    if drug_name:
        params.append(f"%{drug_name}%")
        filters.append(f"trade_name ILIKE ${len(params)}")

    if ingredient:
        params.append(f"%{ingredient}%")
        filters.append(f"ingredient ILIKE ${len(params)}")

    if nda_number:
        params.append(nda_number)
        filters.append(f"appl_no = ${len(params)}")

    if not filters:
        return []

    params.extend([limit, offset])
    sql = f"""
        SELECT appl_no, product_no, trade_name, ingredient, dosage_form, route, strength,
               te_code, applicant, approval_date, drug_type, appl_type, rld, rs, applicant_full_name
        FROM orange_book_products
        WHERE {" OR ".join(filters)}
        ORDER BY trade_name, appl_no, product_no
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return await async_config.execute_query(sql, *params)


async def _fetch_patents(
    async_config: AsyncDatabaseConfig,
    appl_no: str,
    product_no: str,
) -> list[OrangeBookPatentHit]:
    rows = await async_config.execute_query(
        """
        SELECT patent_no, patent_expiration_date, drug_substance_flag, drug_product_flag,
               patent_use_code, patent_use_code_description
        FROM orange_book_patents
        WHERE appl_no = $1 AND product_no = $2
        ORDER BY patent_expiration_date
        """,
        appl_no,
        product_no,
    )
    return [OrangeBookPatentHit(**row) for row in rows]


async def _fetch_exclusivity(
    async_config: AsyncDatabaseConfig,
    appl_no: str,
    product_no: str,
) -> list[OrangeBookExclusivityHit]:
    rows = await async_config.execute_query(
        """
        SELECT exclusivity_code, exclusivity_date
        FROM orange_book_exclusivity
        WHERE appl_no = $1 AND product_no = $2
        ORDER BY exclusivity_date
        """,
        appl_no,
        product_no,
    )
    return [OrangeBookExclusivityHit(**row) for row in rows]


async def orange_book_search_async(
    db_config: DatabaseConfig,
    search_input: OrangeBookSearchInput,
) -> OrangeBookSearchOutput:
    """
    Search Orange Book products, patents, and exclusivity data.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: OrangeBookSearchInput with mode and filters.

    Returns:
        OrangeBookSearchOutput with status, hits, and query summary.
    """
    errors = search_input.validate_for_mode()
    if errors:
        return OrangeBookSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="; ".join(errors),
            query_summary="invalid input",
        )

    try:
        async_config = await get_async_connection(db_config)
        rows = await _fetch_products(
            async_config,
            search_input.drug_name,
            search_input.ingredient,
            search_input.nda_number,
            search_input.limit,
            search_input.offset,
        )
        if not rows:
            return OrangeBookSearchOutput(
                status="not_found",
                mode=search_input.mode,
                query_summary=search_input.drug_name or search_input.ingredient or search_input.nda_number or "",
            )

        hits: list[OrangeBookProductHit] = []
        for row in rows:
            hit = OrangeBookProductHit(**row)
            if search_input.mode in ("patents", "te_codes", "generics") and search_input.include_patents:
                if hit.appl_no and hit.product_no:
                    hit.patents = await _fetch_patents(async_config, hit.appl_no, hit.product_no)
            if search_input.mode in ("exclusivity", "te_codes", "generics") and search_input.include_exclusivity:
                if hit.appl_no and hit.product_no:
                    hit.exclusivity = await _fetch_exclusivity(async_config, hit.appl_no, hit.product_no)
            hits.append(hit)

        status = "success" if hits else "not_found"
        return OrangeBookSearchOutput(
            status=status,
            mode=search_input.mode,
            total_hits=len(hits),
            hits=hits,
            query_summary=search_input.drug_name or search_input.ingredient or search_input.nda_number or "",
        )
    except Exception as exc:
        return OrangeBookSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )
