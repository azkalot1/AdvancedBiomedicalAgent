#!/usr/bin/env python3
"""
Open Targets Platform GraphQL API client (v4).

API-only: no local tables. Uses pre-built query templates and optional dm_target
resolution for gene symbol → Ensembl gene id.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

from bioagent.data.ingest.async_config import get_async_connection
from bioagent.data.ingest.config import DatabaseConfig

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# --- GraphQL templates ---

SEARCH_QUERY = """
query OpenTargetsSearch($q: String!, $entities: [String!], $index: Int!, $size: Int!) {
  search(queryString: $q, entityNames: $entities, page: { index: $index, size: $size }) {
    hits {
      id
      entity
      name
      description
      category
      score
    }
  }
}
"""

TARGET_INFO_QUERY = """
query OpenTargetsTargetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    geneticConstraint {
      constraintType
      exp
      obs
      score
      oe
      oeLower
      oeUpper
    }
    tractability {
      label
      modality
      value
    }
    safetyLiabilities {
      event
      datasource
      url
      effects {
        direction
      }
    }
    pathways {
      pathwayId
      topLevelTerm
    }
    functionDescriptions
  }
}
"""

TARGET_ASSOCIATIONS_QUERY = """
query OpenTargetsTargetAssociations($ensemblId: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: { index: $index, size: $size }) {
      count
      rows {
        score
        disease {
          id
          name
        }
      }
    }
  }
}
"""

TARGET_DRUGS_QUERY = """
query OpenTargetsTargetDrugs($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    knownDrugs(size: $size) {
      uniqueDrugs
      rows {
        drug {
          id
          name
        }
        mechanismOfAction
      }
    }
  }
}
"""

DISEASE_INFO_QUERY = """
query OpenTargetsDiseaseInfo($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    description
    therapeuticAreas {
      id
      name
    }
    synonyms {
      relation
      terms
    }
  }
}
"""

DISEASE_TARGETS_QUERY = """
query OpenTargetsDiseaseTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: { index: $index, size: $size }) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
        }
      }
    }
  }
}
"""

DRUG_INFO_QUERY = """
query OpenTargetsDrugInfo($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    drugType
    maximumClinicalTrialPhase
    hasBeenWithdrawn
    tradeNames
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targetName
      }
    }
    indications {
      rows {
        maxPhaseForIndication
        disease {
          id
          name
        }
      }
    }
  }
}
"""

_DISEASE_ID_RE = re.compile(r"^(EFO|MONDO|HP)_", re.I)
_CHEMBL_RE = re.compile(r"^CHEMBL\d+$", re.I)


class OpenTargetsSearchInput(BaseModel):
    mode: Literal[
        "search",
        "target_info",
        "target_associations",
        "target_drugs",
        "disease_info",
        "disease_targets",
        "drug_info",
    ]
    query: str | None = None
    gene: str | None = None
    disease: str | None = None
    drug: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    search_entities: list[str] | None = None

    @field_validator("gene", "disease", "drug", "query", mode="before")
    @classmethod
    def strip_opt(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    def validate_for_mode(self) -> list[str]:
        err: list[str] = []
        m = self.mode
        if m == "search":
            if not self.query:
                err.append("search mode requires query (free text)")
        elif m in {"target_info", "target_associations", "target_drugs"}:
            if not self.gene:
                err.append(f"{m} requires gene (symbol or Ensembl gene id)")
        elif m in {"disease_info", "disease_targets"}:
            if not self.disease:
                err.append(f"{m} requires disease (name or ontology id, e.g. EFO_...)")
        elif m == "drug_info":
            if not self.drug:
                err.append("drug_info requires drug (name or ChEMBL id)")
        return err


class OpenTargetsSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input"]
    mode: str
    total_results: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    resolved_id: str | None = None
    messages: list[str] = Field(default_factory=list)
    error: str | None = None
    raw_payload: dict[str, Any] | None = None


def _open_targets_http_retryable(exc: BaseException) -> bool:
    """Retry on transport failures, timeouts, rate limits, and 5xx (not GraphQL body errors)."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.RequestError)


@retry(
    retry=retry_if_exception(_open_targets_http_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=True,
)
async def _post_graphql_http(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST to Open Targets GraphQL; retries transient HTTP/network errors only."""
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            OT_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Open Targets API returned non-object JSON")
    return body


async def _post_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    body = await _post_graphql_http(query, variables)

    if "syntaxError" in body:
        raise RuntimeError(str(body.get("syntaxError")))
    errs = body.get("errors")
    if errs:
        parts = [str(e.get("message", e)) for e in errs]
        raise RuntimeError("; ".join(parts))
    data = body.get("data")
    if not isinstance(data, dict):
        return {}
    return data


async def _run_search(
    query_string: str,
    entity_names: list[str] | None,
    *,
    index: int = 0,
    size: int = 20,
) -> list[dict[str, Any]]:
    variables: dict[str, Any] = {
        "q": query_string,
        "entities": entity_names,
        "index": index,
        "size": size,
    }
    data = await _post_graphql(SEARCH_QUERY, variables)
    search_block = data.get("search") or {}
    hits = search_block.get("hits") or []
    return [h for h in hits if isinstance(h, dict)]


async def resolve_gene_to_ensembl(
    db_config: DatabaseConfig,
    gene: str,
) -> tuple[str | None, list[str]]:
    """Resolve gene symbol or ENSG to Ensembl gene id (dm_target → OT search fallback)."""
    messages: list[str] = []
    g = gene.strip()
    if not g:
        return None, messages

    if g.upper().startswith("ENSG"):
        return g, [f"Using Ensembl gene id {g}."]

    try:
        async_config = await get_async_connection(db_config)
        row = await async_config.execute_one(
            """
            SELECT ensembl_gene_id, gene_symbol FROM dm_target
            WHERE UPPER(gene_symbol) = UPPER($1)
              AND ensembl_gene_id IS NOT NULL
              AND TRIM(ensembl_gene_id) <> ''
            ORDER BY target_id
            LIMIT 1
            """,
            g,
        )
        if row and row.get("ensembl_gene_id"):
            eid = str(row["ensembl_gene_id"]).strip()
            sym = row.get("gene_symbol")
            messages.append(f"Resolved '{g}' → {eid} via dm_target (symbol {sym}).")
            return eid, messages

        syn = await async_config.execute_query(
            """
            SELECT DISTINCT t.ensembl_gene_id, t.gene_symbol
            FROM dm_target_gene_synonyms s
            JOIN dm_target t ON t.target_id = s.target_id
            WHERE UPPER(s.gene_symbol) = UPPER($1)
              AND t.ensembl_gene_id IS NOT NULL
              AND TRIM(t.ensembl_gene_id) <> ''
            LIMIT 1
            """,
            g,
        )
        if syn:
            eid = str(syn[0]["ensembl_gene_id"]).strip()
            messages.append(f"Resolved synonym '{g}' → {eid} via dm_target_gene_synonyms.")
            return eid, messages
    except Exception as exc:
        messages.append(
            f"dm_target resolution skipped ({type(exc).__name__}: {exc}); trying Open Targets search."
        )

    try:
        hits = await _run_search(g, ["target"], size=8)
    except Exception as exc:
        messages.append(f"Open Targets search fallback failed: {type(exc).__name__}: {exc}")
        return None, messages

    for h in hits:
        if h.get("entity") != "target":
            continue
        hid = str(h.get("id") or "")
        if hid.upper().startswith("ENSG"):
            nm = h.get("name") or ""
            messages.append(f"Resolved '{g}' → {hid} via Open Targets search ({nm}).")
            return hid, messages

    messages.append(f"No Ensembl gene id found for '{g}'.")
    return None, messages


async def resolve_disease_id(disease: str) -> tuple[str | None, list[str]]:
    """Resolve disease name or ontology id for disease(efoId:)."""
    messages: list[str] = []
    d = disease.strip()
    if not d:
        return None, messages

    if _DISEASE_ID_RE.match(d):
        return d, [f"Using disease id {d}."]

    try:
        hits = await _run_search(d, ["disease"], size=10)
    except Exception as exc:
        messages.append(f"Open Targets disease search failed: {type(exc).__name__}: {exc}")
        return None, messages

    for h in hits:
        if h.get("entity") == "disease" and h.get("id"):
            did = str(h["id"])
            nm = h.get("name") or ""
            messages.append(f"Resolved disease '{d}' → {did} ({nm}).")
            return did, messages

    messages.append(f"No disease id found for '{d}'.")
    return None, messages


async def resolve_drug_chembl_id(drug: str) -> tuple[str | None, list[str]]:
    """Resolve drug name or ChEMBL id for drug(chemblId:)."""
    messages: list[str] = []
    raw = drug.strip()
    if not raw:
        return None, messages

    u = raw.upper()
    if u.startswith("CHEMBL"):
        if _CHEMBL_RE.match(u):
            return u, [f"Using ChEMBL id {u}."]
        messages.append(f"Invalid ChEMBL id format: {raw}")
        return None, messages

    try:
        hits = await _run_search(raw, ["drug"], size=10)
    except Exception as exc:
        messages.append(f"Open Targets drug search failed: {type(exc).__name__}: {exc}")
        return None, messages

    for h in hits:
        if h.get("entity") == "drug" and h.get("id"):
            cid = str(h["id"]).upper()
            nm = h.get("name") or ""
            messages.append(f"Resolved drug '{raw}' → {cid} ({nm}).")
            return cid, messages

    messages.append(f"No ChEMBL id found for '{raw}'.")
    return None, messages


async def opentargets_search_async(
    db_config: DatabaseConfig,
    search_input: OpenTargetsSearchInput,
) -> OpenTargetsSearchOutput:
    """Execute one Open Targets GraphQL operation according to mode."""
    mode = search_input.mode
    errors = search_input.validate_for_mode()
    if errors:
        return OpenTargetsSearchOutput(
            status="invalid_input",
            mode=mode,
            error="; ".join(errors),
        )

    limit = min(int(search_input.limit), 100)

    try:
        if mode == "search":
            assert search_input.query
            entities = search_input.search_entities
            hits = await _run_search(search_input.query, entities, size=limit)
            return OpenTargetsSearchOutput(
                status="success",
                mode=mode,
                total_results=len(hits),
                results=hits,
            )

        if mode in {"target_info", "target_associations", "target_drugs"}:
            assert search_input.gene
            ensembl_id, msgs = await resolve_gene_to_ensembl(db_config, search_input.gene)
            if not ensembl_id:
                return OpenTargetsSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=msgs,
                )

            if mode == "target_info":
                data = await _post_graphql(TARGET_INFO_QUERY, {"ensemblId": ensembl_id})
                node = data.get("target")
                if not node:
                    return OpenTargetsSearchOutput(
                        status="not_found",
                        mode=mode,
                        messages=msgs + ["Open Targets returned no target for this id."],
                        resolved_id=ensembl_id,
                    )
                return OpenTargetsSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=1,
                    results=[node],
                    resolved_id=ensembl_id,
                    messages=msgs,
                    raw_payload=data,
                )

            if mode == "target_associations":
                data = await _post_graphql(
                    TARGET_ASSOCIATIONS_QUERY,
                    {"ensemblId": ensembl_id, "index": 0, "size": limit},
                )
                tgt = data.get("target") or {}
                block = tgt.get("associatedDiseases") or {}
                rows = block.get("rows") or []
                out_rows = [r for r in rows if isinstance(r, dict)]
                return OpenTargetsSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=len(out_rows),
                    results=out_rows,
                    resolved_id=ensembl_id,
                    messages=msgs
                    + [f"associatedDiseases.count={block.get('count', '?')}"],
                    raw_payload=data,
                )

            # target_drugs
            data = await _post_graphql(
                TARGET_DRUGS_QUERY,
                {"ensemblId": ensembl_id, "size": limit},
            )
            tgt = data.get("target") or {}
            block = tgt.get("knownDrugs") or {}
            rows = block.get("rows") or []
            out_rows = [r for r in rows if isinstance(r, dict)]
            return OpenTargetsSearchOutput(
                status="success",
                mode=mode,
                total_results=len(out_rows),
                results=out_rows,
                resolved_id=ensembl_id,
                messages=msgs + [f"knownDrugs.uniqueDrugs={block.get('uniqueDrugs', '?')}"],
                raw_payload=data,
            )

        if mode in {"disease_info", "disease_targets"}:
            assert search_input.disease
            disease_id, msgs = await resolve_disease_id(search_input.disease)
            if not disease_id:
                return OpenTargetsSearchOutput(
                    status="not_found",
                    mode=mode,
                    messages=msgs,
                )

            if mode == "disease_info":
                data = await _post_graphql(DISEASE_INFO_QUERY, {"efoId": disease_id})
                node = data.get("disease")
                if not node:
                    return OpenTargetsSearchOutput(
                        status="not_found",
                        mode=mode,
                        messages=msgs + ["Open Targets returned no disease for this id."],
                        resolved_id=disease_id,
                    )
                return OpenTargetsSearchOutput(
                    status="success",
                    mode=mode,
                    total_results=1,
                    results=[node],
                    resolved_id=disease_id,
                    messages=msgs,
                    raw_payload=data,
                )

            data = await _post_graphql(
                DISEASE_TARGETS_QUERY,
                {"efoId": disease_id, "index": 0, "size": limit},
            )
            dis = data.get("disease") or {}
            block = dis.get("associatedTargets") or {}
            rows = block.get("rows") or []
            out_rows = [r for r in rows if isinstance(r, dict)]
            return OpenTargetsSearchOutput(
                status="success",
                mode=mode,
                total_results=len(out_rows),
                results=out_rows,
                resolved_id=disease_id,
                messages=msgs + [f"associatedTargets.count={block.get('count', '?')}"],
                raw_payload=data,
            )

        # drug_info
        assert search_input.drug
        chembl_id, msgs = await resolve_drug_chembl_id(search_input.drug)
        if not chembl_id:
            return OpenTargetsSearchOutput(
                status="not_found",
                mode=mode,
                messages=msgs,
            )
        data = await _post_graphql(DRUG_INFO_QUERY, {"chemblId": chembl_id})
        node = data.get("drug")
        if not node:
            return OpenTargetsSearchOutput(
                status="not_found",
                mode=mode,
                messages=msgs + ["Open Targets returned no drug for this id."],
                resolved_id=chembl_id,
            )
        return OpenTargetsSearchOutput(
            status="success",
            mode=mode,
            total_results=1,
            results=[node],
            resolved_id=chembl_id,
            messages=msgs,
            raw_payload=data,
        )

    except Exception as exc:
        return OpenTargetsSearchOutput(
            status="error",
            mode=mode,
            error=f"{type(exc).__name__}: {exc}",
        )
