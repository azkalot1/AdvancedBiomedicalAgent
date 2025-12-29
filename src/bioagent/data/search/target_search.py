#!/usr/bin/env python3
# target_search.py
"""
Unified Pharmacology Search Module

Single entry point for all drug-target, structure, and clinical trial searches.

Usage:
    from bioagent.search import PharmacologySearch, SearchInput, SearchMode
    
    search = PharmacologySearch(db_config)
    
    # All searches go through one function
    results = await search.search(SearchInput(
        mode=SearchMode.TARGETS_FOR_DRUG,
        query="imatinib",
        min_pchembl=6.0
    ))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, List, Dict, Union
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Adjust imports based on your project structure
try:
    from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
    from bioagent.data.ingest.config import DatabaseConfig, DEFAULT_CONFIG
except ImportError:
    from async_config import AsyncDatabaseConfig, get_async_connection
    from config import DatabaseConfig, DEFAULT_CONFIG


# =============================================================================
# ENUMS
# =============================================================================

class SearchMode(str, Enum):
    """All available search modes."""
    # Drug-centric searches
    TARGETS_FOR_DRUG = "targets_for_drug"          # Drug name → all targets
    TRIALS_FOR_DRUG = "trials_for_drug"            # Drug name → clinical trials
    DRUG_PROFILE = "drug_profile"                  # Drug name → full profile
    DRUG_FORMS = "drug_forms"                      # Drug name → all salt/stereo forms
    
    # Target-centric searches
    DRUGS_FOR_TARGET = "drugs_for_target"          # Target → all drugs
    SELECTIVE_DRUGS = "selective_drugs"            # Target + off-targets → selective drugs
    
    # Structure-based searches
    SIMILAR_MOLECULES = "similar_molecules"        # SMILES → similar structures
    EXACT_STRUCTURE = "exact_structure"            # SMILES → exact match
    SUBSTRUCTURE = "substructure"                  # SMARTS → containing structures
    
    # Comparison searches
    COMPARE_DRUGS = "compare_drugs"                # Multiple drugs + target → comparison
    
    # Activity searches (raw, not deduplicated)
    ACTIVITIES_FOR_DRUG = "activities_for_drug"    # Drug → all activity measurements
    ACTIVITIES_FOR_TARGET = "activities_for_target" # Target → all activity measurements


class ActivityType(str, Enum):
    IC50 = "IC50"
    KI = "Ki"
    KD = "Kd"
    EC50 = "EC50"
    ALL = "ALL"


class DataConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    ANY = "ANY"


class TrialPhase(str, Enum):
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    ANY = "ANY"


class SortOrder(str, Enum):
    POTENCY_DESC = "potency_desc"       # Most potent first
    POTENCY_ASC = "potency_asc"         # Least potent first
    SIMILARITY_DESC = "similarity_desc"  # Most similar first
    NAME_ASC = "name_asc"               # Alphabetical
    MEASUREMENTS_DESC = "measurements_desc"  # Most data first
    SELECTIVITY_DESC = "selectivity_desc"    # Most selective first


# =============================================================================
# INPUT MODEL
# =============================================================================

class TargetSearchInput(BaseModel):
    """
    Unified input for all search types.
    
    Required fields depend on the search mode:
    
    Mode                    | Required              | Optional
    ------------------------|-----------------------|---------------------------
    TARGETS_FOR_DRUG        | query (drug name)     | min_pchembl, activity_type, confidence
    DRUGS_FOR_TARGET        | query (gene symbol)   | min_pchembl, activity_type, confidence
    SIMILAR_MOLECULES       | smiles                | similarity_threshold, min_pchembl
    EXACT_STRUCTURE         | smiles                | -
    SUBSTRUCTURE            | smarts                | -
    TRIALS_FOR_DRUG         | query (drug name)     | trial_phase, trial_status
    DRUG_PROFILE            | query (drug name)     | include_trials, include_forms
    DRUG_FORMS              | query (drug name)     | -
    COMPARE_DRUGS           | drug_names, target    | activity_type
    SELECTIVE_DRUGS         | target, off_targets   | min_selectivity_fold, min_pchembl
    ACTIVITIES_FOR_DRUG     | query (drug name)     | activity_type, min_pchembl
    ACTIVITIES_FOR_TARGET   | query (gene symbol)   | activity_type, min_pchembl
    
    Examples:
        # Find all targets for imatinib
        SearchInput(mode=SearchMode.TARGETS_FOR_DRUG, query="imatinib")
        
        # Find JAK2 inhibitors with IC50 < 100nM
        SearchInput(mode=SearchMode.DRUGS_FOR_TARGET, query="JAK2", min_pchembl=7.0)
        
        # Similarity search
        SearchInput(mode=SearchMode.SIMILAR_MOLECULES, smiles="CCO", similarity_threshold=0.7)
        
        # Compare drugs on target
        SearchInput(
            mode=SearchMode.COMPARE_DRUGS, 
            target="ABL1",
            drug_names=["imatinib", "dasatinib", "nilotinib"]
        )
    """
    
    # Core parameters
    mode: SearchMode
    query: str | None = None  # Drug name, gene symbol, or general query
    
    # Structure parameters
    smiles: str | None = None
    smarts: str | None = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Activity filters
    min_pchembl: float = Field(default=5.0, ge=0.0, le=15.0)
    max_pchembl: float | None = None
    activity_type: ActivityType = ActivityType.ALL
    min_confidence: DataConfidence = DataConfidence.ANY
    
    # Target parameters (for selectivity searches)
    target: str | None = None  # Primary target gene symbol
    off_targets: list[str] = Field(default_factory=list)
    min_selectivity_fold: float = Field(default=10.0, ge=1.0)
    
    # Drug list (for comparison)
    drug_names: list[str] = Field(default_factory=list)
    
    # Clinical trial filters
    trial_phase: TrialPhase = TrialPhase.ANY
    trial_status: str | None = None  # e.g., "RECRUITING", "COMPLETED"
    
    # Profile options
    include_trials: bool = True
    include_forms: bool = True
    include_activities: bool = True
    
    # Output control
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: SortOrder = SortOrder.POTENCY_DESC
    
    # Organism filter
    organism: str = "Homo sapiens"
    include_all_organisms: bool = False

    @field_validator('query', 'smiles', 'smarts', 'target', mode='before')
    @classmethod
    def strip_strings(cls, v):
        """Strip whitespace from string inputs."""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator('off_targets', 'drug_names', mode='before')
    @classmethod
    def uppercase_gene_lists(cls, v):
        """Uppercase gene symbols in lists."""
        if isinstance(v, list):
            return [x.upper() if isinstance(x, str) else x for x in v]
        return v

    def validate_for_mode(self) -> list[str]:
        """Validate that required fields are present for the search mode."""
        errors = []
        
        if self.mode in [
            SearchMode.TARGETS_FOR_DRUG,
            SearchMode.TRIALS_FOR_DRUG,
            SearchMode.DRUG_PROFILE,
            SearchMode.DRUG_FORMS,
            SearchMode.ACTIVITIES_FOR_DRUG
        ]:
            if not self.query:
                errors.append(f"'query' (drug name) is required for {self.mode.value}")
        
        elif self.mode in [
            SearchMode.DRUGS_FOR_TARGET,
            SearchMode.ACTIVITIES_FOR_TARGET
        ]:
            if not self.query:
                errors.append(f"'query' (gene symbol) is required for {self.mode.value}")
        
        elif self.mode in [SearchMode.SIMILAR_MOLECULES, SearchMode.EXACT_STRUCTURE]:
            if not self.smiles:
                errors.append(f"'smiles' is required for {self.mode.value}")
        
        elif self.mode == SearchMode.SUBSTRUCTURE:
            if not self.smarts and not self.smiles:
                errors.append(f"'smarts' or 'smiles' is required for {self.mode.value}")
        
        elif self.mode == SearchMode.COMPARE_DRUGS:
            if not self.drug_names:
                errors.append("'drug_names' list is required for compare_drugs")
            if not self.target and not self.query:
                errors.append("'target' or 'query' (gene symbol) is required for compare_drugs")
        
        elif self.mode == SearchMode.SELECTIVE_DRUGS:
            if not self.target and not self.query:
                errors.append("'target' or 'query' (gene symbol) is required for selective_drugs")
            if not self.off_targets:
                errors.append("'off_targets' list is required for selective_drugs")
        
        return errors

    @property
    def effective_target(self) -> str | None:
        """Get the target gene symbol from either 'target' or 'query' field."""
        return self.target or self.query

    @property
    def effective_smarts(self) -> str | None:
        """Get SMARTS pattern, falling back to SMILES if not provided."""
        return self.smarts or self.smiles


# =============================================================================
# RESULT MODELS
# =============================================================================

class DrugTargetHit(BaseModel):
    """A single drug-target interaction result."""
    # Drug info
    concept_id: int | None = None
    concept_name: str | None = None
    mol_id: int | None = None
    molecule_name: str | None = None
    canonical_smiles: str | None = None
    chembl_id: str | None = None
    
    # Target info
    target_id: int | None = None
    gene_symbol: str = "N/A"
    target_name: str | None = None
    target_organism: str | None = "Homo sapiens"
    
    # Activity info
    activity_type: str = "N/A"
    activity_value_nm: float = 0.0
    pchembl: float | None = None
    
    # Quality metrics
    n_measurements: int | None = None
    data_confidence: str | None = None
    sources: list[str] | None = None
    
    # Similarity (for structure searches)
    tanimoto_similarity: float | None = None
    
    # Selectivity (for selectivity searches)
    selectivity_fold: float | None = None

    @property
    def display_name(self) -> str:
        """Best available name for display."""
        return self.concept_name or self.molecule_name or self.chembl_id or f"MOL_{self.mol_id}"

    @property
    def activity_value_um(self) -> float:
        return self.activity_value_nm / 1000

    @property
    def potency_category(self) -> str:
        if not self.pchembl:
            return "Unknown"
        if self.pchembl >= 9:
            return "Very Potent (<1 nM)"
        elif self.pchembl >= 8:
            return "Potent (1-10 nM)"
        elif self.pchembl >= 7:
            return "Moderate (10-100 nM)"
        elif self.pchembl >= 6:
            return "Weak (100-1000 nM)"
        else:
            return "Very Weak (>1 μM)"


class ClinicalTrialHit(BaseModel):
    """A clinical trial result."""
    nct_id: str
    trial_title: str
    trial_status: str | None = None
    phase: str | None = None
    concept_id: int | None = None
    concept_name: str | None = None
    molecule_form: str | None = None
    salt_form: str | None = None
    match_type: str | None = None
    confidence: float | None = None


class MoleculeForm(BaseModel):
    """A specific form of a drug."""
    mol_id: int
    form_name: str | None = None
    inchi_key: str | None = None
    canonical_smiles: str | None = None
    is_salt: bool = False
    salt_form: str | None = None
    stereo_type: str | None = None
    chembl_id: str | None = None
    drugcentral_id: int | None = None


class DrugComparisonHit(BaseModel):
    """Result of comparing drugs on a target."""
    drug_name: str
    activity_value_nm: float | None = None
    median_value_nm: float | None = None
    n_measurements: int = 0
    fold_vs_best: float | None = None
    data_confidence: str | None = None
    sources: list[str] | None = None


class DrugProfileResult(BaseModel):
    """Complete drug profile."""
    concept_id: int
    concept_name: str
    parent_inchi_key_14: str | None = None
    n_forms: int = 0
    n_stereo_variants: int = 0
    n_salt_forms: int = 0
    n_clinical_trials: int = 0
    n_targets: int = 0
    forms: list[MoleculeForm] = []
    top_targets: list[DrugTargetHit] = []
    recent_trials: list[ClinicalTrialHit] = []


# Union type for all possible hit types
SearchHit = Union[DrugTargetHit, ClinicalTrialHit, MoleculeForm, DrugComparisonHit, DrugProfileResult]


class TargetSearchOutput(BaseModel):
    """Universal search results container."""
    status: Literal["success", "not_found", "error", "invalid_input"] = "success"
    mode: SearchMode
    query_summary: str = ""
    total_hits: int = 0
    hits: list[Any] = []
    error: str | None = None
    warnings: list[str] = []
    
    # Input echo
    input_params: dict = {}
    
    # Metadata
    execution_time_ms: float | None = None

    def pretty_print(self, console: Console | None = None, max_hits: int = 20):
        """Pretty print results."""
        console = console or Console()
        
        status_color = {
            "success": "green", 
            "not_found": "yellow", 
            "error": "red",
            "invalid_input": "red"
        }[self.status]
        
        console.print(Panel(
            f"[bold {status_color}]{self.status.upper()}[/bold {status_color}] | "
            f"Mode: [cyan]{self.mode.value}[/cyan] | "
            f"Query: [yellow]{self.query_summary}[/yellow] | "
            f"Hits: [bold]{self.total_hits}[/bold]",
            title="🔬 Search Results",
            border_style=status_color
        ))
        
        if self.error:
            console.print(f"[red]Error: {self.error}[/red]")
            return
        
        for w in self.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")
        
        if not self.hits:
            console.print("[dim]No results found.[/dim]")
            return
        
        for i, hit in enumerate(self.hits[:max_hits], 1):
            self._print_hit(console, i, hit)
        
        if self.total_hits > max_hits:
            console.print(f"\n[dim]... and {self.total_hits - max_hits} more results[/dim]")

    def _print_hit(self, console: Console, rank: int, hit: Any):
        """Print a single hit based on type."""
        if isinstance(hit, DrugTargetHit):
            # Build display components
            name = hit.display_name
            
            # Similarity badge
            sim_str = f"[blue](Sim: {hit.tanimoto_similarity:.2f})[/blue] " if hit.tanimoto_similarity else ""
            
            # Selectivity badge
            sel_str = f"[magenta](Sel: {hit.selectivity_fold:.1f}x)[/magenta] " if hit.selectivity_fold else ""
            
            # Sources badge
            src_str = f" [dim][{', '.join(hit.sources)}][/dim]" if hit.sources else ""
            
            # Target info
            if hit.gene_symbol and hit.gene_symbol != "N/A":
                target_str = f" → [bold magenta]{hit.gene_symbol}[/bold magenta]"
            else:
                target_str = ""
            
            # Activity info
            if hit.pchembl and hit.activity_value_nm > 0:
                p_color = "green" if hit.pchembl >= 8 else "yellow" if hit.pchembl >= 6 else "white"
                activity_str = f" | {hit.activity_type}: [{p_color}]{hit.activity_value_nm:.2f} nM[/{p_color}] (pChEMBL: [{p_color}]{hit.pchembl:.2f}[/{p_color}])"
            elif hit.pchembl:
                p_color = "green" if hit.pchembl >= 8 else "yellow" if hit.pchembl >= 6 else "white"
                activity_str = f" | pChEMBL: [{p_color}]{hit.pchembl:.2f}[/{p_color}]"
            else:
                activity_str = ""
            
            # SMILES preview (for structure searches)
            smiles_str = ""
            if not target_str and hit.canonical_smiles:
                smiles_preview = hit.canonical_smiles[:50] + "..." if len(hit.canonical_smiles) > 50 else hit.canonical_smiles
                smiles_str = f"\n      [dim]{smiles_preview}[/dim]"
            
            console.print(
                f"  {rank}. {sim_str}{sel_str}[bold cyan]{name}[/bold cyan]"
                f"{target_str}{activity_str}{src_str}{smiles_str}"
            )
        
        elif isinstance(hit, ClinicalTrialHit):
            phase_color = {"PHASE1": "yellow", "PHASE2": "cyan", "PHASE3": "green", "PHASE4": "blue"}.get(hit.phase or "", "white")
            status_str = f"[{hit.trial_status}]" if hit.trial_status else ""
            console.print(
                f"  {rank}. [bold]{hit.nct_id}[/bold] [{phase_color}]{hit.phase or 'N/A'}[/{phase_color}] "
                f"{status_str}\n      {hit.trial_title[:80]}..."
            )
        
        elif isinstance(hit, DrugComparisonHit):
            fold_str = f"{hit.fold_vs_best:.1f}x" if hit.fold_vs_best else "best"
            val_str = f"{hit.activity_value_nm:.2f} nM" if hit.activity_value_nm else "N/A"
            console.print(
                f"  {rank}. [bold cyan]{hit.drug_name}[/bold cyan]: {val_str} "
                f"(fold vs best: {fold_str}, n={hit.n_measurements})"
            )
        
        elif isinstance(hit, MoleculeForm):
            name = hit.form_name or hit.chembl_id or f"MOL_{hit.mol_id}"
            salt_str = f" [yellow]({hit.salt_form})[/yellow]" if hit.salt_form else ""
            stereo_str = f" [dim][{hit.stereo_type}][/dim]" if hit.stereo_type else ""
            smiles_preview = ""
            if hit.canonical_smiles:
                s = hit.canonical_smiles[:50] + "..." if len(hit.canonical_smiles) > 50 else hit.canonical_smiles
                smiles_preview = f"\n      [dim]{s}[/dim]"
            console.print(
                f"  {rank}. [cyan]{name}[/cyan]{salt_str}{stereo_str}{smiles_preview}"
            )
        
        elif isinstance(hit, DrugProfileResult):
            console.print(f"\n[bold cyan]{hit.concept_name}[/bold cyan]")
            console.print(f"  Forms: {hit.n_forms} ({hit.n_salt_forms} salts, {hit.n_stereo_variants} stereoisomers)")
            console.print(f"  Targets: {hit.n_targets}")
            console.print(f"  Clinical Trials: {hit.n_clinical_trials}")
            if hit.top_targets:
                console.print("  Top Targets:")
                for t in hit.top_targets[:5]:
                    val_str = f"{t.activity_value_nm:.1f} nM" if t.activity_value_nm else "N/A"
                    console.print(f"    • {t.gene_symbol}: {val_str}")
        
        else:
            console.print(f"  {rank}. {hit}")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_dataframe(self):
        """Convert to pandas DataFrame if pandas is available."""
        try:
            import pandas as pd
            if self.hits and hasattr(self.hits[0], 'model_dump'):
                return pd.DataFrame([h.model_dump() for h in self.hits])
            return pd.DataFrame(self.hits)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

    def filter_by_potency(self, min_pchembl: float) -> "TargetSearchOutput":
        """Return filtered results."""
        if not self.hits:
            return self
        filtered = [h for h in self.hits if hasattr(h, 'pchembl') and h.pchembl and h.pchembl >= min_pchembl]
        return TargetSearchOutput(
            status=self.status,
            mode=self.mode,
            query_summary=self.query_summary,
            total_hits=len(filtered),
            hits=filtered,
            input_params={**self.input_params, "filter_min_pchembl": min_pchembl}
        )


# =============================================================================
# MAIN SEARCH CLASS
# =============================================================================

class PharmacologySearch:
    """
    Unified pharmacology search interface.
    
    All searches go through the single `search()` method with a `SearchInput` object.
    
    Usage:
        search = PharmacologySearch(db_config)
        
        # Find targets for a drug
        results = await search.search(SearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query="imatinib",
            min_pchembl=6.0
        ))
        
        # Find similar molecules
        results = await search.search(SearchInput(
            mode=SearchMode.SIMILAR_MOLECULES,
            smiles="Cc1ccc(cc1)NC(=O)c2ccc(cc2)CN3CCN(CC3)C",
            similarity_threshold=0.7
        ))
    """
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self._async_config: AsyncDatabaseConfig | None = None
    
    async def _get_conn(self) -> AsyncDatabaseConfig:
        if self._async_config is None:
            self._async_config = await get_async_connection(self.db_config)
        return self._async_config

    async def search(self, input: TargetSearchInput) -> TargetSearchOutput:
        """
        Execute a search based on the input parameters.
        
        This is the main entry point for all searches.
        
        Args:
            input: TargetSearchInput object specifying the search mode and parameters
            
        Returns:
            TargetSearchOutput containing the hits and metadata
        """
        import time
        start_time = time.time()
        
        # Validate input
        validation_errors = input.validate_for_mode()
        if validation_errors:
            return TargetSearchOutput(
                status="invalid_input",
                mode=input.mode,
                query_summary=input.query or input.smiles or str(input.drug_names) or "N/A",
                error="; ".join(validation_errors),
                input_params=input.model_dump(exclude_none=True)
            )
        
        # Route to appropriate handler
        try:
            handler = self._get_handler(input.mode)
            results = await handler(input)
            results.execution_time_ms = (time.time() - start_time) * 1000
            results.input_params = input.model_dump(exclude_none=True)
            return results
        except Exception as e:
            return TargetSearchOutput(
                status="error",
                mode=input.mode,
                query_summary=input.query or input.smiles or "N/A",
                error=str(e),
                input_params=input.model_dump(exclude_none=True),
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _get_handler(self, mode: SearchMode):
        """Get the handler function for a search mode."""
        handlers = {
            SearchMode.TARGETS_FOR_DRUG: self._search_targets_for_drug,
            SearchMode.DRUGS_FOR_TARGET: self._search_drugs_for_target,
            SearchMode.SIMILAR_MOLECULES: self._search_similar_molecules,
            SearchMode.EXACT_STRUCTURE: self._search_exact_structure,
            SearchMode.SUBSTRUCTURE: self._search_substructure,
            SearchMode.TRIALS_FOR_DRUG: self._search_trials_for_drug,
            SearchMode.DRUG_PROFILE: self._search_drug_profile,
            SearchMode.DRUG_FORMS: self._search_drug_forms,
            SearchMode.COMPARE_DRUGS: self._search_compare_drugs,
            SearchMode.SELECTIVE_DRUGS: self._search_selective_drugs,
            SearchMode.ACTIVITIES_FOR_DRUG: self._search_activities_for_drug,
            SearchMode.ACTIVITIES_FOR_TARGET: self._search_activities_for_target,
        }
        return handlers[mode]

    # =========================================================================
    # HANDLER: TARGETS_FOR_DRUG
    # =========================================================================
    
    async def _search_targets_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all targets for a drug."""
        conn = await self._get_conn()
        
        sql = """
            SELECT 
                mts.concept_id,
                mts.concept_name,
                mts.representative_mol_id as mol_id,
                mts.representative_smiles as canonical_smiles,
                mts.target_id,
                mts.gene_symbol,
                mts.target_name,
                mts.target_organism,
                COALESCE(mts.best_ic50_nm, mts.best_ki_nm, mts.best_kd_nm, mts.best_ec50_nm) as activity_value_nm,
                CASE 
                    WHEN mts.best_ic50_nm IS NOT NULL THEN 'IC50'
                    WHEN mts.best_ki_nm IS NOT NULL THEN 'Ki'
                    WHEN mts.best_kd_nm IS NOT NULL THEN 'Kd'
                    WHEN mts.best_ec50_nm IS NOT NULL THEN 'EC50'
                END as activity_type,
                mts.best_pchembl as pchembl,
                mts.n_total_measurements,
                mts.data_confidence,
                mts.sources
            FROM dm_molecule_target_summary mts
            WHERE (
                mts.concept_name ILIKE $1
                OR mts.concept_id IN (
                    SELECT DISTINCT dm.concept_id
                    FROM dm_molecule_synonyms dms
                    JOIN dm_molecule dm ON dms.mol_id = dm.mol_id
                    WHERE dms.synonym_lower ILIKE LOWER($1)
                )
            )
            AND mts.best_pchembl >= $2
        """
        
        params = [f"%{input.query}%", input.min_pchembl]
        param_idx = 3
        
        # Activity type filter
        if input.activity_type != ActivityType.ALL:
            col_map = {
                ActivityType.IC50: "best_ic50_nm",
                ActivityType.KI: "best_ki_nm",
                ActivityType.KD: "best_kd_nm",
                ActivityType.EC50: "best_ec50_nm"
            }
            sql += f" AND mts.{col_map[input.activity_type]} IS NOT NULL"
        
        # Confidence filter
        if input.min_confidence != DataConfidence.ANY:
            if input.min_confidence == DataConfidence.HIGH:
                sql += " AND mts.data_confidence = 'HIGH'"
            elif input.min_confidence == DataConfidence.MEDIUM:
                sql += " AND mts.data_confidence IN ('HIGH', 'MEDIUM')"
        
        # Organism filter
        if not input.include_all_organisms:
            sql += f" AND mts.target_organism = ${param_idx}"
            params.append(input.organism)
            param_idx += 1
        
        sql += f" ORDER BY mts.best_pchembl DESC NULLS LAST LIMIT ${param_idx}"
        params.append(input.limit)
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"targets for '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: DRUGS_FOR_TARGET
    # =========================================================================
    
    async def _search_drugs_for_target(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all drugs for a target."""
        conn = await self._get_conn()
        
        # Build value column based on activity type
        if input.activity_type == ActivityType.ALL:
            value_col = "COALESCE(mts.best_ic50_nm, mts.best_ki_nm, mts.best_kd_nm, mts.best_ec50_nm)"
            type_expr = """CASE 
                WHEN mts.best_ic50_nm IS NOT NULL THEN 'IC50'
                WHEN mts.best_ki_nm IS NOT NULL THEN 'Ki'
                WHEN mts.best_kd_nm IS NOT NULL THEN 'Kd'
                WHEN mts.best_ec50_nm IS NOT NULL THEN 'EC50'
            END"""
        else:
            col_map = {
                ActivityType.IC50: "best_ic50_nm",
                ActivityType.KI: "best_ki_nm",
                ActivityType.KD: "best_kd_nm",
                ActivityType.EC50: "best_ec50_nm"
            }
            value_col = f"mts.{col_map[input.activity_type]}"
            type_expr = f"'{input.activity_type.value}'"
        
        sql = f"""
            SELECT 
                mts.concept_id,
                mts.concept_name,
                mts.representative_mol_id as mol_id,
                mts.representative_smiles as canonical_smiles,
                mts.target_id,
                mts.gene_symbol,
                mts.target_name,
                mts.target_organism,
                {value_col} as activity_value_nm,
                {type_expr} as activity_type,
                mts.best_pchembl as pchembl,
                mts.n_total_measurements,
                mts.data_confidence,
                mts.sources
            FROM dm_molecule_target_summary mts
            WHERE mts.gene_symbol = UPPER($1)
              AND mts.best_pchembl >= $2
        """
        
        params = [input.query, input.min_pchembl]
        param_idx = 3
        
        # Activity type filter
        if input.activity_type != ActivityType.ALL:
            sql += f" AND mts.{col_map[input.activity_type]} IS NOT NULL"
        
        # Confidence filter
        if input.min_confidence != DataConfidence.ANY:
            if input.min_confidence == DataConfidence.HIGH:
                sql += " AND mts.data_confidence = 'HIGH'"
            elif input.min_confidence == DataConfidence.MEDIUM:
                sql += " AND mts.data_confidence IN ('HIGH', 'MEDIUM')"
        
        sql += f" ORDER BY mts.best_pchembl DESC NULLS LAST LIMIT ${param_idx}"
        params.append(input.limit)
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"drugs for '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: SIMILAR_MOLECULES
    # =========================================================================
    
    async def _search_similar_molecules(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find molecules similar to a query structure."""
        conn = await self._get_conn()
        
        if input.include_activities:
            sql = f"""
                WITH query_fp AS (
                    SELECT morganbv_fp(mol_from_smiles($1::cstring)) as fp
                ),
                similar_mols AS (
                    SELECT 
                        dm.mol_id,
                        dm.concept_id,
                        dm.pref_name as molecule_name,
                        dm.canonical_smiles,
                        dm.chembl_id,
                        tanimoto_sml(q.fp, dm.mfp2)::float as similarity
                    FROM dm_molecule dm
                    CROSS JOIN query_fp q
                    WHERE dm.mfp2 % q.fp
                      AND tanimoto_sml(q.fp, dm.mfp2) >= $2
                    ORDER BY similarity DESC
                    LIMIT $3
                )
                SELECT DISTINCT ON (sm.mol_id, mts.gene_symbol)
                    sm.mol_id,
                    sm.concept_id,
                    COALESCE(mts.concept_name, sm.molecule_name) as concept_name,
                    sm.molecule_name,
                    sm.canonical_smiles,
                    sm.chembl_id,
                    sm.similarity as tanimoto_similarity,
                    mts.gene_symbol,
                    mts.target_name,
                    mts.target_organism,
                    COALESCE(mts.best_ic50_nm, mts.best_ki_nm) as activity_value_nm,
                    CASE WHEN mts.best_ic50_nm IS NOT NULL THEN 'IC50' ELSE 'Ki' END as activity_type,
                    mts.best_pchembl as pchembl,
                    mts.n_total_measurements,
                    mts.data_confidence,
                    mts.sources
                FROM similar_mols sm
                LEFT JOIN dm_molecule_target_summary mts ON sm.concept_id = mts.concept_id
                WHERE (mts.best_pchembl >= $4 OR mts.best_pchembl IS NULL)
                ORDER BY sm.mol_id, mts.gene_symbol, mts.best_pchembl DESC NULLS LAST
            """
            params = [input.smiles, input.similarity_threshold, input.limit * 3, input.min_pchembl]
        else:
            sql = """
                WITH query_fp AS (
                    SELECT morganbv_fp(mol_from_smiles($1::cstring)) as fp
                )
                SELECT 
                    dm.mol_id,
                    dm.concept_id,
                    mc.preferred_name as concept_name,
                    dm.pref_name as molecule_name,
                    dm.canonical_smiles,
                    dm.chembl_id,
                    tanimoto_sml(q.fp, dm.mfp2)::float as tanimoto_similarity,
                    NULL as gene_symbol,
                    NULL as target_name,
                    NULL as target_organism,
                    NULL::numeric as activity_value_nm,
                    NULL as activity_type,
                    NULL::numeric as pchembl,
                    NULL::int as n_total_measurements,
                    NULL as data_confidence,
                    NULL::text[] as sources
                FROM dm_molecule dm
                CROSS JOIN query_fp q
                LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
                WHERE dm.mfp2 % q.fp
                  AND tanimoto_sml(q.fp, dm.mfp2) >= $2
                ORDER BY tanimoto_sml(q.fp, dm.mfp2) DESC
                LIMIT $3
            """
            params = [input.smiles, input.similarity_threshold, input.limit]
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        # Deduplicate and limit
        seen = set()
        unique_hits = []
        for h in hits:
            key = (h.mol_id, h.gene_symbol)
            if key not in seen:
                seen.add(key)
                unique_hits.append(h)
                if len(unique_hits) >= input.limit:
                    break
        
        smiles_display = input.smiles[:40] + "..." if len(input.smiles) > 40 else input.smiles
        
        return TargetSearchOutput(
            status="success" if unique_hits else "not_found",
            mode=input.mode,
            query_summary=f"similar to '{smiles_display}'",
            total_hits=len(unique_hits),
            hits=unique_hits
        )

    # =========================================================================
    # HANDLER: EXACT_STRUCTURE
    # =========================================================================
    
    async def _search_exact_structure(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find exact structure match."""
        conn = await self._get_conn()
        
        sql = """
            SELECT 
                dm.mol_id,
                dm.concept_id,
                mc.preferred_name as concept_name,
                dm.pref_name as molecule_name,
                dm.canonical_smiles,
                dm.chembl_id,
                1.0::float as tanimoto_similarity,
                NULL as gene_symbol,
                NULL as target_name,
                NULL as target_organism,
                NULL::numeric as activity_value_nm,
                NULL as activity_type,
                NULL::numeric as pchembl,
                NULL::int as n_total_measurements,
                NULL as data_confidence,
                NULL::text[] as sources
            FROM dm_molecule dm
            LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
            WHERE dm.inchi_key = mol_inchikey(mol_from_smiles($1::cstring))::text
            LIMIT $2
        """
        
        rows = await conn.execute_query(sql, input.smiles, input.limit)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"exact match for SMILES",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: SUBSTRUCTURE
    # =========================================================================
    
    # =========================================================================
    # HANDLER: SUBSTRUCTURE
    # =========================================================================
    
    async def _search_substructure(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find molecules containing a substructure."""
        conn = await self._get_conn()
        
        pattern = input.effective_smarts
        
        if input.include_activities:
            # Include activity data for matched molecules
            sql = """
                WITH matched_mols AS (
                    SELECT 
                        dm.mol_id,
                        dm.concept_id,
                        COALESCE(mc.preferred_name, dm.pref_name, dm.chembl_id) as concept_name,
                        dm.pref_name as molecule_name,
                        dm.canonical_smiles,
                        dm.chembl_id,
                        dm.is_salt,
                        dm.salt_form
                    FROM dm_molecule dm
                    LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
                    WHERE dm.mol @> qmol_from_smarts($1::cstring)
                    LIMIT $2
                )
                SELECT DISTINCT ON (mm.mol_id, mts.gene_symbol)
                    mm.mol_id,
                    mm.concept_id,
                    mm.concept_name,
                    mm.molecule_name,
                    mm.canonical_smiles,
                    mm.chembl_id,
                    NULL::float as tanimoto_similarity,
                    mts.gene_symbol,
                    mts.target_name,
                    mts.target_organism,
                    COALESCE(mts.best_ic50_nm, mts.best_ki_nm, mts.best_kd_nm, mts.best_ec50_nm) as activity_value_nm,
                    CASE 
                        WHEN mts.best_ic50_nm IS NOT NULL THEN 'IC50'
                        WHEN mts.best_ki_nm IS NOT NULL THEN 'Ki'
                        WHEN mts.best_kd_nm IS NOT NULL THEN 'Kd'
                        WHEN mts.best_ec50_nm IS NOT NULL THEN 'EC50'
                        ELSE NULL
                    END as activity_type,
                    mts.best_pchembl as pchembl,
                    mts.n_total_measurements,
                    mts.data_confidence,
                    mts.sources
                FROM matched_mols mm
                LEFT JOIN dm_molecule_target_summary mts ON mm.concept_id = mts.concept_id
                WHERE (mts.best_pchembl >= $3 OR mts.best_pchembl IS NULL)
                ORDER BY mm.mol_id, mts.gene_symbol, mts.best_pchembl DESC NULLS LAST
            """
            params = [pattern, input.limit * 5, input.min_pchembl]  # Multiply limit to account for multiple targets
        else:
            # Just return molecules without activity data
            sql = """
                SELECT 
                    dm.mol_id,
                    dm.concept_id,
                    COALESCE(mc.preferred_name, dm.pref_name, dm.chembl_id, 'MOL_' || dm.mol_id::TEXT) as concept_name,
                    dm.pref_name as molecule_name,
                    dm.canonical_smiles,
                    dm.chembl_id,
                    NULL::float as tanimoto_similarity,
                    NULL as gene_symbol,
                    NULL as target_name,
                    NULL as target_organism,
                    NULL::numeric as activity_value_nm,
                    NULL as activity_type,
                    NULL::numeric as pchembl,
                    NULL::int as n_total_measurements,
                    NULL as data_confidence,
                    NULL::text[] as sources
                FROM dm_molecule dm
                LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
                WHERE dm.mol @> qmol_from_smarts($1::cstring)
                LIMIT $2
            """
            params = [pattern, input.limit]
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        # Deduplicate if we included activities (one hit per mol_id for non-activity searches)
        if input.include_activities:
            seen = set()
            unique_hits = []
            for h in hits:
                key = (h.mol_id, h.gene_symbol)
                if key not in seen:
                    seen.add(key)
                    unique_hits.append(h)
                    if len(unique_hits) >= input.limit:
                        break
            hits = unique_hits
        
        pattern_display = pattern[:30] + "..." if len(pattern) > 30 else pattern
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"substructure '{pattern_display}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: TRIALS_FOR_DRUG
    # =========================================================================
    
    async def _search_trials_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find clinical trials for a drug."""
        conn = await self._get_conn()
        
        sql = """
            SELECT DISTINCT
                s.nct_id,
                s.brief_title as trial_title,
                s.overall_status as trial_status,
                s.phase,
                mc.concept_id,
                mc.preferred_name as concept_name,
                dm.pref_name as molecule_form,
                dm.salt_form,
                map.match_type,
                map.confidence
            FROM dm_molecule_concept mc
            JOIN map_ctgov_molecules map ON map.concept_id = mc.concept_id
            LEFT JOIN dm_molecule dm ON map.mol_id = dm.mol_id
            JOIN ctgov_studies s ON map.nct_id = s.nct_id
            WHERE mc.preferred_name ILIKE $1
               OR mc.concept_id IN (
                   SELECT DISTINCT dm2.concept_id
                   FROM dm_molecule_synonyms dms
                   JOIN dm_molecule dm2 ON dms.mol_id = dm2.mol_id
                   WHERE dms.synonym_lower ILIKE LOWER($1)
               )
        """
        
        params = [f"%{input.query}%"]
        param_idx = 2
        
        if input.trial_phase != TrialPhase.ANY:
            sql += f" AND s.phase ILIKE ${param_idx}"
            params.append(f"%{input.trial_phase.value}%")
            param_idx += 1
        
        if input.trial_status:
            sql += f" AND s.overall_status ILIKE ${param_idx}"
            params.append(f"%{input.trial_status}%")
            param_idx += 1
        
        sql += f" ORDER BY s.nct_id DESC LIMIT ${param_idx}"
        params.append(input.limit)
        
        rows = await conn.execute_query(sql, *params)
        hits = [
            ClinicalTrialHit(
                nct_id=r['nct_id'],
                trial_title=r['trial_title'],
                trial_status=r['trial_status'],
                phase=r['phase'],
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                molecule_form=r['molecule_form'],
                salt_form=r['salt_form'],
                match_type=r['match_type'],
                confidence=r['confidence']
            )
            for r in rows
        ]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"trials for '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: DRUG_PROFILE
    # =========================================================================
    
    async def _search_drug_profile(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get complete drug profile."""
        conn = await self._get_conn()
        
        # Get concept info
        sql = """
            SELECT DISTINCT ON (mc.concept_id)
                mc.concept_id,
                mc.preferred_name as concept_name,
                mc.parent_inchi_key_14,
                mc.n_forms,
                (SELECT COUNT(DISTINCT stereo_id) FROM dm_molecule WHERE concept_id = mc.concept_id) as n_stereo_variants,
                (SELECT COUNT(*) FROM dm_molecule WHERE concept_id = mc.concept_id AND is_salt = TRUE) as n_salt_forms,
                (SELECT COUNT(DISTINCT nct_id) FROM map_ctgov_molecules WHERE concept_id = mc.concept_id) as n_clinical_trials,
                (SELECT COUNT(DISTINCT gene_symbol) FROM dm_molecule_target_summary WHERE concept_id = mc.concept_id) as n_targets
            FROM dm_molecule_concept mc
            LEFT JOIN dm_molecule_synonyms dms ON mc.concept_id = (
                SELECT dm.concept_id FROM dm_molecule dm WHERE dm.mol_id = dms.mol_id LIMIT 1
            )
            WHERE mc.preferred_name ILIKE $1
               OR dms.synonym_lower ILIKE LOWER($1)
            LIMIT 1
        """
        
        rows = await conn.execute_query(sql, f"%{input.query}%")
        
        if not rows:
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"profile for '{input.query}'"
            )
        
        r = rows[0]
        concept_id = r['concept_id']
        
        profile = DrugProfileResult(
            concept_id=concept_id,
            concept_name=r['concept_name'],
            parent_inchi_key_14=r['parent_inchi_key_14'],
            n_forms=r['n_forms'] or 0,
            n_stereo_variants=r['n_stereo_variants'] or 0,
            n_salt_forms=r['n_salt_forms'] or 0,
            n_clinical_trials=r['n_clinical_trials'] or 0,
            n_targets=r['n_targets'] or 0
        )
        
        # Get top targets
        if input.include_activities:
            target_input = TargetSearchInput(
                mode=SearchMode.TARGETS_FOR_DRUG,
                query=input.query,
                min_pchembl=input.min_pchembl,
                limit=20
            )
            target_results = await self._search_targets_for_drug(target_input)
            profile.top_targets = target_results.hits
        
        # Get forms
        if input.include_forms:
            forms_sql = """
                SELECT mol_id, pref_name, inchi_key, canonical_smiles, 
                       is_salt, salt_form, stereo_type, chembl_id, drugcentral_id
                FROM dm_molecule
                WHERE concept_id = $1
                ORDER BY is_salt, stereo_type
                LIMIT 50
            """
            form_rows = await conn.execute_query(forms_sql, concept_id)
            profile.forms = [
                MoleculeForm(
                    mol_id=fr['mol_id'],
                    form_name=fr['pref_name'],
                    inchi_key=fr['inchi_key'],
                    canonical_smiles=fr['canonical_smiles'],
                    is_salt=fr['is_salt'],
                    salt_form=fr['salt_form'],
                    stereo_type=fr['stereo_type'],
                    chembl_id=fr['chembl_id'],
                    drugcentral_id=fr['drugcentral_id']
                )
                for fr in form_rows
            ]
        
        # Get recent trials
        if input.include_trials:
            trial_input = TargetSearchInput(
                mode=SearchMode.TRIALS_FOR_DRUG,
                query=input.query,
                limit=10
            )
            trial_results = await self._search_trials_for_drug(trial_input)
            profile.recent_trials = trial_results.hits
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"profile for '{input.query}'",
            total_hits=1,
            hits=[profile]
        )

    # =========================================================================
    # HANDLER: DRUG_FORMS
    # =========================================================================
    
    async def _search_drug_forms(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all forms of a drug."""
        conn = await self._get_conn()
        
        sql = """
            SELECT 
                dm.mol_id,
                dm.pref_name as form_name,
                dm.inchi_key,
                dm.canonical_smiles,
                dm.is_salt,
                dm.salt_form,
                dm.stereo_type,
                dm.chembl_id,
                dm.drugcentral_id
            FROM dm_molecule dm
            JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
            WHERE mc.preferred_name ILIKE $1
               OR mc.concept_id IN (
                   SELECT DISTINCT dm2.concept_id
                   FROM dm_molecule_synonyms dms
                   JOIN dm_molecule dm2 ON dms.mol_id = dm2.mol_id
                   WHERE dms.synonym_lower ILIKE LOWER($1)
               )
            ORDER BY dm.is_salt, dm.stereo_type
            LIMIT $2
        """
        
        rows = await conn.execute_query(sql, f"%{input.query}%", input.limit)
        hits = [
            MoleculeForm(
                mol_id=r['mol_id'],
                form_name=r['form_name'],
                inchi_key=r['inchi_key'],
                canonical_smiles=r['canonical_smiles'],
                is_salt=r['is_salt'],
                salt_form=r['salt_form'],
                stereo_type=r['stereo_type'],
                chembl_id=r['chembl_id'],
                drugcentral_id=r['drugcentral_id']
            )
            for r in rows
        ]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"forms of '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: COMPARE_DRUGS
    # =========================================================================
    
    async def _search_compare_drugs(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Compare drugs on a target."""
        conn = await self._get_conn()
        
        target = input.effective_target
        
        col_map = {
            ActivityType.IC50: ("best_ic50_nm", "median_ic50_nm", "n_ic50_measurements"),
            ActivityType.KI: ("best_ki_nm", "median_ki_nm", "n_ki_measurements"),
            ActivityType.KD: ("best_kd_nm", "median_kd_nm", "n_kd_measurements"),
            ActivityType.EC50: ("best_ec50_nm", "median_ec50_nm", "n_ec50_measurements"),
        }
        
        act_type = input.activity_type if input.activity_type != ActivityType.ALL else ActivityType.IC50
        best_col, median_col, count_col = col_map[act_type]
        
        drug_patterns = [f"%{name}%" for name in input.drug_names]
        
        sql = f"""
            WITH drug_data AS (
                SELECT 
                    mts.concept_name,
                    mts.{best_col} as best_value,
                    mts.{median_col} as median_value,
                    mts.{count_col} as n_measurements,
                    mts.data_confidence,
                    mts.sources
                FROM dm_molecule_target_summary mts
                WHERE mts.gene_symbol = UPPER($1)
                  AND mts.concept_name ILIKE ANY($2::text[])
                  AND mts.{best_col} IS NOT NULL
            ),
            best_overall AS (
                SELECT MIN(best_value) as min_value FROM drug_data
            )
            SELECT 
                d.concept_name as drug_name,
                d.best_value as activity_value_nm,
                d.median_value as median_value_nm,
                d.n_measurements,
                ROUND((d.best_value / NULLIF(b.min_value, 0))::numeric, 2) as fold_vs_best,
                d.data_confidence,
                d.sources
            FROM drug_data d
            CROSS JOIN best_overall b
            ORDER BY d.best_value ASC NULLS LAST
        """
        
        rows = await conn.execute_query(sql, target, drug_patterns)
        hits = [
            DrugComparisonHit(
                drug_name=r['drug_name'],
                activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else None,
                median_value_nm=float(r['median_value_nm']) if r['median_value_nm'] else None,
                n_measurements=r['n_measurements'] or 0,
                fold_vs_best=float(r['fold_vs_best']) if r['fold_vs_best'] else None,
                data_confidence=r['data_confidence'],
                sources=r['sources']
            )
            for r in rows
        ]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"compare {', '.join(input.drug_names)} on {target}",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: SELECTIVE_DRUGS
    # =========================================================================
    
    async def _search_selective_drugs(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find selective drugs."""
        conn = await self._get_conn()
        
        target = input.effective_target
        
        sql = """
            WITH on_target AS (
                SELECT concept_id, concept_name, best_pchembl,
                       COALESCE(best_ic50_nm, best_ki_nm) as on_target_nm,
                       data_confidence, sources
                FROM dm_molecule_target_summary
                WHERE gene_symbol = UPPER($1)
                  AND best_pchembl >= $2
            ),
            off_target AS (
                SELECT concept_id, 
                       MIN(COALESCE(best_ic50_nm, best_ki_nm)) as best_off_target_nm
                FROM dm_molecule_target_summary
                WHERE gene_symbol = ANY($3::text[])
                GROUP BY concept_id
            )
            SELECT 
                ot.concept_id,
                ot.concept_name,
                ot.best_pchembl as pchembl,
                ot.on_target_nm as activity_value_nm,
                oft.best_off_target_nm,
                ROUND((oft.best_off_target_nm / NULLIF(ot.on_target_nm, 0))::numeric, 1) as selectivity_fold,
                ot.data_confidence,
                ot.sources
            FROM on_target ot
            LEFT JOIN off_target oft ON ot.concept_id = oft.concept_id
            WHERE oft.best_off_target_nm IS NULL 
               OR (oft.best_off_target_nm / NULLIF(ot.on_target_nm, 0)) >= $4
            ORDER BY 
                CASE WHEN oft.best_off_target_nm IS NULL THEN 1 ELSE 0 END,
                (oft.best_off_target_nm / NULLIF(ot.on_target_nm, 0)) DESC NULLS LAST,
                ot.best_pchembl DESC
            LIMIT $5
        """
        
        off_targets_upper = [g.upper() for g in input.off_targets]
        rows = await conn.execute_query(
            sql, target, input.min_pchembl, off_targets_upper, 
            input.min_selectivity_fold, input.limit
        )
        
        hits = [
            DrugTargetHit(
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                gene_symbol=target.upper(),
                activity_type="IC50/Ki",
                activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else 0,
                pchembl=float(r['pchembl']) if r['pchembl'] else None,
                data_confidence=r['data_confidence'],
                sources=r['sources'],
                selectivity_fold=float(r['selectivity_fold']) if r['selectivity_fold'] else None
            )
            for r in rows
        ]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"selective for {target} vs {', '.join(input.off_targets)}",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: ACTIVITIES_FOR_DRUG (raw measurements)
    # =========================================================================
    
    async def _search_activities_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get raw activity measurements for a drug."""
        conn = await self._get_conn()
        
        sql = """
            SELECT 
                cta.mol_id,
                m.concept_id,
                cta.molecule_name,
                m.canonical_smiles,
                m.chembl_id,
                cta.target_id,
                cta.gene_symbol,
                t.protein_name as target_name,
                t.organism as target_organism,
                cta.activity_value as activity_value_nm,
                cta.activity_type,
                cta.pchembl_value as pchembl,
                1 as n_total_measurements,
                NULL as data_confidence,
                ARRAY[cta.source] as sources
            FROM dm_compound_target_activity cta
            JOIN dm_molecule m ON cta.mol_id = m.mol_id
            LEFT JOIN dm_target t ON cta.target_id = t.target_id
            WHERE (
                cta.molecule_name ILIKE $1
                OR m.concept_id IN (
                    SELECT DISTINCT dm2.concept_id
                    FROM dm_molecule_synonyms dms
                    JOIN dm_molecule dm2 ON dms.mol_id = dm2.mol_id
                    WHERE dms.synonym_lower ILIKE LOWER($1)
                )
            )
            AND cta.pchembl_value >= $2
        """
        
        params = [f"%{input.query}%", input.min_pchembl]
        param_idx = 3
        
        if input.activity_type != ActivityType.ALL:
            sql += f" AND cta.activity_type = ${param_idx}"
            params.append(input.activity_type.value)
            param_idx += 1
        
        sql += f" ORDER BY cta.pchembl_value DESC NULLS LAST LIMIT ${param_idx}"
        params.append(input.limit)
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"activities for '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: ACTIVITIES_FOR_TARGET (raw measurements)
    # =========================================================================
    
    async def _search_activities_for_target(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get raw activity measurements for a target."""
        conn = await self._get_conn()
        
        sql = """
            SELECT 
                cta.mol_id,
                m.concept_id,
                cta.molecule_name,
                m.canonical_smiles,
                m.chembl_id,
                cta.target_id,
                cta.gene_symbol,
                t.protein_name as target_name,
                t.organism as target_organism,
                cta.activity_value as activity_value_nm,
                cta.activity_type,
                cta.pchembl_value as pchembl,
                1 as n_total_measurements,
                NULL as data_confidence,
                ARRAY[cta.source] as sources
            FROM dm_compound_target_activity cta
            JOIN dm_molecule m ON cta.mol_id = m.mol_id
            LEFT JOIN dm_target t ON cta.target_id = t.target_id
            WHERE cta.gene_symbol = UPPER($1)
              AND cta.pchembl_value >= $2
        """
        
        params = [input.query, input.min_pchembl]
        param_idx = 3
        
        if input.activity_type != ActivityType.ALL:
            sql += f" AND cta.activity_type = ${param_idx}"
            params.append(input.activity_type.value)
            param_idx += 1
        
        sql += f" ORDER BY cta.pchembl_value DESC NULLS LAST LIMIT ${param_idx}"
        params.append(input.limit)
        
        rows = await conn.execute_query(sql, *params)
        hits = [self._row_to_drug_target_hit(r) for r in rows]
        
        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"activities for target '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _row_to_drug_target_hit(self, r: dict) -> DrugTargetHit:
        """Convert a database row to DrugTargetHit."""
        return DrugTargetHit(
            concept_id=r.get('concept_id'),
            concept_name=r.get('concept_name'),
            mol_id=r.get('mol_id'),
            molecule_name=r.get('molecule_name'),
            canonical_smiles=r.get('canonical_smiles'),
            chembl_id=r.get('chembl_id'),
            target_id=r.get('target_id'),
            gene_symbol=r.get('gene_symbol') or "N/A",
            target_name=r.get('target_name'),
            target_organism=r.get('target_organism'),
            activity_type=r.get('activity_type') or "N/A",
            activity_value_nm=float(r['activity_value_nm']) if r.get('activity_value_nm') else 0,
            pchembl=float(r['pchembl']) if r.get('pchembl') else None,
            n_measurements=r.get('n_total_measurements'),
            data_confidence=r.get('data_confidence'),
            sources=r.get('sources'),
            tanimoto_similarity=r.get('tanimoto_similarity'),
            selectivity_fold=r.get('selectivity_fold')
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def target_search_async(
    db_config: DatabaseConfig,
    input: TargetSearchInput
) -> TargetSearchOutput:
    """Convenience function for one-off searches."""
    searcher = PharmacologySearch(db_config)
    return await searcher.search(input)