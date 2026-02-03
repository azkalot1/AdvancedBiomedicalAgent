#!/usr/bin/env python3
# target_search.py
"""
Unified Pharmacology Search Module.

Key features:
    - Integrates drug mechanism data (curated MOA) alongside activity data (IC50/Ki)
    - Supports multiple modes for targets, drugs, trials, indications, pathways
    - Groups results by compound with nested target/mechanism lists
    - Enhanced diagnostics and warnings for agent-facing use

Data Sources:
    - dm_molecule / dm_molecule_concept / dm_molecule_synonyms
    - dm_compound_target_activity / dm_target / dm_target_uniprot_mappings
    - dm_mechanism_of_action / drugcentral_indications / drugcentral_drug_interactions
    - map_ctgov_molecules / rag_study_search
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

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
    TARGETS_FOR_DRUG = "targets_for_drug"
    TRIALS_FOR_DRUG = "trials_for_drug"
    DRUG_PROFILE = "drug_profile"
    DRUG_FORMS = "drug_forms"
    DRUGS_FOR_TARGET = "drugs_for_target"
    SELECTIVE_DRUGS = "selective_drugs"
    SIMILAR_MOLECULES = "similar_molecules"
    EXACT_STRUCTURE = "exact_structure"
    SUBSTRUCTURE = "substructure"
    COMPARE_DRUGS = "compare_drugs"
    ACTIVITIES_FOR_DRUG = "activities_for_drug"
    ACTIVITIES_FOR_TARGET = "activities_for_target"
    INDICATIONS_FOR_DRUG = "indications_for_drug"
    DRUGS_FOR_INDICATION = "drugs_for_indication"
    TARGET_PATHWAYS = "target_pathways"
    DRUG_INTERACTIONS = "drug_interactions"


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


class DataSource(str, Enum):
    """Source of target interaction data."""
    ACTIVITY = "activity"
    MECHANISM = "mechanism"
    BOTH = "both"


# =============================================================================
# RESULT MODELS
# =============================================================================

class TargetActivity(BaseModel):
    """A single quantitative activity measurement for a target."""
    gene_symbol: str
    target_name: str | None = None
    target_organism: str = "Homo sapiens"
    activity_type: str
    activity_value_nm: float
    pchembl: float | None = None
    n_measurements: int = 1
    data_confidence: str | None = None
    sources: list[str] = []

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


class TargetMechanism(BaseModel):
    """A curated drug mechanism of action for a target."""
    mechanism_of_action: str
    action_type: str
    target_name: str | None = None
    target_type: str | None = None
    target_organism: str = "Homo sapiens"
    gene_symbols: list[str] = []
    uniprot_accessions: list[str] = []
    n_components: int = 1
    direct_interaction: bool = False
    molecular_mechanism: bool = False
    disease_efficacy: bool = False
    ref_type: str | None = None
    ref_id: str | None = None
    ref_url: str | None = None


class CompoundTargetProfile(BaseModel):
    """Complete target profile for a single compound."""
    concept_id: int
    concept_name: str
    chembl_id: str | None = None
    canonical_smiles: str | None = None
    is_biotherapeutic: bool = False
    molecule_type: str | None = None
    biotherapeutic_type: str | None = None
    activities: list[TargetActivity] = []
    mechanisms: list[TargetMechanism] = []
    n_activity_targets: int = 0
    n_mechanism_targets: int = 0
    best_pchembl: float | None = None
    tanimoto_similarity: float | None = None

    @property
    def has_activity_data(self) -> bool:
        return len(self.activities) > 0

    @property
    def has_mechanism_data(self) -> bool:
        return len(self.mechanisms) > 0

    @property
    def all_target_genes(self) -> set[str]:
        genes = {a.gene_symbol for a in self.activities}
        for m in self.mechanisms:
            genes.update(m.gene_symbols)
        return genes


class DrugForTargetHit(BaseModel):
    """A drug that hits a specific target."""
    concept_id: int
    concept_name: str
    chembl_id: str | None = None
    canonical_smiles: str | None = None
    is_biotherapeutic: bool = False
    molecule_type: str | None = None
    biotherapeutic_type: str | None = None
    activity_type: str | None = None
    activity_value_nm: float | None = None
    pchembl: float | None = None
    n_measurements: int = 0
    data_confidence: str | None = None
    sources: list[str] = []
    mechanism_of_action: str | None = None
    action_type: str | None = None
    selectivity_fold: float | None = None

    @property
    def has_activity(self) -> bool:
        return self.activity_value_nm is not None

    @property
    def has_mechanism(self) -> bool:
        return self.mechanism_of_action is not None


class ClinicalTrialHit(BaseModel):
    """A clinical trial result."""
    nct_id: str
    trial_title: str
    trial_status: str | None = None
    phase: str | None = None
    concept_id: int | None = None
    concept_name: str | None = None
    molecule_form: str | None = None
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


class IndicationHit(BaseModel):
    """An indication linked to a drug concept."""
    indication_id: int
    preferred_name: str
    therapeutic_area: str | None = None
    max_phase: float | None = None
    is_approved: bool | None = None
    sources: list[str] = []
    ref_type: str | None = None
    ref_id: str | None = None
    ref_url: str | None = None


class DrugIndicationHit(BaseModel):
    """A drug linked to an indication."""
    concept_id: int
    concept_name: str
    max_phase: float | None = None
    is_approved: bool | None = None
    sources: list[str] = []


class TargetPathwayHit(BaseModel):
    """Protein class summary for a target gene."""
    gene_symbol: str
    protein_class_id: int | None = None
    protein_class_desc: str | None = None
    synonyms: list[str] = []


class DrugInteractionHit(BaseModel):
    """Metabolism-based interaction data."""
    concept_id: int | None = None
    concept_name: str | None = None
    chembl_id: str | None = None
    interaction_role: str | None = None
    enzyme_name: str | None = None
    pathway_key: str | None = None
    metabolite_chembl_id: str | None = None
    substrate_chembl_id: str | None = None


class DrugProfileResult(BaseModel):
    """Complete drug profile with all data."""
    concept_id: int
    concept_name: str
    canonical_smiles: str | None = None
    is_biotherapeutic: bool = False
    molecule_type: str | None = None
    biotherapeutic_type: str | None = None
    sequence_info: list[dict] = []
    n_forms: int = 0
    has_salt_forms: bool = False
    n_clinical_trials: int = 0
    n_activity_targets: int = 0
    n_mechanism_targets: int = 0
    forms: list[MoleculeForm] = []
    target_profile: CompoundTargetProfile | None = None
    recent_trials: list[ClinicalTrialHit] = []


# =============================================================================
# INPUT MODEL
# =============================================================================

class TargetSearchInput(BaseModel):
    """Unified input for all search types."""
    mode: SearchMode
    query: str | None = None
    smiles: str | None = None
    smarts: str | None = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_pchembl: float = Field(default=5.0, ge=0.0, le=15.0)
    activity_type: ActivityType = ActivityType.ALL
    min_confidence: DataConfidence = DataConfidence.ANY
    data_source: DataSource = DataSource.BOTH
    target: str | None = None
    off_targets: list[str] = Field(default_factory=list)
    min_selectivity_fold: float = Field(default=10.0, ge=1.0)
    drug_names: list[str] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    organism: str = "Homo sapiens"
    include_all_organisms: bool = False
    
    # Existing flags
    include_trials: bool = True
    include_forms: bool = True
    include_pathways: bool = False
    include_indications: bool = False
    
    # NEW: Enrichment flags for SIMILAR_MOLECULES mode
    include_activities: bool = True          # Target binding data (default ON)
    include_mechanisms: bool = False         # Curated MOA data
    include_trial_summary: bool = False      # Trial counts/phases per molecule
    include_indication_summary: bool = False # Indications per molecule
    include_aggregated_summary: bool = False # Cross-molecule summaries
    
    approval_status: Literal["approved", "investigational", "all"] = "all"
    max_off_targets: int | None = None

    def validate_for_mode(self) -> list[str]:
        """Validate required fields for the search mode."""
        errors = []
        
        if self.mode in [SearchMode.TARGETS_FOR_DRUG, SearchMode.TRIALS_FOR_DRUG,
                         SearchMode.DRUG_PROFILE, SearchMode.DRUG_FORMS,
                         SearchMode.ACTIVITIES_FOR_DRUG]:
            if not self.query:
                errors.append(f"'query' (drug name) is required for {self.mode.value}")
        
        elif self.mode in [SearchMode.DRUGS_FOR_TARGET, SearchMode.ACTIVITIES_FOR_TARGET]:
            if not self.query:
                errors.append(f"'query' (gene symbol like 'EGFR', 'ABL1') is required for {self.mode.value}")
        
        elif self.mode in [SearchMode.SIMILAR_MOLECULES, SearchMode.EXACT_STRUCTURE]:
            if not self.smiles:
                errors.append(f"'smiles' is required for {self.mode.value}")
        
        elif self.mode == SearchMode.SUBSTRUCTURE:
            if not self.smiles and not self.smarts:
                errors.append("'smiles' or 'smarts' pattern is required for substructure search")
        
        elif self.mode == SearchMode.COMPARE_DRUGS:
            if not self.target and not self.query:
                errors.append("'target' (gene symbol) is required for compare_drugs")
            if not self.drug_names or len(self.drug_names) < 2:
                errors.append("'drug_names' list with at least 2 drugs is required for compare_drugs")
        
        elif self.mode == SearchMode.SELECTIVE_DRUGS:
            if not self.target and not self.query:
                errors.append("'target' (gene symbol) is required for selective_drugs")
            if not self.off_targets:
                errors.append("'off_targets' list is required for selective_drugs")
        
        elif self.mode in [SearchMode.INDICATIONS_FOR_DRUG, SearchMode.DRUGS_FOR_INDICATION]:
            if not self.query:
                errors.append(f"'query' is required for {self.mode.value}")
        
        elif self.mode in [SearchMode.TARGET_PATHWAYS, SearchMode.DRUG_INTERACTIONS]:
            if not self.query:
                errors.append(f"'query' (gene symbol or drug name) is required for {self.mode.value}")
        
        return errors

    @property
    def effective_target(self) -> str | None:
        return self.target or self.query


# =============================================================================
# OUTPUT MODEL WITH ENHANCED DIAGNOSTICS
# =============================================================================

class AggregatedStructureSummary(BaseModel):
    """Cross-molecule aggregated summaries for structure search."""
    
    # Target distribution: gene → count of molecules hitting it
    target_distribution: dict[str, int] = Field(default_factory=dict)
    top_targets: list[str] = Field(default_factory=list)  # Top 10
    
    # Mechanism distribution: action_type → count
    mechanism_distribution: dict[str, int] = Field(default_factory=dict)
    
    # Indication distribution: indication → count
    indication_distribution: dict[str, int] = Field(default_factory=dict)
    top_indications: list[str] = Field(default_factory=list)  # Top 10
    
    # Clinical development summary
    phase_distribution: dict[str, int] = Field(default_factory=dict)
    n_molecules_in_trials: int = 0
    n_molecules_approved: int = 0
    
    # Activity summary
    n_molecules_with_activity: int = 0
    n_molecules_with_mechanisms: int = 0
    n_unique_targets: int = 0
    n_unique_indications: int = 0
    
    # Most potent similar molecule
    most_potent_molecule: str | None = None
    most_potent_pchembl: float | None = None


class SearchDiagnostics(BaseModel):
    """Diagnostic information about the search process."""
    search_steps: list[dict] = []
    query_info: dict = {}
    performance: dict = {}
    suggestions: list[str] = []


class TargetSearchOutput(BaseModel):
    """Universal search results container with enhanced diagnostics."""
    status: Literal["success", "not_found", "error", "invalid_input", "invalid_structure"] = "success"
    mode: SearchMode
    query_summary: str = ""
    total_hits: int = 0
    hits: list[Any] = []
    error: str | None = None
    warnings: list[str] = []
    input_params: dict = {}
    execution_time_ms: float | None = None
    diagnostics: SearchDiagnostics | None = None
    data_source_breakdown: dict[str, int] = {}
    confidence_explanation: str | None = None
    # Structure-search modes (SIMILAR_MOLECULES, EXACT_STRUCTURE): validation and cross-molecule summary
    structure_info: dict | None = None
    aggregated_summary: AggregatedStructureSummary | None = None

    def pretty_print(self, console: Console | None = None, max_hits: int = 20):
        """Pretty print results with improved formatting."""
        console = console or Console()
        
        status_color = {
            "success": "green",
            "not_found": "yellow",
            "error": "red",
            "invalid_input": "red",
            "invalid_structure": "red",
        }[self.status]
        
        console.print(Panel(
            f"[bold {status_color}]{self.status.upper()}[/bold {status_color}] | "
            f"Mode: [cyan]{self.mode.value}[/cyan] | "
            f"Query: [yellow]{self.query_summary}[/yellow] | "
            f"Hits: [bold]{self.total_hits}[/bold]"
            + (f" | Time: {self.execution_time_ms:.0f}ms" if self.execution_time_ms else ""),
            title="🔬 Pharmacology Search Results",
            border_style=status_color
        ))
        
        if self.error:
            console.print(f"[red]Error: {self.error}[/red]")
        
        for w in self.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")
        
        if self.diagnostics and self.diagnostics.suggestions:
            console.print("\n[cyan]Suggestions:[/cyan]")
            for s in self.diagnostics.suggestions:
                console.print(f"  → {s}")
        
        if not self.hits:
            console.print("[dim]No results found.[/dim]")
            return
        
        for i, hit in enumerate(self.hits[:max_hits], 1):
            self._print_hit(console, i, hit)
        
        if self.total_hits > max_hits:
            console.print(f"\n[dim]... and {self.total_hits - max_hits} more results[/dim]")

    def _print_hit(self, console: Console, rank: int, hit: Any):
        if isinstance(hit, CompoundTargetProfile):
            self._print_compound_profile(console, rank, hit)
        elif isinstance(hit, DrugForTargetHit):
            self._print_drug_for_target(console, rank, hit)
        elif isinstance(hit, DrugProfileResult):
            self._print_drug_profile(console, rank, hit)
        elif isinstance(hit, ClinicalTrialHit):
            self._print_trial(console, rank, hit)
        elif isinstance(hit, MoleculeForm):
            self._print_form(console, rank, hit)
        elif isinstance(hit, IndicationHit):
            self._print_indication(console, rank, hit)
        elif isinstance(hit, DrugIndicationHit):
            self._print_drug_indication(console, rank, hit)
        elif isinstance(hit, TargetPathwayHit):
            self._print_target_pathway(console, rank, hit)
        elif isinstance(hit, DrugInteractionHit):
            self._print_drug_interaction(console, rank, hit)
        else:
            console.print(f"  {rank}. {hit}")

    def _print_compound_profile(self, console: Console, rank: int, hit: CompoundTargetProfile):
        sim_badge = f" [blue](Sim: {hit.tanimoto_similarity:.2f})[/blue]" if hit.tanimoto_similarity else ""
        pchembl_badge = ""
        if hit.best_pchembl:
            p_color = "green" if hit.best_pchembl >= 8 else "yellow" if hit.best_pchembl >= 6 else "white"
            pchembl_badge = f" [{p_color}]pChEMBL={hit.best_pchembl:.2f}[/{p_color}]"
        
        console.print(f"\n[bold cyan]{rank}. {hit.concept_name}[/bold cyan]{sim_badge}{pchembl_badge}")
        
        if hit.chembl_id:
            console.print(f"   [dim]ChEMBL: {hit.chembl_id}[/dim]")
        
        if hit.canonical_smiles:
            smiles_preview = hit.canonical_smiles[:60] + "..." if len(hit.canonical_smiles) > 60 else hit.canonical_smiles
            console.print(f"   [dim]{smiles_preview}[/dim]")
        
        if hit.mechanisms:
            console.print(f"\n   [bold magenta]📋 Drug Mechanisms ({len(hit.mechanisms)}):[/bold magenta]")
            for m in hit.mechanisms[:5]:
                genes_str = ", ".join(m.gene_symbols[:3])
                if len(m.gene_symbols) > 3:
                    genes_str += f" (+{len(m.gene_symbols) - 3})"
                flags = []
                if m.disease_efficacy:
                    flags.append("✓efficacy")
                if m.direct_interaction:
                    flags.append("direct")
                flags_str = f" [dim][{', '.join(flags)}][/dim]" if flags else ""
                console.print(f"      • [magenta]{m.action_type}[/magenta]: {m.mechanism_of_action}")
                if genes_str:
                    console.print(f"        Target: {m.target_name or 'Unknown'} ({genes_str}){flags_str}")
            if len(hit.mechanisms) > 5:
                console.print(f"      [dim]... and {len(hit.mechanisms) - 5} more mechanisms[/dim]")
        
        if hit.activities:
            console.print(f"\n   [bold green]📊 Activity Data ({len(hit.activities)} targets):[/bold green]")
            for a in hit.activities[:8]:
                p_color = "green" if a.pchembl and a.pchembl >= 8 else "yellow" if a.pchembl and a.pchembl >= 6 else "white"
                pchembl_str = f" [{p_color}](pChEMBL={a.pchembl:.2f})[/{p_color}]" if a.pchembl else ""
                conf_str = f" [{a.data_confidence}]" if a.data_confidence else ""
                console.print(
                    f"      • [green]{a.gene_symbol}[/green]: "
                    f"{a.activity_type} = [{p_color}]{a.activity_value_nm:.1f} nM[/{p_color}]"
                    f"{pchembl_str}{conf_str}"
                )
            if len(hit.activities) > 8:
                console.print(f"      [dim]... and {len(hit.activities) - 8} more targets[/dim]")

    def _print_drug_for_target(self, console: Console, rank: int, hit: DrugForTargetHit):
        if hit.has_activity:
            p_color = "green" if hit.pchembl and hit.pchembl >= 8 else "yellow" if hit.pchembl and hit.pchembl >= 6 else "white"
            activity_str = f" | {hit.activity_type}: [{p_color}]{hit.activity_value_nm:.1f} nM[/{p_color}]"
            if hit.pchembl:
                activity_str += f" [{p_color}](pChEMBL={hit.pchembl:.2f})[/{p_color}]"
        else:
            activity_str = ""
        mech_str = ""
        if hit.has_mechanism:
            mech_str = f" | [magenta]{hit.action_type}: {hit.mechanism_of_action}[/magenta]"
        sel_str = f" [blue](Sel: {hit.selectivity_fold:.1f}x)[/blue]" if hit.selectivity_fold else ""
        console.print(f"  {rank}. [bold cyan]{hit.concept_name}[/bold cyan]{sel_str}{activity_str}{mech_str}")
        if hit.chembl_id:
            console.print(f"      [dim]ChEMBL: {hit.chembl_id}[/dim]")

    def _print_drug_profile(self, console: Console, rank: int, hit: DrugProfileResult):
        console.print(f"\n[bold cyan]{hit.concept_name}[/bold cyan]")
        console.print(f"  Concept ID: {hit.concept_id}")
        salt_info = " (includes salts)" if hit.has_salt_forms else ""
        console.print(f"  Forms: {hit.n_forms}{salt_info}")
        console.print(f"  Activity targets: {hit.n_activity_targets}")
        console.print(f"  Mechanism targets: {hit.n_mechanism_targets}")
        console.print(f"  Clinical Trials: {hit.n_clinical_trials}")
        if hit.target_profile:
            self._print_compound_profile(console, 0, hit.target_profile)

    def _print_trial(self, console: Console, rank: int, hit: ClinicalTrialHit):
        phase_color = {"PHASE1": "yellow", "PHASE2": "cyan", "PHASE3": "green", "PHASE4": "blue"}.get(hit.phase or "", "white")
        console.print(
            f"  {rank}. [bold]{hit.nct_id}[/bold] [{phase_color}]{hit.phase or 'N/A'}[/{phase_color}] "
            f"[{hit.trial_status}]\n      {hit.trial_title[:80]}..."
        )

    def _print_form(self, console: Console, rank: int, hit: MoleculeForm):
        name = hit.form_name or hit.chembl_id or f"MOL_{hit.mol_id}"
        salt_str = f" [yellow]({hit.salt_form})[/yellow]" if hit.salt_form else ""
        console.print(f"  {rank}. [cyan]{name}[/cyan]{salt_str}")

    def _print_indication(self, console: Console, rank: int, hit: IndicationHit):
        phase = f"phase {hit.max_phase}" if hit.max_phase is not None else "phase N/A"
        status = "approved" if hit.is_approved else "investigational"
        console.print(f"  {rank}. [bold]{hit.preferred_name}[/bold] [{status}, {phase}]")
        if hit.therapeutic_area:
            console.print(f"      [dim]Area: {hit.therapeutic_area}[/dim]")

    def _print_drug_indication(self, console: Console, rank: int, hit: DrugIndicationHit):
        phase = f"phase {hit.max_phase}" if hit.max_phase is not None else "phase N/A"
        status = "approved" if hit.is_approved else "investigational"
        console.print(f"  {rank}. [bold cyan]{hit.concept_name}[/bold cyan] [{status}, {phase}]")

    def _print_target_pathway(self, console: Console, rank: int, hit: TargetPathwayHit):
        desc = hit.protein_class_desc or "Unknown class"
        console.print(f"  {rank}. [bold]{hit.gene_symbol}[/bold] → {desc}")
        if hit.synonyms:
            console.print(f"      [dim]Synonyms: {', '.join(hit.synonyms[:5])}[/dim]")

    def _print_drug_interaction(self, console: Console, rank: int, hit: DrugInteractionHit):
        role = hit.interaction_role or "interaction"
        console.print(f"  {rank}. [bold]{hit.chembl_id or 'Unknown'}[/bold] ({role})")
        if hit.enzyme_name:
            console.print(f"      [dim]Enzyme: {hit.enzyme_name}[/dim]")
        if hit.pathway_key:
            console.print(f"      [dim]Pathway: {hit.pathway_key}[/dim]")


@dataclass
class SmilesValidationResult:
    """Result of SMILES validation and preprocessing."""
    is_valid: bool
    canonical_smiles: str | None = None
    original_smiles: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    warnings: list[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# =============================================================================
# MAIN SEARCH CLASS WITH ENHANCED DIAGNOSTICS
# =============================================================================

class PharmacologySearch:
    """Unified pharmacology search interface with enhanced diagnostics."""
    
    CONCEPT_LOOKUP_CTE = """
        WITH 
        exact_synonym AS (
            SELECT DISTINCT concept_id
            FROM (
                SELECT dm.concept_id
                FROM dm_molecule_synonyms dms
                JOIN dm_molecule dm ON dms.mol_id = dm.mol_id
                WHERE dms.synonym_lower = LOWER($1)
                UNION
                SELECT dbt.concept_id
                FROM dm_biotherapeutic_synonyms dbs
                JOIN dm_biotherapeutic dbt ON dbs.bio_id = dbt.bio_id
                WHERE dbs.synonym_lower = LOWER($1)
            ) syn
            LIMIT 5
        ),
        exact_name AS (
            SELECT concept_id
            FROM dm_molecule_concept
            WHERE LOWER(preferred_name) = LOWER($1)
            LIMIT 5
        ),
        trgm_match AS (
            SELECT DISTINCT concept_id
            FROM (
                SELECT dm.concept_id
                FROM dm_molecule_synonyms dms
                JOIN dm_molecule dm ON dms.mol_id = dm.mol_id
                WHERE dms.synonym_lower ILIKE $2
                UNION
                SELECT dbt.concept_id
                FROM dm_biotherapeutic_synonyms dbs
                JOIN dm_biotherapeutic dbt ON dbs.bio_id = dbt.bio_id
                WHERE dbs.synonym_lower ILIKE $2
            ) syn
            WHERE NOT EXISTS (SELECT 1 FROM exact_synonym)
              AND NOT EXISTS (SELECT 1 FROM exact_name)
            LIMIT 10
        ),
        matched_concepts AS (
            SELECT concept_id FROM exact_synonym
            UNION
            SELECT concept_id FROM exact_name
            UNION
            SELECT concept_id FROM trgm_match
        )
    """
    
    def __init__(self, db_config: DatabaseConfig, verbose: bool = False):
        self.db_config = db_config
        self._async_config: AsyncDatabaseConfig | None = None
        self.verbose = verbose
    
    async def _get_conn(self) -> AsyncDatabaseConfig:
        if self._async_config is None:
            self._async_config = await get_async_connection(self.db_config)
        return self._async_config

    @staticmethod
    def _summarize_sources(source_lists: list[list[str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sources in source_lists:
            for source in sources or []:
                counts[source] = counts.get(source, 0) + 1
        return counts

    async def _find_concept_ids_with_diagnostics(
        self, conn, query: str
    ) -> tuple[list[int], list[dict]]:
        """
        Find concept_ids with detailed diagnostics about what was searched.
        Returns (concept_ids, search_steps).
        """
        query_exact = query.strip()
        query_pattern = f"%{query}%"
        search_steps = []
        
        # Step 1: Exact synonym match
        exact_syn_sql = """
            SELECT DISTINCT concept_id, matched_on
            FROM (
                SELECT dm.concept_id, dms.synonym_lower as matched_on
                FROM dm_molecule_synonyms dms
                JOIN dm_molecule dm ON dms.mol_id = dm.mol_id
                WHERE dms.synonym_lower = LOWER($1)
                UNION
                SELECT dbt.concept_id, dbs.synonym_lower as matched_on
                FROM dm_biotherapeutic_synonyms dbs
                JOIN dm_biotherapeutic dbt ON dbs.bio_id = dbt.bio_id
                WHERE dbs.synonym_lower = LOWER($1)
            ) syn
            LIMIT 5
        """
        rows = await conn.execute_query(exact_syn_sql, query_exact)
        search_steps.append({
            "step": "exact_synonym_match",
            "query": query_exact.lower(),
            "found": len(rows),
            "matches": [r["matched_on"] for r in rows[:3]] if rows else []
        })
        if rows:
            return [r['concept_id'] for r in rows], search_steps
        
        # Step 2: Exact preferred name match
        exact_name_sql = """
            SELECT concept_id, preferred_name as matched_on
            FROM dm_molecule_concept
            WHERE LOWER(preferred_name) = LOWER($1)
            LIMIT 5
        """
        rows = await conn.execute_query(exact_name_sql, query_exact)
        search_steps.append({
            "step": "exact_name_match",
            "query": query_exact.lower(),
            "found": len(rows),
            "matches": [r["matched_on"] for r in rows[:3]] if rows else []
        })
        if rows:
            return [r['concept_id'] for r in rows], search_steps
        
        # Step 3: Partial synonym match (ILIKE)
        partial_syn_sql = """
            SELECT DISTINCT concept_id, matched_on
            FROM (
                SELECT dm.concept_id, dms.synonym_lower as matched_on
                FROM dm_molecule_synonyms dms
                JOIN dm_molecule dm ON dms.mol_id = dm.mol_id
                WHERE dms.synonym_lower ILIKE $1
                UNION
                SELECT dbt.concept_id, dbs.synonym_lower as matched_on
                FROM dm_biotherapeutic_synonyms dbs
                JOIN dm_biotherapeutic dbt ON dbs.bio_id = dbt.bio_id
                WHERE dbs.synonym_lower ILIKE $1
            ) syn
            LIMIT 10
        """
        rows = await conn.execute_query(partial_syn_sql, query_pattern)
        search_steps.append({
            "step": "partial_synonym_match",
            "query": query_pattern,
            "found": len(rows),
            "matches": [r["matched_on"] for r in rows[:3]] if rows else []
        })
        if rows:
            return [r['concept_id'] for r in rows], search_steps
        
        # Step 4: Partial name match
        partial_name_sql = """
            SELECT concept_id, preferred_name as matched_on
            FROM dm_molecule_concept
            WHERE preferred_name ILIKE $1
            LIMIT 10
        """
        rows = await conn.execute_query(partial_name_sql, query_pattern)
        search_steps.append({
            "step": "partial_name_match",
            "query": query_pattern,
            "found": len(rows),
            "matches": [r["matched_on"] for r in rows[:3]] if rows else []
        })
        if rows:
            return [r['concept_id'] for r in rows], search_steps
        
        return [], search_steps

    async def _find_concept_ids(self, conn, query: str) -> list[int]:
        """Simple version without diagnostics for internal use."""
        ids, _ = await self._find_concept_ids_with_diagnostics(conn, query)
        return ids

    def _get_mode_help(self, mode: SearchMode) -> dict:
        """Get help information for a search mode."""
        help_info = {
            SearchMode.TARGETS_FOR_DRUG: {
                "description": "Find all protein targets for a drug",
                "required": "query (drug name)",
                "examples": ["imatinib", "aspirin", "Gleevec", "pembrolizumab"],
                "tips": [
                    "Try generic name if brand name doesn't work",
                    "Check spelling carefully",
                    "Biologics (antibodies) may have limited target data"
                ]
            },
            SearchMode.DRUGS_FOR_TARGET: {
                "description": "Find all drugs that modulate a protein target",
                "required": "query (gene symbol)",
                "examples": ["EGFR", "ABL1", "JAK2", "BRAF", "CDK4"],
                "tips": [
                    "Use official HGNC gene symbols",
                    "Gene symbols are case-insensitive but should be uppercase",
                    "Try lowering min_pchembl for more results"
                ]
            },
            SearchMode.SIMILAR_MOLECULES: {
                "description": "Find structurally similar molecules",
                "required": "smiles (valid SMILES string)",
                "examples": ["CC(=O)Oc1ccccc1C(=O)O (aspirin)"],
                "tips": [
                    "Verify SMILES is valid before searching",
                    "Lower similarity_threshold for more results",
                    "Default threshold 0.7 finds similar scaffolds"
                ]
            },
            SearchMode.EXACT_STRUCTURE: {
                "description": "Find exact structure match",
                "required": "smiles (valid SMILES string)",
                "tips": [
                    "Uses high similarity matching (>0.98)",
                    "If not found, try search_similar_molecules"
                ]
            },
            SearchMode.SELECTIVE_DRUGS: {
                "description": "Find drugs selective for one target over others",
                "required": "target (gene symbol) + off_targets (list of gene symbols)",
                "examples": ["target='JAK2', off_targets=['JAK1', 'JAK3']"],
                "tips": [
                    "Lower min_selectivity_fold for more results",
                    "Lower min_pchembl to include weaker binders"
                ]
            },
            SearchMode.COMPARE_DRUGS: {
                "description": "Compare multiple drugs on a single target",
                "required": "target (gene symbol) + drug_names (list of 2+ drugs)",
                "examples": ["target='ABL1', drug_names=['imatinib', 'dasatinib']"],
                "tips": [
                    "All drugs must have activity data on the target",
                    "Check drug name spelling if no results"
                ]
            },
            SearchMode.INDICATIONS_FOR_DRUG: {
                "description": "Find indications for a drug",
                "required": "query (drug name)",
                "examples": ["imatinib", "pembrolizumab"],
                "tips": [
                    "Try generic name if brand name doesn't work",
                    "Approved indications have max_phase >= 4"
                ]
            },
            SearchMode.DRUGS_FOR_INDICATION: {
                "description": "Find drugs associated with an indication",
                "required": "query (indication name)",
                "examples": ["melanoma", "non-small cell lung cancer"],
                "tips": [
                    "Try broader disease terms if no results",
                    "Use approval_status='approved' for approved drugs only"
                ]
            },
            SearchMode.TARGET_PATHWAYS: {
                "description": "Get protein class summary for a target gene",
                "required": "query (gene symbol)",
                "examples": ["EGFR", "BRAF"],
                "tips": [
                    "Returns protein class hierarchy from ChEMBL",
                    "Useful for high-level target categorization"
                ]
            },
            SearchMode.DRUG_INTERACTIONS: {
                "description": "Find metabolism-based interactions for a drug",
                "required": "query (drug name)",
                "examples": ["warfarin", "midazolam"],
                "tips": [
                    "Uses ChEMBL metabolism records",
                    "May be sparse for biologics"
                ]
            }
        }
        return help_info.get(mode, {"description": mode.value, "tips": []})

    async def _validate_and_canonicalize_smiles(
        self,
        conn: AsyncDatabaseConfig,
        smiles: str
    ) -> SmilesValidationResult:
        """
        Validate and canonicalize SMILES with detailed error reporting.
        """
        warnings = []
        original = smiles
        
        # === Preprocessing ===
        smiles = smiles.strip()
        
        # Remove common prefixes
        for prefix in ["SMILES:", "smiles:", "SMILES=", "smiles=", "SMILES ", "smiles "]:
            if smiles.startswith(prefix):
                smiles = smiles[len(prefix):].strip()
                warnings.append(f"Removed prefix '{prefix.strip()}'")
                break
        
        # Remove surrounding quotes
        if (smiles.startswith('"') and smiles.endswith('"')) or \
           (smiles.startswith("'") and smiles.endswith("'")):
            smiles = smiles[1:-1]
            warnings.append("Removed surrounding quotes")
        
        # Remove newlines/tabs
        import re
        if re.search(r'[\n\t\r]', smiles):
            smiles = re.sub(r'[\n\t\r]+', '', smiles)
            warnings.append("Removed newline/tab characters")
        
        # Handle multiple space-separated tokens (take first)
        if ' ' in smiles and '.' not in smiles:
            parts = smiles.split()
            if len(parts) > 1:
                smiles = parts[0]
                warnings.append(f"Multiple tokens detected; using first: '{smiles[:30]}...'")
        
        # === Basic validation ===
        if not smiles:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original,
                error_message="Empty SMILES string after preprocessing",
                error_type="empty",
                warnings=warnings
            )
        
        if len(smiles) > 5000:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original,
                error_message=f"SMILES too long ({len(smiles)} chars, max 5000)",
                error_type="too_long",
                warnings=warnings
            )
        
        # Bracket balance
        if smiles.count('[') != smiles.count(']'):
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original,
                error_message=f"Unbalanced square brackets: {smiles.count('[')} '[' vs {smiles.count(']')} ']'",
                error_type="syntax",
                warnings=warnings
            )
        
        if smiles.count('(') != smiles.count(')'):
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original,
                error_message=f"Unbalanced parentheses: {smiles.count('(')} '(' vs {smiles.count(')')} ')'",
                error_type="syntax",
                warnings=warnings
            )
        
        # Check common typos before sending to RDKit
        typo_warnings = []
        if 'CL' in smiles.upper() and '[Cl]' not in smiles and 'Cl' not in smiles:
            typo_warnings.append("'CL' should likely be 'Cl' (chlorine)")
        if 'BR' in smiles.upper() and '[Br]' not in smiles and 'Br' not in smiles:
            typo_warnings.append("'BR' should likely be 'Br' (bromine)")
        
        # === RDKit validation ===
        try:
            sql = """
                SELECT 
                    mol_from_smiles($1::cstring) IS NOT NULL AS is_valid,
                    mol_to_smiles(mol_from_smiles($1::cstring)) AS canonical,
                    mol_formula(mol_from_smiles($1::cstring)) AS formula,
                    mol_amw(mol_from_smiles($1::cstring))::float AS mol_weight,
                    mol_numheavyatoms(mol_from_smiles($1::cstring))::int AS heavy_atoms
            """
            rows = await conn.execute_query(sql, smiles)
            
            if not rows or not rows[0]["is_valid"]:
                # Build detailed error message
                error_msg = self._build_smiles_error_message(smiles, typo_warnings)
                return SmilesValidationResult(
                    is_valid=False,
                    original_smiles=original,
                    error_message=error_msg,
                    error_type="parse_error",
                    warnings=warnings + typo_warnings
                )
            
            row = rows[0]
            canonical = row["canonical"]
            
            # Add info about the molecule
            if row.get("mol_weight") and row["mol_weight"] > 1500:
                warnings.append(f"Large molecule (MW: {row['mol_weight']:.0f})")
            if row.get("heavy_atoms") and row["heavy_atoms"] < 4:
                warnings.append(f"Very small molecule ({row['heavy_atoms']} heavy atoms)")
            if canonical and canonical != smiles:
                warnings.append("SMILES was canonicalized")
            
            return SmilesValidationResult(
                is_valid=True,
                canonical_smiles=canonical,
                original_smiles=original,
                warnings=warnings
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "smiles" in error_str or "mol_from" in error_str or "cstring" in error_str:
                error_msg = self._build_smiles_error_message(smiles, typo_warnings)
                return SmilesValidationResult(
                    is_valid=False,
                    original_smiles=original,
                    error_message=error_msg,
                    error_type="rdkit_error",
                    warnings=warnings + typo_warnings
                )
            else:
                return SmilesValidationResult(
                    is_valid=False,
                    original_smiles=original,
                    error_message=f"Database error: {e}",
                    error_type="database_error",
                    warnings=warnings
                )
    
    def _build_smiles_error_message(self, smiles: str, typo_warnings: list[str]) -> str:
        """Build detailed, actionable error message for invalid SMILES."""
        import re
        issues = typo_warnings.copy()
        
        # Check ring closure balance
        ring_digits = re.findall(r'(?<![%\d])(\d)', smiles)
        digit_counts: dict[str, int] = {}
        for d in ring_digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1
        odd_rings = [d for d, c in digit_counts.items() if c % 2 != 0]
        if odd_rings:
            issues.append(f"Unmatched ring closures: {odd_rings}")
        
        # Check for invalid patterns
        if '==' in smiles:
            issues.append("Invalid '==' (double bond is single '=')")
        if '##' in smiles:
            issues.append("Invalid '##' (triple bond is single '#')")
        if '[]' in smiles:
            issues.append("Empty brackets '[]' are invalid")
        
        if issues:
            return f"Invalid SMILES. Issues found: {'; '.join(issues)}"
        else:
            return (
                "Invalid SMILES structure. Please check: "
                "1) Element symbols (Cl not CL, Br not BR), "
                "2) Ring closures are paired, "
                "3) Brackets/parentheses are balanced, "
                "4) Aromatic atoms are lowercase (c,n,o,s) in aromatic rings"
            )


    def _build_not_found_diagnostics(
        self,
        mode: SearchMode,
        input: "TargetSearchInput",
        search_steps: list[dict] | None = None,
        extra_info: dict | None = None
    ) -> tuple[list[str], SearchDiagnostics]:
        """Build detailed diagnostics for not_found results."""
        warnings = []
        suggestions = []
        
        mode_help = self._get_mode_help(mode)
        
        if mode in [SearchMode.TARGETS_FOR_DRUG, SearchMode.DRUG_PROFILE,
                    SearchMode.DRUG_FORMS, SearchMode.TRIALS_FOR_DRUG]:
            warnings.append(f"No molecule found matching '{input.query}'")
            
            if search_steps:
                steps_tried = [s["step"] for s in search_steps]
                warnings.append(f"Search methods tried: {' → '.join(steps_tried)}")
            
            suggestions.extend([
                f"Check spelling of '{input.query}'",
                "Try the generic drug name instead of brand name",
                "Try alternative synonyms or trade names",
            ])
            
            if input.query and len(input.query) < 4:
                suggestions.append("Query may be too short - try full drug name")
            
        elif mode == SearchMode.DRUGS_FOR_TARGET:
            warnings.append(f"No drugs found for target '{input.query}'")
            
            if extra_info:
                if extra_info.get("activity_count", 0) == 0:
                    warnings.append(f"No activity data found for gene symbol '{input.query}'")
                if extra_info.get("mechanism_count", 0) == 0:
                    warnings.append(f"No mechanism data found for gene symbol '{input.query}'")
            
            suggestions.extend([
                f"Verify '{input.query}' is a valid HGNC gene symbol",
                f"Current min_pchembl={input.min_pchembl} - try lowering to 5.0 for more results",
                "Common gene symbols: EGFR, BRAF, ABL1, JAK2, CDK4, ROS1"
            ])
            
        elif mode in [SearchMode.SIMILAR_MOLECULES, SearchMode.EXACT_STRUCTURE]:
            smiles_preview = input.smiles[:50] + "..." if input.smiles and len(input.smiles) > 50 else input.smiles
            warnings.append(f"No molecules found matching structure")
            warnings.append(f"Query SMILES: {smiles_preview}")
            
            if mode == SearchMode.SIMILAR_MOLECULES:
                suggestions.append(f"Current similarity_threshold={input.similarity_threshold} - try lowering to 0.5")
            else:
                suggestions.append("Try search_similar_molecules with threshold 0.9 instead")
            
            suggestions.append("Verify SMILES string is valid using a chemical structure tool")
            
        elif mode == SearchMode.SELECTIVE_DRUGS:
            warnings.append(f"No selective drugs found for {input.effective_target} vs {input.off_targets}")
            
            suggestions.extend([
                f"Current min_selectivity_fold={input.min_selectivity_fold}x - try lowering to 5x",
                f"Current min_pchembl={input.min_pchembl} - try lowering to 5.0",
                "Verify all gene symbols are correct"
            ])
            
        elif mode == SearchMode.COMPARE_DRUGS:
            warnings.append(f"No comparison data found for drugs on {input.effective_target}")
            
            suggestions.extend([
                "Verify all drug names are spelled correctly",
                f"Verify '{input.effective_target}' is a valid gene symbol",
                "Not all drugs may have activity data on this target"
            ])
            
        elif mode == SearchMode.TRIALS_FOR_DRUG:
            warnings.append(f"No clinical trials found for '{input.query}'")
            
            suggestions.extend([
                "Drug may not have registered clinical trials",
                "Try searching with the INN (generic) name",
                "Biologics may be listed under different names"
            ])

        elif mode == SearchMode.INDICATIONS_FOR_DRUG:
            warnings.append(f"No indications found for '{input.query}'")
            suggestions.extend([
                "Check the drug name spelling",
                "Try the generic name or INN",
                "Set approval_status='all' to include investigational indications"
            ])

        elif mode == SearchMode.DRUGS_FOR_INDICATION:
            warnings.append(f"No drugs found for indication '{input.query}'")
            suggestions.extend([
                "Try broader disease terms",
                "Check spelling of the indication",
                "Set approval_status='all' to include investigational drugs"
            ])

        elif mode == SearchMode.TARGET_PATHWAYS:
            warnings.append(f"No protein class data found for '{input.query}'")
            suggestions.extend([
                "Verify the gene symbol is correct",
                "Try an official HGNC symbol (uppercase)"
            ])

        elif mode == SearchMode.DRUG_INTERACTIONS:
            warnings.append(f"No metabolism interactions found for '{input.query}'")
            suggestions.extend([
                "This dataset is sparse for some drugs",
                "Try an alternate drug synonym"
            ])
        
        # Add general tips from mode help
        if mode_help.get("tips"):
            suggestions.extend([f"Tip: {t}" for t in mode_help["tips"][:2]])
        
        diagnostics = SearchDiagnostics(
            search_steps=search_steps or [],
            query_info={
                "mode": mode.value,
                "query": input.query,
                "smiles": input.smiles[:50] + "..." if input.smiles and len(input.smiles) > 50 else input.smiles,
                "min_pchembl": input.min_pchembl,
                "data_source": input.data_source.value if input.data_source else None,
            },
            suggestions=suggestions
        )
        
        return warnings, diagnostics

    def _build_error_diagnostics(
        self,
        mode: SearchMode,
        input: "TargetSearchInput",
        error: Exception
    ) -> tuple[str, list[str], SearchDiagnostics]:
        """Build detailed diagnostics for error results."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        warnings = []
        suggestions = []
        
        if "smiles" in error_str or "mol_from_smiles" in error_str or "cstring" in error_str:
            error_msg = f"Invalid SMILES structure: {error}"
            warnings.append("The provided SMILES string could not be parsed")
            if input.smiles:
                warnings.append(f"SMILES: {input.smiles[:60]}...")
            suggestions.extend([
                "Verify SMILES syntax is correct",
                "Use a chemical structure editor to generate valid SMILES",
                "Check for missing or extra parentheses/brackets"
            ])
            
        elif "timeout" in error_str or "canceling statement" in error_str:
            error_msg = f"Query timed out: {error}"
            warnings.append("The search took too long and was cancelled")
            suggestions.extend([
                "Try a more specific search",
                "Reduce the 'limit' parameter",
                "For substructure search, use a more specific pattern"
            ])
            
        elif "connection" in error_str or "connect" in error_str:
            error_msg = f"Database connection error: {error}"
            warnings.append("Could not connect to the database")
            suggestions.append("This may be a temporary issue - try again in a moment")
            
        else:
            error_msg = f"{error_type}: {error}"
            warnings.append(f"An unexpected error occurred: {error_type}")
            suggestions.append("Check input parameters and try again")
        
        diagnostics = SearchDiagnostics(
            query_info={
                "mode": mode.value,
                "query": input.query,
                "smiles": input.smiles[:50] + "..." if input.smiles and len(input.smiles) > 50 else input.smiles,
            },
            suggestions=suggestions
        )
        
        return error_msg, warnings, diagnostics

    async def search(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Execute a search with enhanced diagnostics."""
        import time
        start_time = time.time()
        
        # Validate input
        validation_errors = input.validate_for_mode()
        if validation_errors:
            mode_help = self._get_mode_help(input.mode)
            suggestions = [
                f"Required for {input.mode.value}: {mode_help.get('required', 'see documentation')}",
            ]
            if mode_help.get("examples"):
                suggestions.append(f"Examples: {', '.join(mode_help['examples'][:3])}")
            
            return TargetSearchOutput(
                status="invalid_input",
                mode=input.mode,
                query_summary=input.query or input.smiles or "N/A",
                error="; ".join(validation_errors),
                warnings=[f"Validation failed for {input.mode.value}"],
                input_params=input.model_dump(exclude_none=True),
                diagnostics=SearchDiagnostics(suggestions=suggestions)
            )
        
        try:
            handler = self._get_handler(input.mode)
            results = await handler(input)
            results.execution_time_ms = (time.time() - start_time) * 1000
            results.input_params = input.model_dump(exclude_none=True)
            
            # Add performance info to diagnostics
            if results.diagnostics:
                results.diagnostics.performance = {
                    "execution_time_ms": results.execution_time_ms,
                    "total_hits": results.total_hits
                }
            
            return results
            
        except Exception as e:
            import traceback
            if self.verbose:
                traceback.print_exc()
            
            error_msg, warnings, diagnostics = self._build_error_diagnostics(input.mode, input, e)
            
            return TargetSearchOutput(
                status="error",
                mode=input.mode,
                query_summary=input.query or input.smiles or "N/A",
                error=error_msg,
                warnings=warnings,
                input_params=input.model_dump(exclude_none=True),
                execution_time_ms=(time.time() - start_time) * 1000,
                diagnostics=diagnostics
            )

    def _get_handler(self, mode: SearchMode):
        """Get handler function for a search mode."""
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
            SearchMode.INDICATIONS_FOR_DRUG: self._search_indications_for_drug,
            SearchMode.DRUGS_FOR_INDICATION: self._search_drugs_for_indication,
            SearchMode.TARGET_PATHWAYS: self._search_target_pathways,
            SearchMode.DRUG_INTERACTIONS: self._search_drug_interactions,
        }
        return handlers[mode]

    # =========================================================================
    # HANDLER: TARGETS_FOR_DRUG
    # =========================================================================
    
    async def _search_targets_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all targets for a drug with enhanced diagnostics."""
        conn = await self._get_conn()
        
        # Find concepts with diagnostics
        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        
        if not concept_ids:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"targets for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        # Get concept details
        concept_sql = """
            SELECT DISTINCT ON (mc.concept_id)
                mc.concept_id,
                mc.preferred_name as concept_name,
                COALESCE(dm.chembl_id, dbt.chembl_id) as chembl_id,
                dm.canonical_smiles,
                mc.is_biotherapeutic,
                dbt.molecule_type,
                dbt.biotherapeutic_type
            FROM dm_molecule_concept mc
            LEFT JOIN dm_molecule dm ON dm.concept_id = mc.concept_id AND dm.is_salt = FALSE
            LEFT JOIN dm_biotherapeutic dbt ON dbt.concept_id = mc.concept_id
            WHERE mc.concept_id = ANY($1::bigint[])
            ORDER BY mc.concept_id, dm.mol_id
            LIMIT 10
        """
        concept_rows = await conn.execute_query(concept_sql, concept_ids)
        
        profiles = []
        
        for concept_row in concept_rows:
            concept_id = concept_row['concept_id']
            
            # Get activity data
            activities = []
            if input.data_source in [DataSource.ACTIVITY, DataSource.BOTH]:
                activity_sql = """
                    SELECT 
                        gene_symbol, target_name, target_organism,
                        COALESCE(best_ic50_nm, best_ki_nm, best_kd_nm, best_ec50_nm) as activity_value_nm,
                        CASE 
                            WHEN best_ic50_nm IS NOT NULL THEN 'IC50'
                            WHEN best_ki_nm IS NOT NULL THEN 'Ki'
                            WHEN best_kd_nm IS NOT NULL THEN 'Kd'
                            ELSE 'EC50'
                        END as activity_type,
                        best_pchembl as pchembl,
                        n_total_measurements,
                        data_confidence,
                        sources
                    FROM dm_molecule_target_summary
                    WHERE concept_id = $1 AND best_pchembl >= $2
                    ORDER BY best_pchembl DESC NULLS LAST
                    LIMIT $3
                """
                activity_rows = await conn.execute_query(
                    activity_sql, concept_id, input.min_pchembl, input.limit
                )
                
                activities = [
                    TargetActivity(
                        gene_symbol=r['gene_symbol'],
                        target_name=r['target_name'],
                        target_organism=r['target_organism'] or "Homo sapiens",
                        activity_type=r['activity_type'] or "Unknown",
                        activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else 0,
                        pchembl=float(r['pchembl']) if r['pchembl'] else None,
                        n_measurements=r['n_total_measurements'] or 1,
                        data_confidence=r['data_confidence'],
                        sources=r['sources'] or []
                    )
                    for r in activity_rows
                ]
            
            # Get mechanism data
            mechanisms = []
            if input.data_source in [DataSource.MECHANISM, DataSource.BOTH]:
                mechanism_sql = """
                    SELECT 
                        mechanism_of_action, action_type, target_name, target_type,
                        target_organism, gene_symbols, uniprot_accessions, n_components,
                        direct_interaction, molecular_mechanism, disease_efficacy,
                        ref_type, ref_id, ref_url
                    FROM dm_drug_mechanism
                    WHERE concept_id = $1
                    ORDER BY action_type, target_name
                    LIMIT $2
                """
                mechanism_rows = await conn.execute_query(mechanism_sql, concept_id, input.limit)
                
                mechanisms = [
                    TargetMechanism(
                        mechanism_of_action=r['mechanism_of_action'] or "Unknown",
                        action_type=r['action_type'] or "Unknown",
                        target_name=r['target_name'],
                        target_type=r['target_type'],
                        target_organism=r['target_organism'] or "Homo sapiens",
                        gene_symbols=r['gene_symbols'] or [],
                        uniprot_accessions=r['uniprot_accessions'] or [],
                        n_components=r['n_components'] or 1,
                        direct_interaction=r['direct_interaction'] or False,
                        molecular_mechanism=r['molecular_mechanism'] or False,
                        disease_efficacy=r['disease_efficacy'] or False,
                        ref_type=r['ref_type'],
                        ref_id=r['ref_id'],
                        ref_url=r['ref_url']
                    )
                    for r in mechanism_rows
                ]
            
            if activities or mechanisms:
                best_pchembl = max((a.pchembl for a in activities if a.pchembl), default=None)
                
                profiles.append(CompoundTargetProfile(
                    concept_id=concept_id,
                    concept_name=concept_row['concept_name'],
                    chembl_id=concept_row['chembl_id'],
                    canonical_smiles=concept_row['canonical_smiles'],
                    is_biotherapeutic=concept_row['is_biotherapeutic'] or False,
                    molecule_type=concept_row['molecule_type'],
                    biotherapeutic_type=concept_row['biotherapeutic_type'],
                    activities=activities,
                    mechanisms=mechanisms,
                    n_activity_targets=len(activities),
                    n_mechanism_targets=len(mechanisms),
                    best_pchembl=best_pchembl
                ))
        
        # Build warnings for edge cases
        warnings = []
        diagnostics = SearchDiagnostics(search_steps=search_steps)
        
        if concept_rows and not profiles:
            warnings.append(f"Drug '{input.query}' found but no target data available")
            if input.data_source == DataSource.ACTIVITY:
                diagnostics.suggestions.append("Try data_source='both' to include mechanism data")
            if input.min_pchembl > 5.0:
                diagnostics.suggestions.append(f"Try lowering min_pchembl (current: {input.min_pchembl})")
        
        if profiles and not any(p.activities for p in profiles):
            warnings.append("No quantitative activity data found - showing mechanism data only")
        
        if profiles and not any(p.mechanisms for p in profiles):
            warnings.append("No curated mechanism data found - showing activity data only")

        source_lists: list[list[str]] = []
        for profile in profiles:
            source_lists.extend([a.sources for a in profile.activities])
            source_lists.extend([m.ref_type.split(",") if m.ref_type else [] for m in profile.mechanisms])
        data_source_breakdown = self._summarize_sources(source_lists)
        confidence_explanation = (
            f"Activity data filtered at pChEMBL >= {input.min_pchembl}; "
            f"data_source={input.data_source.value}"
        )

        return TargetSearchOutput(
            status="success" if profiles else "not_found",
            mode=input.mode,
            query_summary=f"targets for '{input.query}'",
            total_hits=len(profiles),
            hits=profiles,
            warnings=warnings,
            diagnostics=diagnostics if warnings or self.verbose else None,
            data_source_breakdown=data_source_breakdown,
            confidence_explanation=confidence_explanation
        )

    # =========================================================================
    # HANDLER: DRUGS_FOR_TARGET
    # =========================================================================
    
    async def _search_drugs_for_target(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all drugs for a target with enhanced diagnostics.
        
        Respects data_source parameter:
        - ACTIVITY: Only quantitative binding/activity data from assays
        - MECHANISM: Only curated drug mechanism data (approved drugs with known MOA)
        - BOTH: Combines both sources
        """
        conn = await self._get_conn()
        
        gene_symbol = input.query.upper()
        activity_rows: list[dict] = []
        mechanism_rows: list[dict] = []
        
        # Get activity data only if requested
        if input.data_source in [DataSource.ACTIVITY, DataSource.BOTH]:
            activity_sql = """
                SELECT 
                    concept_id, concept_name,
                    representative_smiles as canonical_smiles,
                    COALESCE(best_ic50_nm, best_ki_nm, best_kd_nm, best_ec50_nm) as activity_value_nm,
                    CASE 
                        WHEN best_ic50_nm IS NOT NULL THEN 'IC50'
                        WHEN best_ki_nm IS NOT NULL THEN 'Ki'
                        WHEN best_kd_nm IS NOT NULL THEN 'Kd'
                        WHEN best_ec50_nm IS NOT NULL THEN 'EC50'
                    END as activity_type,
                    best_pchembl as pchembl,
                    n_total_measurements,
                    data_confidence,
                    sources
                FROM dm_molecule_target_summary
                WHERE gene_symbol = $1 AND best_pchembl >= $2
                ORDER BY best_pchembl DESC NULLS LAST
                LIMIT $3
            """
            activity_rows = await conn.execute_query(activity_sql, gene_symbol, input.min_pchembl, input.limit)
        
        # Get mechanism data only if requested
        if input.data_source in [DataSource.MECHANISM, DataSource.BOTH]:
            mechanism_sql = """
                SELECT DISTINCT
                    concept_id, concept_name, chembl_id,
                    mechanism_of_action, action_type
                FROM dm_drug_mechanism
                WHERE $1 = ANY(gene_symbols)
                LIMIT $2
            """
            mechanism_rows = await conn.execute_query(mechanism_sql, gene_symbol, input.limit)
        
        mechanism_lookup = {
            r['concept_id']: (r['mechanism_of_action'], r['action_type'])
            for r in mechanism_rows
        }

        concept_ids = list({r['concept_id'] for r in activity_rows} | {r['concept_id'] for r in mechanism_rows})
        concept_lookup = {}
        if concept_ids:
            concept_info_sql = """
                SELECT
                    mc.concept_id,
                    mc.is_biotherapeutic,
                    dbt.molecule_type,
                    dbt.biotherapeutic_type
                FROM dm_molecule_concept mc
                LEFT JOIN dm_biotherapeutic dbt ON dbt.concept_id = mc.concept_id
                WHERE mc.concept_id = ANY($1::bigint[])
            """
            concept_info_rows = await conn.execute_query(concept_info_sql, concept_ids)
            concept_lookup = {
                r['concept_id']: r for r in concept_info_rows
            }
        
        hits_by_concept: dict[int, DrugForTargetHit] = {}
        
        # Process activity data
        for r in activity_rows:
            concept_id = r['concept_id']
            mech = mechanism_lookup.get(concept_id, (None, None))
            concept_info = concept_lookup.get(concept_id, {})
            
            hits_by_concept[concept_id] = DrugForTargetHit(
                concept_id=concept_id,
                concept_name=r['concept_name'],
                canonical_smiles=r['canonical_smiles'],
                is_biotherapeutic=concept_info.get('is_biotherapeutic') or False,
                molecule_type=concept_info.get('molecule_type'),
                biotherapeutic_type=concept_info.get('biotherapeutic_type'),
                activity_type=r['activity_type'],
                activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else None,
                pchembl=float(r['pchembl']) if r['pchembl'] else None,
                n_measurements=r['n_total_measurements'] or 0,
                data_confidence=r['data_confidence'],
                sources=r['sources'] or [],
                mechanism_of_action=mech[0],
                action_type=mech[1]
            )
        
        # Process mechanism data (add drugs not already in activity data)
        for r in mechanism_rows:
            concept_id = r['concept_id']
            if concept_id not in hits_by_concept:
                concept_info = concept_lookup.get(concept_id, {})
                hits_by_concept[concept_id] = DrugForTargetHit(
                    concept_id=concept_id,
                    concept_name=r['concept_name'],
                    chembl_id=r['chembl_id'],
                    is_biotherapeutic=concept_info.get('is_biotherapeutic') or False,
                    molecule_type=concept_info.get('molecule_type'),
                    biotherapeutic_type=concept_info.get('biotherapeutic_type'),
                    mechanism_of_action=r['mechanism_of_action'],
                    action_type=r['action_type']
                )
        
        # Sort based on data source:
        # - MECHANISM: Sort by name (curated approved drugs)
        # - ACTIVITY/BOTH: Sort by pchembl (binding affinity)
        if input.data_source == DataSource.MECHANISM:
            hits = sorted(
                hits_by_concept.values(),
                key=lambda h: (h.concept_name or "").lower()
            )
        else:
            hits = sorted(
                hits_by_concept.values(),
                key=lambda h: (h.pchembl is None, -(h.pchembl or 0))
            )
        
        # Build diagnostics for not_found
        if not hits:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input,
                extra_info={
                    "activity_count": len(activity_rows),
                    "mechanism_count": len(mechanism_rows)
                }
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"drugs for '{gene_symbol}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        # Build warnings for partial results
        warnings = []
        if input.data_source == DataSource.BOTH:
            if activity_rows and not mechanism_rows:
                warnings.append(f"No curated mechanism data found for {gene_symbol}")
            if mechanism_rows and not activity_rows:
                warnings.append(f"No quantitative activity data found for {gene_symbol} at pChEMBL >= {input.min_pchembl}")

        source_lists = [h.sources for h in hits]
        data_source_breakdown = self._summarize_sources(source_lists)
        confidence_explanation = (
            f"Activity data filtered at pChEMBL >= {input.min_pchembl}; "
            f"data_source={input.data_source.value}"
        )

        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"drugs for '{gene_symbol}'",
            total_hits=len(hits),
            hits=hits[:input.limit],
            warnings=warnings,
            data_source_breakdown=data_source_breakdown,
            confidence_explanation=confidence_explanation
        )

    # =========================================================================
    # HANDLER: DRUG_PROFILE
    # =========================================================================
    
    async def _search_drug_profile(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get complete drug profile with enhanced diagnostics."""
        conn = await self._get_conn()
        
        # Find concept with diagnostics
        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        
        if not concept_ids:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"profile for '{input.query}'",
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        query_exact = input.query.strip()
        query_pattern = f"%{input.query}%"
        
        concept_sql = f"""
            {self.CONCEPT_LOOKUP_CTE}
            SELECT DISTINCT ON (mc.concept_id)
                mc.concept_id,
                mc.preferred_name as concept_name,
                dm.canonical_smiles,
                mc.n_forms,
                mc.has_salt_forms,
                mc.is_biotherapeutic,
                dbt.molecule_type,
                dbt.biotherapeutic_type,
                (SELECT COUNT(DISTINCT nct_id) FROM map_ctgov_molecules WHERE concept_id = mc.concept_id) as n_clinical_trials,
                (SELECT COUNT(DISTINCT gene_symbol) FROM dm_molecule_target_summary WHERE concept_id = mc.concept_id) as n_activity_targets,
                (SELECT COUNT(*) FROM dm_drug_mechanism WHERE concept_id = mc.concept_id) as n_mechanism_targets
            FROM matched_concepts m
            JOIN dm_molecule_concept mc ON mc.concept_id = m.concept_id
            LEFT JOIN dm_molecule dm ON dm.concept_id = m.concept_id AND dm.is_salt = FALSE
            LEFT JOIN dm_biotherapeutic dbt ON dbt.concept_id = m.concept_id
            ORDER BY mc.concept_id, dm.mol_id
            LIMIT 1
        """
        
        rows = await conn.execute_query(concept_sql, query_exact, query_pattern)
        
        if not rows:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"profile for '{input.query}'",
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        r = rows[0]
        concept_id = r['concept_id']
        
        # Get target profile
        target_input = TargetSearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query=r['concept_name'],
            min_pchembl=input.min_pchembl,
            data_source=input.data_source,
            limit=30
        )
        target_results = await self._search_targets_for_drug(target_input)
        target_profile = target_results.hits[0] if target_results.hits else None
        
        # Get forms
        forms = []
        if input.include_forms and not r['is_biotherapeutic']:
            forms_sql = """
                SELECT mol_id, pref_name, inchi_key, canonical_smiles, 
                       is_salt, salt_form, stereo_type, chembl_id
                FROM dm_molecule
                WHERE concept_id = $1
                ORDER BY is_salt, stereo_type
                LIMIT 20
            """
            form_rows = await conn.execute_query(forms_sql, concept_id)
            forms = [
                MoleculeForm(
                    mol_id=fr['mol_id'],
                    form_name=fr['pref_name'],
                    inchi_key=fr['inchi_key'],
                    canonical_smiles=fr['canonical_smiles'],
                    is_salt=fr['is_salt'],
                    salt_form=fr['salt_form'],
                    stereo_type=fr['stereo_type'],
                    chembl_id=fr['chembl_id']
                )
                for fr in form_rows
            ]
        
        sequence_info = []
        if r['is_biotherapeutic']:
            sequence_sql = """
                SELECT
                    dbc.component_type,
                    dbc.description,
                    dbc.sequence,
                    dbc.sequence_length,
                    dbc.organism,
                    dbc.tax_id,
                    dbc.uniprot_accession
                FROM dm_biotherapeutic_component dbc
                JOIN dm_biotherapeutic dbt ON dbc.bio_id = dbt.bio_id
                WHERE dbt.concept_id = $1
                ORDER BY dbc.component_id
                LIMIT 20
            """
            sequence_rows = await conn.execute_query(sequence_sql, concept_id)
            sequence_info = [
                {
                    "component_type": sr["component_type"],
                    "description": sr["description"],
                    "sequence": sr["sequence"],
                    "sequence_length": sr["sequence_length"],
                    "organism": sr["organism"],
                    "tax_id": sr["tax_id"],
                    "uniprot_accession": sr["uniprot_accession"],
                }
                for sr in sequence_rows
            ]
        
        # Get trials
        trials = []
        if input.include_trials:
            trial_sql = """
                SELECT DISTINCT
                    s.nct_id,
                    s.brief_title as trial_title,
                    s.overall_status as trial_status,
                    s.phase,
                    map.match_type,
                    map.confidence
                FROM map_ctgov_molecules map
                JOIN ctgov_studies s ON map.nct_id = s.nct_id
                WHERE map.concept_id = $1
                ORDER BY s.nct_id DESC
                LIMIT 10
            """
            trial_rows = await conn.execute_query(trial_sql, concept_id)
            trials = [
                ClinicalTrialHit(
                    nct_id=tr['nct_id'],
                    trial_title=tr['trial_title'],
                    trial_status=tr['trial_status'],
                    phase=tr['phase'],
                    concept_id=concept_id,
                    concept_name=r['concept_name'],
                    match_type=tr['match_type'],
                    confidence=tr['confidence']
                )
                for tr in trial_rows
            ]
        
        profile = DrugProfileResult(
            concept_id=concept_id,
            concept_name=r['concept_name'],
            canonical_smiles=r['canonical_smiles'],
            is_biotherapeutic=r['is_biotherapeutic'] or False,
            molecule_type=r['molecule_type'],
            biotherapeutic_type=r['biotherapeutic_type'],
            sequence_info=sequence_info,
            n_forms=r['n_forms'] or 0,
            has_salt_forms=r['has_salt_forms'] or False,
            n_clinical_trials=r['n_clinical_trials'] or 0,
            n_activity_targets=r['n_activity_targets'] or 0,
            n_mechanism_targets=r['n_mechanism_targets'] or 0,
            forms=forms,
            target_profile=target_profile,
            recent_trials=trials
        )
        
        # Build warnings
        warnings = []
        if r['n_activity_targets'] == 0 and r['n_mechanism_targets'] == 0:
            warnings.append("No target data available for this drug")
        if r['n_clinical_trials'] == 0:
            warnings.append("No clinical trial associations found")
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"profile for '{input.query}'",
            total_hits=1,
            hits=[profile],
            warnings=warnings,
            diagnostics=SearchDiagnostics(search_steps=search_steps) if self.verbose else None
        )

    # =========================================================================
    # HANDLER: TRIALS_FOR_DRUG
    # =========================================================================
    
    async def _search_trials_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find clinical trials for a drug with enhanced diagnostics."""
        conn = await self._get_conn()
        
        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        
        if not concept_ids:
            # Try direct search in interventions as fallback for biologics
            direct_sql = """
                SELECT DISTINCT
                    s.nct_id,
                    s.brief_title as trial_title,
                    s.overall_status as trial_status,
                    s.phase,
                    NULL::bigint as concept_id,
                    i.name as concept_name,
                    NULL as molecule_form,
                    'direct_intervention' as match_type,
                    NULL::float as confidence
                FROM ctgov_interventions i
                JOIN ctgov_studies s ON i.nct_id = s.nct_id
                WHERE i.intervention_type IN ('DRUG', 'BIOLOGICAL')
                  AND i.name ILIKE $1
                ORDER BY s.nct_id DESC
                LIMIT $2
            """
            rows = await conn.execute_query(direct_sql, f"%{input.query}%", input.limit)
            
            if rows:
                hits = [
                    ClinicalTrialHit(
                        nct_id=r['nct_id'],
                        trial_title=r['trial_title'],
                        trial_status=r['trial_status'],
                        phase=r['phase'],
                        concept_id=r['concept_id'],
                        concept_name=r['concept_name'],
                        molecule_form=r['molecule_form'],
                        match_type=r['match_type'],
                        confidence=r['confidence']
                    )
                    for r in rows
                ]
                return TargetSearchOutput(
                    status="success",
                    mode=input.mode,
                    query_summary=f"trials for '{input.query}' (direct match)",
                    total_hits=len(hits),
                    hits=hits,
                    warnings=["Drug not found in molecule database - showing direct intervention matches"]
                )
            
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"trials for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        sql = """
            SELECT DISTINCT
                s.nct_id,
                s.brief_title as trial_title,
                s.overall_status as trial_status,
                s.phase,
                mc.concept_id,
                mc.preferred_name as concept_name,
                dm.pref_name as molecule_form,
                map.match_type,
                map.confidence
            FROM map_ctgov_molecules map
            JOIN ctgov_studies s ON map.nct_id = s.nct_id
            JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
            LEFT JOIN dm_molecule dm ON map.mol_id = dm.mol_id
            WHERE map.concept_id = ANY($1::bigint[])
            ORDER BY s.nct_id DESC
            LIMIT $2
        """
        
        rows = await conn.execute_query(sql, concept_ids, input.limit)
        
        hits = [
            ClinicalTrialHit(
                nct_id=r['nct_id'],
                trial_title=r['trial_title'],
                trial_status=r['trial_status'],
                phase=r['phase'],
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                molecule_form=r['molecule_form'],
                match_type=r['match_type'],
                confidence=r['confidence']
            )
            for r in rows
        ]
        
        if not hits:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            diagnostics.suggestions.insert(0, "Drug found but has no registered clinical trials")
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"trials for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=["Drug found in database but no clinical trial associations"],
                diagnostics=diagnostics
            )
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"trials for '{input.query}'",
            total_hits=len(hits),
            hits=hits
        )

    # =========================================================================
    # HANDLER: DRUG_FORMS
    # =========================================================================
    
    async def _search_drug_forms(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find all forms of a drug with enhanced diagnostics."""
        conn = await self._get_conn()
        
        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        
        if not concept_ids:
            warnings, diagnostics = self._build_not_found_diagnostics(
                input.mode, input, search_steps
            )
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"forms of '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        sql = """
            SELECT mol_id, pref_name, inchi_key, canonical_smiles,
                   is_salt, salt_form, stereo_type, chembl_id
            FROM dm_molecule
            WHERE concept_id = ANY($1::bigint[])
            ORDER BY is_salt, stereo_type
            LIMIT $2
        """
        
        rows = await conn.execute_query(sql, concept_ids, input.limit)
        
        hits = [
            MoleculeForm(
                mol_id=r['mol_id'],
                form_name=r['pref_name'],
                inchi_key=r['inchi_key'],
                canonical_smiles=r['canonical_smiles'],
                is_salt=r['is_salt'],
                salt_form=r['salt_form'],
                stereo_type=r['stereo_type'],
                chembl_id=r['chembl_id']
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
    # HANDLER: SIMILAR_MOLECULES
    # =========================================================================
    
    async def _search_similar_molecules(self, input: TargetSearchInput) -> TargetSearchOutput:
        """
        Find molecules similar to a query structure with flexible enrichment.
        
        Enrichment controlled by flags:
        - include_activities: Target binding data (default: True)
        - include_mechanisms: Curated MOA data (default: False)
        - include_trial_summary: Trial counts per molecule (default: False)
        - include_indication_summary: Indications per molecule (default: False)
        - include_aggregated_summary: Cross-molecule summaries (default: False)
        """
        import time
        start_time = time.time()
        
        conn = await self._get_conn()
        
        # === Step 1: Validate SMILES ===
        validation = await self._validate_and_canonicalize_smiles(conn, input.smiles)
        
        if not validation.is_valid:
            smiles_preview = input.smiles[:50] + "..." if len(input.smiles) > 50 else input.smiles
            return TargetSearchOutput(
                status="invalid_structure",
                mode=input.mode,
                query_summary=f"Invalid SMILES: '{smiles_preview}'",
                error=validation.error_message,
                warnings=validation.warnings,
                structure_info={
                    "original_smiles": input.smiles,
                    "error_type": validation.error_type,
                },
                diagnostics=SearchDiagnostics(
                    suggestions=[
                        "Verify SMILES syntax is correct",
                        "Check element symbols: Cl (not CL), Br (not BR)",
                        "Ensure ring closures are paired (e.g., C1CC1)",
                        "Use a molecule editor to generate valid SMILES",
                    ]
                )
            )
        
        canonical_smiles = validation.canonical_smiles
        
        # === Step 2: Find similar molecules ===
        similar_sql = """
            WITH query_fp AS (
                SELECT morganbv_fp(mol_from_smiles($1::cstring)) AS fp
            ),
            similar_mols AS (
                SELECT 
                    dm.mol_id,
                    dm.concept_id,
                    dm.pref_name AS molecule_name,
                    dm.canonical_smiles,
                    dm.chembl_id,
                    tanimoto_sml(q.fp, dm.mfp2)::float AS similarity
                FROM dm_molecule dm
                CROSS JOIN query_fp q
                WHERE dm.mfp2 % q.fp
                  AND tanimoto_sml(q.fp, dm.mfp2) >= $2
                ORDER BY similarity DESC
                LIMIT $3
            )
            SELECT DISTINCT ON (sm.concept_id)
                sm.concept_id,
                COALESCE(mc.preferred_name, sm.molecule_name) AS concept_name,
                sm.chembl_id,
                sm.canonical_smiles,
                sm.similarity AS tanimoto_similarity,
                mc.is_biotherapeutic
            FROM similar_mols sm
            LEFT JOIN dm_molecule_concept mc ON sm.concept_id = mc.concept_id
            ORDER BY sm.concept_id, sm.similarity DESC
        """
        
        mol_rows = await conn.execute_query(
            similar_sql,
            canonical_smiles,
            input.similarity_threshold,
            input.limit * 3  # Fetch extra to dedupe by concept
        )
        
        smiles_display = input.smiles[:40] + "..." if len(input.smiles) > 40 else input.smiles
        
        if not mol_rows:
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"No similar molecules for '{smiles_display}'",
                warnings=validation.warnings + [
                    f"No molecules found at similarity >= {input.similarity_threshold}"
                ],
                structure_info={
                    "original_smiles": input.smiles,
                    "canonical_smiles": canonical_smiles,
                    "preprocessing_notes": validation.warnings,
                },
                diagnostics=SearchDiagnostics(
                    suggestions=[
                        f"Try lowering similarity_threshold (current: {input.similarity_threshold})",
                        "Try 0.5 or 0.6 for broader matches",
                        "The structure may be novel with no similar compounds in database",
                    ]
                )
            )
        
        # Limit to requested count
        mol_rows = mol_rows[:input.limit]
        concept_ids = [r['concept_id'] for r in mol_rows if r.get('concept_id')]
        similarity_lookup = {r['concept_id']: r['tanimoto_similarity'] for r in mol_rows}
        
        # === Step 3: Fetch enrichment data based on flags ===
        
        activities_by_concept: dict[int, list[TargetActivity]] = {}
        mechanisms_by_concept: dict[int, list[TargetMechanism]] = {}
        trials_by_concept: dict[int, dict] = {}
        indications_by_concept: dict[int, list[dict]] = {}
        
        # 3a: Activities (default ON)
        if input.include_activities and concept_ids:
            activity_sql = """
                SELECT 
                    concept_id,
                    gene_symbol, 
                    target_name, 
                    target_organism,
                    COALESCE(best_ic50_nm, best_ki_nm, best_kd_nm, best_ec50_nm) AS activity_value_nm,
                    CASE 
                        WHEN best_ic50_nm IS NOT NULL THEN 'IC50'
                        WHEN best_ki_nm IS NOT NULL THEN 'Ki'
                        WHEN best_kd_nm IS NOT NULL THEN 'Kd'
                        ELSE 'EC50'
                    END AS activity_type,
                    best_pchembl AS pchembl,
                    n_total_measurements,
                    data_confidence,
                    sources
                FROM dm_molecule_target_summary
                WHERE concept_id = ANY($1::bigint[]) 
                  AND best_pchembl >= $2
                ORDER BY concept_id, best_pchembl DESC NULLS LAST
            """
            activity_rows = await conn.execute_query(activity_sql, concept_ids, input.min_pchembl)
            
            for r in activity_rows:
                cid = r['concept_id']
                if cid not in activities_by_concept:
                    activities_by_concept[cid] = []
                if len(activities_by_concept[cid]) < 15:  # Limit per molecule
                    activities_by_concept[cid].append(TargetActivity(
                        gene_symbol=r['gene_symbol'],
                        target_name=r['target_name'],
                        target_organism=r['target_organism'] or "Homo sapiens",
                        activity_type=r['activity_type'] or "Unknown",
                        activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else 0,
                        pchembl=float(r['pchembl']) if r['pchembl'] else None,
                        n_measurements=r['n_total_measurements'] or 1,
                        data_confidence=r['data_confidence'],
                        sources=r['sources'] or []
                    ))
        
        # 3b: Mechanisms
        if input.include_mechanisms and concept_ids:
            mechanism_sql = """
                SELECT 
                    concept_id,
                    mechanism_of_action, 
                    action_type, 
                    target_name, 
                    target_type,
                    target_organism, 
                    gene_symbols, 
                    uniprot_accessions,
                    direct_interaction, 
                    molecular_mechanism, 
                    disease_efficacy,
                    ref_type, 
                    ref_id, 
                    ref_url
                FROM dm_drug_mechanism
                WHERE concept_id = ANY($1::bigint[])
                ORDER BY concept_id, action_type
            """
            mechanism_rows = await conn.execute_query(mechanism_sql, concept_ids)
            
            for r in mechanism_rows:
                cid = r['concept_id']
                if cid not in mechanisms_by_concept:
                    mechanisms_by_concept[cid] = []
                if len(mechanisms_by_concept[cid]) < 5:
                    mechanisms_by_concept[cid].append(TargetMechanism(
                        mechanism_of_action=r['mechanism_of_action'] or "Unknown",
                        action_type=r['action_type'] or "Unknown",
                        target_name=r['target_name'],
                        target_type=r['target_type'],
                        target_organism=r['target_organism'] or "Homo sapiens",
                        gene_symbols=r['gene_symbols'] or [],
                        uniprot_accessions=r['uniprot_accessions'] or [],
                        direct_interaction=r['direct_interaction'] or False,
                        molecular_mechanism=r['molecular_mechanism'] or False,
                        disease_efficacy=r['disease_efficacy'] or False,
                        ref_type=r['ref_type'],
                        ref_id=r['ref_id'],
                        ref_url=r['ref_url']
                    ))
        
        # 3c: Trial summary
        if input.include_trial_summary and concept_ids:
            trial_sql = """
                SELECT 
                    map.concept_id,
                    COUNT(DISTINCT map.nct_id)::int AS n_trials,
                    array_agg(DISTINCT rs.phase) FILTER (WHERE rs.phase IS NOT NULL) AS phases,
                    array_agg(DISTINCT rs.overall_status) FILTER (WHERE rs.overall_status IS NOT NULL) AS statuses
                FROM map_ctgov_molecules map
                JOIN rag_study_search rs ON map.nct_id = rs.nct_id
                WHERE map.concept_id = ANY($1::bigint[])
                GROUP BY map.concept_id
            """
            trial_rows = await conn.execute_query(trial_sql, concept_ids)
            trials_by_concept = {r['concept_id']: r for r in trial_rows}
        
        # 3d: Indication summary
        if input.include_indication_summary and concept_ids:
            indication_sql = """
                SELECT 
                    d.concept_id,
                    i.preferred_name AS indication,
                    i.therapeutic_area,
                    d.max_phase,
                    d.is_approved
                FROM dm_drug_indication d
                JOIN dm_indication i ON d.indication_id = i.indication_id
                WHERE d.concept_id = ANY($1::bigint[])
                ORDER BY d.concept_id, d.max_phase DESC NULLS LAST
            """
            indication_rows = await conn.execute_query(indication_sql, concept_ids)
            
            for r in indication_rows:
                cid = r['concept_id']
                if cid not in indications_by_concept:
                    indications_by_concept[cid] = []
                indications_by_concept[cid].append({
                    'indication': r['indication'],
                    'therapeutic_area': r['therapeutic_area'],
                    'max_phase': r['max_phase'],
                    'is_approved': r['is_approved'],
                })
        
        # === Step 4: Build profiles ===
        profiles: list[CompoundTargetProfile] = []
        
        # For aggregated summary
        target_counts: dict[str, int] = {}
        mechanism_counts: dict[str, int] = {}
        indication_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        n_in_trials = 0
        n_approved = 0
        most_potent_name: str | None = None
        most_potent_pchembl: float | None = None
        
        for mol_row in mol_rows:
            concept_id = mol_row['concept_id']
            activities = activities_by_concept.get(concept_id, [])
            mechanisms = mechanisms_by_concept.get(concept_id, [])
            trial_info = trials_by_concept.get(concept_id, {})
            indications = indications_by_concept.get(concept_id, [])
            
            # Aggregate counts (for summary)
            if input.include_aggregated_summary:
                for act in activities:
                    target_counts[act.gene_symbol] = target_counts.get(act.gene_symbol, 0) + 1
                
                for mech in mechanisms:
                    mechanism_counts[mech.action_type] = mechanism_counts.get(mech.action_type, 0) + 1
                
                for ind in indications:
                    indication_counts[ind['indication']] = indication_counts.get(ind['indication'], 0) + 1
                    if ind.get('is_approved'):
                        n_approved += 1
                
                if trial_info.get('phases'):
                    for phase in trial_info['phases']:
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    n_in_trials += 1
            
            best_pchembl = max((a.pchembl for a in activities if a.pchembl), default=None)
            
            # Track most potent
            if best_pchembl and (most_potent_pchembl is None or best_pchembl > most_potent_pchembl):
                most_potent_pchembl = best_pchembl
                most_potent_name = mol_row['concept_name']
            
            profiles.append(CompoundTargetProfile(
                concept_id=concept_id,
                concept_name=mol_row['concept_name'],
                chembl_id=mol_row['chembl_id'],
                canonical_smiles=mol_row['canonical_smiles'],
                is_biotherapeutic=mol_row.get('is_biotherapeutic') or False,
                tanimoto_similarity=mol_row['tanimoto_similarity'],
                activities=activities,
                mechanisms=mechanisms,
                n_activity_targets=len(activities),
                n_mechanism_targets=len(mechanisms),
                best_pchembl=best_pchembl
            ))
        
        # Sort by similarity
        profiles.sort(key=lambda p: -(p.tanimoto_similarity or 0))
        
        # === Step 5: Build aggregated summary if requested ===
        aggregated_summary: AggregatedStructureSummary | None = None
        
        if input.include_aggregated_summary:
            sorted_targets = sorted(target_counts.items(), key=lambda x: -x[1])
            sorted_indications = sorted(indication_counts.items(), key=lambda x: -x[1])
            
            aggregated_summary = AggregatedStructureSummary(
                target_distribution=dict(sorted_targets[:20]),
                top_targets=[t[0] for t in sorted_targets[:10]],
                mechanism_distribution=dict(sorted(mechanism_counts.items(), key=lambda x: -x[1])[:10]),
                indication_distribution=dict(sorted_indications[:20]),
                top_indications=[i[0] for i in sorted_indications[:10]],
                phase_distribution=dict(sorted(phase_counts.items())),
                n_molecules_in_trials=n_in_trials,
                n_molecules_approved=n_approved,
                n_molecules_with_activity=sum(1 for p in profiles if p.activities),
                n_molecules_with_mechanisms=sum(1 for p in profiles if p.mechanisms),
                n_unique_targets=len(target_counts),
                n_unique_indications=len(indication_counts),
                most_potent_molecule=most_potent_name,
                most_potent_pchembl=most_potent_pchembl,
            )
        
        # === Step 6: Build warnings and response ===
        warnings = validation.warnings.copy() if validation.warnings else []
        
        if input.include_activities and not any(p.activities for p in profiles):
            warnings.append("No quantitative activity data found for similar molecules")
        
        if input.include_mechanisms and not any(p.mechanisms for p in profiles):
            warnings.append("No curated mechanism data found for similar molecules")
        
        if input.include_trial_summary and not trials_by_concept:
            warnings.append("No clinical trial associations found for similar molecules")
        
        # Build data source breakdown
        data_breakdown = {
            "molecules_found": len(profiles),
        }
        if input.include_activities:
            data_breakdown["molecules_with_activity"] = sum(1 for p in profiles if p.activities)
        if input.include_mechanisms:
            data_breakdown["molecules_with_mechanisms"] = sum(1 for p in profiles if p.mechanisms)
        if input.include_trial_summary:
            data_breakdown["molecules_in_trials"] = len(trials_by_concept)
        if input.include_indication_summary:
            data_breakdown["molecules_with_indications"] = len(indications_by_concept)
        
        # Build suggestions
        suggestions = []
        if aggregated_summary:
            if aggregated_summary.top_targets:
                suggestions.append(f"Most common target: {aggregated_summary.top_targets[0]}")
            if aggregated_summary.top_indications:
                suggestions.append(f"Most common indication: {aggregated_summary.top_indications[0]}")
            if aggregated_summary.most_potent_molecule:
                suggestions.append(
                    f"Most potent: {aggregated_summary.most_potent_molecule} "
                    f"(pChEMBL={aggregated_summary.most_potent_pchembl:.1f})"
                )
        
        execution_time = (time.time() - start_time) * 1000
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"similar to '{smiles_display}' ({len(profiles)} molecules)",
            total_hits=len(profiles),
            hits=profiles,
            warnings=warnings,
            execution_time_ms=execution_time,
            structure_info={
                "original_smiles": input.smiles,
                "canonical_smiles": canonical_smiles,
                "preprocessing_notes": validation.warnings,
            },
            aggregated_summary=aggregated_summary,
            data_source_breakdown=data_breakdown,
            confidence_explanation=(
                f"Similarity >= {input.similarity_threshold}, "
                f"pChEMBL >= {input.min_pchembl}"
            ),
            diagnostics=SearchDiagnostics(
                query_info={
                    "canonical_smiles": canonical_smiles,
                    "similarity_threshold": input.similarity_threshold,
                    "min_pchembl": input.min_pchembl,
                    "enrichment_flags": {
                        "activities": input.include_activities,
                        "mechanisms": input.include_mechanisms,
                        "trials": input.include_trial_summary,
                        "indications": input.include_indication_summary,
                        "aggregated": input.include_aggregated_summary,
                    }
                },
                suggestions=suggestions
            )
        )

    # =========================================================================
    # HANDLER: EXACT_STRUCTURE
    # =========================================================================
    
    async def _search_exact_structure(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find exact structure match using similarity search."""
        input_copy = input.model_copy()
        input_copy.limit = 1
        
        result = await self._search_similar_molecules(input_copy)
        
        if result.hits:
            hit = result.hits[0]
            if hit.tanimoto_similarity and hit.tanimoto_similarity >= 0.98:
                smiles_display = input.smiles[:30] + "..." if len(input.smiles) > 30 else input.smiles
                match_type = "exact" if hit.tanimoto_similarity >= 0.9999 else f"near-exact ({hit.tanimoto_similarity:.3f})"
                result.mode = input.mode
                result.query_summary = f"{match_type} match for '{smiles_display}' → {hit.concept_name}"
                return result
        
        smiles_display = input.smiles[:50] + "..." if len(input.smiles) > 50 else input.smiles
        warnings, diagnostics = self._build_not_found_diagnostics(input.mode, input)
        
        return TargetSearchOutput(
            status="not_found",
            mode=input.mode,
            query_summary=f"exact structure '{smiles_display}'",
            total_hits=0,
            hits=[],
            warnings=warnings,
            diagnostics=diagnostics
        )

    # =========================================================================
    # HANDLER: SUBSTRUCTURE
    # =========================================================================
    
    async def _search_substructure(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find molecules containing a substructure."""
        conn = await self._get_conn()
        
        pattern = input.smarts or input.smiles
        
        sql = """
            SELECT DISTINCT ON (dm.concept_id)
                dm.concept_id,
                COALESCE(mc.preferred_name, dm.pref_name) as concept_name,
                dm.chembl_id,
                dm.canonical_smiles
            FROM dm_molecule dm
            LEFT JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
            WHERE dm.mol @> qmol_from_smarts($1::cstring)
            ORDER BY dm.concept_id
            LIMIT $2
        """
        
        rows = await conn.execute_query(sql, pattern, input.limit)
        
        profiles = [
            CompoundTargetProfile(
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                chembl_id=r['chembl_id'],
                canonical_smiles=r['canonical_smiles'],
                activities=[],
                mechanisms=[]
            )
            for r in rows
        ]
        
        pattern_display = pattern[:30] + "..." if len(pattern) > 30 else pattern
        
        if not profiles:
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"substructure '{pattern_display}'",
                total_hits=0,
                hits=[],
                warnings=[f"No molecules found containing substructure: {pattern_display}"],
                diagnostics=SearchDiagnostics(
                    suggestions=[
                        "Verify SMARTS/SMILES pattern is valid",
                        "Try a simpler or more common substructure",
                        "Very specific patterns may have no matches"
                    ]
                )
            )
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"substructure '{pattern_display}'",
            total_hits=len(profiles),
            hits=profiles
        )

    # =========================================================================
    # HANDLER: COMPARE_DRUGS
    # =========================================================================
    
    async def _search_compare_drugs(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Compare drugs on a target with enhanced diagnostics."""
        conn = await self._get_conn()
        
        target = input.effective_target.upper()
        drug_patterns = [f"%{name}%" for name in input.drug_names]
        
        sql = """
            WITH drug_data AS (
                SELECT 
                    concept_id, concept_name,
                    representative_smiles as canonical_smiles,
                    COALESCE(best_ic50_nm, best_ki_nm) as activity_value_nm,
                    CASE WHEN best_ic50_nm IS NOT NULL THEN 'IC50' ELSE 'Ki' END as activity_type,
                    best_pchembl as pchembl,
                    n_total_measurements,
                    data_confidence,
                    sources
                FROM dm_molecule_target_summary
                WHERE gene_symbol = $1
                  AND concept_name ILIKE ANY($2::text[])
            ),
            with_rank AS (
                SELECT *,
                       activity_value_nm / NULLIF(MIN(activity_value_nm) OVER (), 0) as fold_vs_best
                FROM drug_data
            )
            SELECT * FROM with_rank
            ORDER BY activity_value_nm ASC NULLS LAST
        """
        
        rows = await conn.execute_query(sql, target, drug_patterns)
        
        hits = [
            DrugForTargetHit(
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                canonical_smiles=r['canonical_smiles'],
                activity_type=r['activity_type'],
                activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else None,
                pchembl=float(r['pchembl']) if r['pchembl'] else None,
                n_measurements=r['n_total_measurements'] or 0,
                data_confidence=r['data_confidence'],
                sources=r['sources'] or [],
                selectivity_fold=float(r['fold_vs_best']) if r['fold_vs_best'] else None
            )
            for r in rows
        ]
        
        if not hits:
            warnings, diagnostics = self._build_not_found_diagnostics(input.mode, input)
            
            # Check which drugs were not found
            found_names = {h.concept_name.lower() for h in hits}
            missing = [d for d in input.drug_names if not any(d.lower() in fn for fn in found_names)]
            if missing:
                warnings.append(f"Drugs not found or lacking data on {target}: {', '.join(missing)}")
            
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"compare {', '.join(input.drug_names)} on {target}",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        # Warn about missing drugs
        found_names = {h.concept_name.lower() for h in hits}
        missing = [d for d in input.drug_names if not any(d.lower() in fn for fn in found_names)]
        warnings = []
        if missing:
            warnings.append(f"Some drugs not found or lacking activity on {target}: {', '.join(missing)}")
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"compare {', '.join(input.drug_names)} on {target}",
            total_hits=len(hits),
            hits=hits,
            warnings=warnings
        )

    # =========================================================================
    # HANDLER: SELECTIVE_DRUGS
    # =========================================================================
    
    async def _search_selective_drugs(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find selective drugs for a target vs off-targets with enhanced diagnostics."""
        conn = await self._get_conn()
        
        target = input.effective_target.upper()
        off_targets = [g.upper() for g in input.off_targets]
        
        # Optimized query
        sql = """
            WITH on_target AS (
                SELECT concept_id, concept_name, best_pchembl,
                       COALESCE(best_ic50_nm, best_ki_nm) as on_target_nm,
                       representative_smiles as canonical_smiles,
                       data_confidence, sources
                FROM dm_molecule_target_summary
                WHERE gene_symbol = $1
                  AND best_pchembl >= $2
                ORDER BY best_pchembl DESC
                LIMIT 200
            ),
            off_target_agg AS (
                SELECT concept_id, 
                       MIN(COALESCE(best_ic50_nm, best_ki_nm)) as best_off_target_nm
                FROM dm_molecule_target_summary
                WHERE gene_symbol = ANY($3::text[])
                  AND concept_id IN (SELECT concept_id FROM on_target)
                GROUP BY concept_id
            )
            SELECT 
                ot.concept_id, ot.concept_name, ot.canonical_smiles,
                ot.best_pchembl as pchembl,
                ot.on_target_nm as activity_value_nm,
                ROUND((oft.best_off_target_nm / NULLIF(ot.on_target_nm, 0))::numeric, 1) as selectivity_fold,
                ot.data_confidence, ot.sources
            FROM on_target ot
            LEFT JOIN off_target_agg oft ON ot.concept_id = oft.concept_id
            WHERE oft.best_off_target_nm IS NULL 
               OR (oft.best_off_target_nm / NULLIF(ot.on_target_nm, 0)) >= $4
            ORDER BY 
                CASE WHEN oft.best_off_target_nm IS NULL THEN 1 ELSE 0 END,
                selectivity_fold DESC NULLS LAST,
                ot.best_pchembl DESC
            LIMIT $5
        """
        
        rows = await conn.execute_query(sql, target, input.min_pchembl, off_targets,
                                        input.min_selectivity_fold, input.limit)
        
        hits = [
            DrugForTargetHit(
                concept_id=r['concept_id'],
                concept_name=r['concept_name'],
                canonical_smiles=r['canonical_smiles'],
                activity_type="IC50/Ki",
                activity_value_nm=float(r['activity_value_nm']) if r['activity_value_nm'] else None,
                pchembl=float(r['pchembl']) if r['pchembl'] else None,
                data_confidence=r['data_confidence'],
                sources=r['sources'] or [],
                selectivity_fold=float(r['selectivity_fold']) if r['selectivity_fold'] else None
            )
            for r in rows
        ]
        
        if not hits:
            warnings, diagnostics = self._build_not_found_diagnostics(input.mode, input)
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"selective for {target} vs {', '.join(off_targets)}",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )
        
        # Add info about selectivity
        n_with_selectivity = sum(1 for h in hits if h.selectivity_fold is not None)
        warnings = []
        if n_with_selectivity < len(hits):
            warnings.append(f"{len(hits) - n_with_selectivity} compounds have no measured off-target activity (may be highly selective or untested)")
        
        return TargetSearchOutput(
            status="success",
            mode=input.mode,
            query_summary=f"selective for {target} vs {', '.join(off_targets)}",
            total_hits=len(hits),
            hits=hits,
            warnings=warnings
        )

    # =========================================================================
    # DELEGATING HANDLERS
    # =========================================================================
    
    async def _search_indications_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find indications linked to a drug."""
        conn = await self._get_conn()

        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        if not concept_ids:
            warnings, diagnostics = self._build_not_found_diagnostics(input.mode, input, search_steps)
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"indications for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )

        approval_filter = ""
        if input.approval_status == "approved":
            approval_filter = "AND d.is_approved = TRUE"
        elif input.approval_status == "investigational":
            approval_filter = "AND (d.is_approved IS NOT TRUE)"

        sql = f"""
            SELECT
                i.indication_id,
                i.preferred_name,
                i.therapeutic_area,
                d.max_phase,
                d.is_approved,
                d.sources,
                d.ref_type,
                d.ref_id,
                d.ref_url
            FROM dm_drug_indication d
            JOIN dm_indication i ON d.indication_id = i.indication_id
            WHERE d.concept_id = ANY($1::bigint[])
            {approval_filter}
            ORDER BY d.max_phase DESC NULLS LAST, i.preferred_name
            LIMIT $2
        """

        rows = await conn.execute_query(sql, concept_ids, input.limit)
        hits = [
            IndicationHit(
                indication_id=r["indication_id"],
                preferred_name=r["preferred_name"],
                therapeutic_area=r["therapeutic_area"],
                max_phase=float(r["max_phase"]) if r["max_phase"] is not None else None,
                is_approved=r["is_approved"],
                sources=r["sources"] or [],
                ref_type=r["ref_type"],
                ref_id=r["ref_id"],
                ref_url=r["ref_url"],
            )
            for r in rows
        ]

        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"indications for '{input.query}'",
            total_hits=len(hits),
            hits=hits,
            warnings=[] if hits else [f"No indications found for '{input.query}'"]
        )

    async def _search_drugs_for_indication(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find drugs linked to an indication."""
        conn = await self._get_conn()
        query_exact = input.query.strip()
        query_pattern = f"%{query_exact}%"

        approval_filter = ""
        if input.approval_status == "approved":
            approval_filter = "AND d.is_approved = TRUE"
        elif input.approval_status == "investigational":
            approval_filter = "AND (d.is_approved IS NOT TRUE)"

        sql = f"""
            WITH matched_indications AS (
                SELECT indication_id
                FROM dm_indication
                WHERE preferred_name_lower = LOWER($1)
                   OR preferred_name ILIKE $2
                   OR EXISTS (
                       SELECT 1 FROM unnest(COALESCE(synonyms, ARRAY[]::text[])) AS s
                       WHERE LOWER(s) = LOWER($1)
                   )
                LIMIT 20
            )
            SELECT
                mc.concept_id,
                mc.preferred_name AS concept_name,
                d.max_phase,
                d.is_approved,
                d.sources
            FROM dm_drug_indication d
            JOIN matched_indications mi ON d.indication_id = mi.indication_id
            JOIN dm_molecule_concept mc ON d.concept_id = mc.concept_id
            WHERE mc.preferred_name IS NOT NULL
            {approval_filter}
            ORDER BY d.max_phase DESC NULLS LAST, mc.preferred_name
            LIMIT $3
        """

        rows = await conn.execute_query(sql, query_exact, query_pattern, input.limit)
        hits = [
            DrugIndicationHit(
                concept_id=r["concept_id"],
                concept_name=r["concept_name"],
                max_phase=float(r["max_phase"]) if r["max_phase"] is not None else None,
                is_approved=r["is_approved"],
                sources=r["sources"] or [],
            )
            for r in rows
        ]

        status = "success" if hits else "not_found"
        warnings = [] if hits else [f"No drugs found for indication '{input.query}'"]
        return TargetSearchOutput(
            status=status,
            mode=input.mode,
            query_summary=f"drugs for indication '{input.query}'",
            total_hits=len(hits),
            hits=hits,
            warnings=warnings
        )

    async def _search_target_pathways(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Return protein class summaries for a target gene."""
        conn = await self._get_conn()
        gene_symbol = input.query.upper()

        sql = """
            SELECT
                t.gene_symbol,
                t.protein_class_id,
                pc.protein_class_desc,
                array_remove(array_agg(DISTINCT pcs.protein_class_synonym), NULL) AS synonyms
            FROM dm_target t
            LEFT JOIN protein_classification pc ON t.protein_class_id = pc.protein_class_id
            LEFT JOIN protein_class_synonyms pcs ON t.protein_class_id = pcs.protein_class_id
            WHERE t.gene_symbol = $1
            GROUP BY t.gene_symbol, t.protein_class_id, pc.protein_class_desc
        """

        rows = await conn.execute_query(sql, gene_symbol)
        hits = [
            TargetPathwayHit(
                gene_symbol=r["gene_symbol"],
                protein_class_id=r["protein_class_id"],
                protein_class_desc=r["protein_class_desc"],
                synonyms=r["synonyms"] or [],
            )
            for r in rows
        ]

        warnings = []
        if hits and not hits[0].protein_class_desc:
            warnings.append("Target found but protein class metadata is missing")
        if not hits:
            warnings.append(f"No target metadata found for '{gene_symbol}'")

        return TargetSearchOutput(
            status="success" if hits else "not_found",
            mode=input.mode,
            query_summary=f"target pathways for '{gene_symbol}'",
            total_hits=len(hits),
            hits=hits,
            warnings=warnings
        )

    async def _search_drug_interactions(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Find metabolism-based interactions for a drug."""
        conn = await self._get_conn()
        concept_ids, search_steps = await self._find_concept_ids_with_diagnostics(conn, input.query)
        if not concept_ids:
            warnings, diagnostics = self._build_not_found_diagnostics(input.mode, input, search_steps)
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"drug interactions for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings,
                diagnostics=diagnostics
            )

        chembl_sql = """
            SELECT DISTINCT dm.chembl_id, dm.concept_id, mc.preferred_name
            FROM dm_molecule dm
            JOIN dm_molecule_concept mc ON dm.concept_id = mc.concept_id
            WHERE dm.concept_id = ANY($1::bigint[])
              AND dm.chembl_id IS NOT NULL
            LIMIT 20
        """
        chembl_rows = await conn.execute_query(chembl_sql, concept_ids)
        chembl_ids = [r["chembl_id"] for r in chembl_rows if r.get("chembl_id")]
        concept_name_lookup = {r["chembl_id"]: r["preferred_name"] for r in chembl_rows if r.get("chembl_id")}
        if not chembl_ids:
            warnings = [f"No ChEMBL IDs found for '{input.query}' to resolve metabolism data"]
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"drug interactions for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings
            )

        molregno_rows = await conn.execute_query(
            "SELECT molregno, chembl_id FROM molecule_dictionary WHERE chembl_id = ANY($1::text[])",
            chembl_ids
        )
        molregnos = [r["molregno"] for r in molregno_rows]
        molregno_to_chembl = {r["molregno"]: r["chembl_id"] for r in molregno_rows}

        if not molregnos:
            warnings = [f"No ChEMBL molregno records found for '{input.query}'"]
            return TargetSearchOutput(
                status="not_found",
                mode=input.mode,
                query_summary=f"drug interactions for '{input.query}'",
                total_hits=0,
                hits=[],
                warnings=warnings
            )

        sql = """
            SELECT
                m.drug_record_id,
                m.substrate_record_id,
                m.metabolite_record_id,
                m.pathway_key,
                m.enzyme_name,
                md_drug.chembl_id AS drug_chembl_id,
                md_sub.chembl_id AS substrate_chembl_id,
                md_met.chembl_id AS metabolite_chembl_id
            FROM metabolism m
            LEFT JOIN molecule_dictionary md_drug ON m.drug_record_id = md_drug.molregno
            LEFT JOIN molecule_dictionary md_sub ON m.substrate_record_id = md_sub.molregno
            LEFT JOIN molecule_dictionary md_met ON m.metabolite_record_id = md_met.molregno
            WHERE m.drug_record_id = ANY($1::bigint[])
               OR m.substrate_record_id = ANY($1::bigint[])
               OR m.metabolite_record_id = ANY($1::bigint[])
            ORDER BY m.met_id DESC
            LIMIT $2
        """

        rows = await conn.execute_query(sql, molregnos, input.limit)
        hits = []
        molregno_set = set(molregnos)
        for r in rows:
            interaction_role = None
            if r["drug_record_id"] in molregno_set:
                interaction_role = "drug"
            elif r["substrate_record_id"] in molregno_set:
                interaction_role = "substrate"
            elif r["metabolite_record_id"] in molregno_set:
                interaction_role = "metabolite"

            chembl_id = molregno_to_chembl.get(
                r["drug_record_id"] or r["substrate_record_id"] or r["metabolite_record_id"]
            )
            hits.append(
                DrugInteractionHit(
                    concept_id=concept_ids[0] if concept_ids else None,
                    concept_name=concept_name_lookup.get(chembl_id),
                    chembl_id=chembl_id,
                    interaction_role=interaction_role,
                    enzyme_name=r["enzyme_name"],
                    pathway_key=r["pathway_key"],
                    metabolite_chembl_id=r["metabolite_chembl_id"],
                    substrate_chembl_id=r["substrate_chembl_id"],
                )
            )

        status = "success" if hits else "not_found"
        warnings = [] if hits else [f"No metabolism interactions found for '{input.query}'"]
        return TargetSearchOutput(
            status=status,
            mode=input.mode,
            query_summary=f"drug interactions for '{input.query}'",
            total_hits=len(hits),
            hits=hits,
            warnings=warnings
        )

    async def _search_activities_for_drug(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get raw activity measurements for a drug."""
        input_copy = input.model_copy()
        input_copy.mode = SearchMode.TARGETS_FOR_DRUG
        input_copy.data_source = DataSource.ACTIVITY
        result = await self._search_targets_for_drug(input_copy)
        result.mode = input.mode
        return result

    async def _search_activities_for_target(self, input: TargetSearchInput) -> TargetSearchOutput:
        """Get raw activity measurements for a target."""
        input_copy = input.model_copy()
        input_copy.mode = SearchMode.DRUGS_FOR_TARGET
        input_copy.data_source = DataSource.ACTIVITY
        result = await self._search_drugs_for_target(input_copy)
        result.mode = input.mode
        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def target_search_async(
    db_config: DatabaseConfig,
    input: TargetSearchInput
) -> TargetSearchOutput:
    """
    Execute a pharmacology search using a TargetSearchInput.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        input: TargetSearchInput specifying mode, query, and filters.

    Returns:
        TargetSearchOutput with status, hits, and diagnostics.
    """
    searcher = PharmacologySearch(db_config)
    return await searcher.search(input)