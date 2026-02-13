#!/usr/bin/env python3
"""
Molecule <-> Clinical Trials Connectivity Search.

Links small molecules to clinical trials using mapping tables and
concept identifiers. Supports molecule-centric, condition-centric,
target-centric, and structure-centric queries.

Modes:
    - trials_by_molecule: find trials linked to a molecule name or InChIKey
    - molecules_by_condition: find molecules associated with a condition
    - trials_by_target: find trials linked to a target gene
    - trials_by_structure: find trials for molecules similar to a SMILES
    - trials_by_substructure: find trials for molecules containing a substructure

Data Sources:
    - map_ctgov_molecules
    - dm_molecule / dm_molecule_concept / dm_molecule_synonyms
    - rag_study_search
    - compound_structures (for RDKit searches)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig
from bioagent.data.semantic_utils import encode_query_vector, normalize_semantic_text
from bioagent.data.search.clinical_trial_search import render_study_text_full
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree


def _sanitize_tsquery(text: str) -> str:
    text = text or ""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " & ".join(cleaned.split())


# =============================================================================
# SMILES Preprocessing and Validation
# =============================================================================

@dataclass
class SmilesValidationResult:
    """Result of SMILES validation and preprocessing."""
    is_valid: bool
    canonical_smiles: str | None = None
    original_smiles: str | None = None
    error_message: str | None = None
    error_type: str | None = None  # "syntax", "sanitization", "empty", "invalid_chars", etc.
    warnings: list[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _preprocess_smiles_string(smiles: str) -> tuple[str, list[str]]:
    """
    Preprocess SMILES string before RDKit parsing.
    
    Returns:
        Tuple of (cleaned_smiles, warnings)
    """
    warnings = []
    original = smiles
    
    # Strip whitespace
    smiles = smiles.strip()
    
    # Remove common prefixes that users might accidentally include
    prefixes_to_remove = [
        "SMILES:", "smiles:", "SMILES=", "smiles=",
        "SMILES ", "smiles ",
    ]
    for prefix in prefixes_to_remove:
        if smiles.startswith(prefix):
            smiles = smiles[len(prefix):].strip()
            warnings.append(f"Removed prefix '{prefix.strip()}' from input")
            break
    
    # Remove surrounding quotes if present
    if (smiles.startswith('"') and smiles.endswith('"')) or \
       (smiles.startswith("'") and smiles.endswith("'")):
        smiles = smiles[1:-1]
        warnings.append("Removed surrounding quotes from SMILES")
    
    # Check for and remove newlines/tabs
    if '\n' in smiles or '\t' in smiles or '\r' in smiles:
        smiles = re.sub(r'[\n\t\r]+', '', smiles)
        warnings.append("Removed newline/tab characters from SMILES")
    
    # Check for multiple SMILES (dot-separated components are valid, but space-separated are not)
    if ' ' in smiles and '.' not in smiles:
        # Might be multiple SMILES - take the first one
        parts = smiles.split()
        if len(parts) > 1:
            smiles = parts[0]
            warnings.append(f"Input contained multiple space-separated tokens; using first: '{smiles[:30]}...'")
    
    # Check for obviously invalid characters (but be careful - SMILES has many valid chars)
    invalid_chars = set(smiles) - set("CNOSPFIBrcnospfib0123456789=#@+\\/-[]()%.H")
    if invalid_chars:
        # Some might be valid in extended SMILES, so just warn
        warnings.append(f"Potentially invalid characters detected: {invalid_chars}")
    
    return smiles, warnings


def _check_smiles_syntax(smiles: str) -> tuple[bool, str | None]:
    """
    Quick syntax checks before sending to RDKit.
    
    Returns:
        Tuple of (is_ok, error_message)
    """
    if not smiles:
        return False, "Empty SMILES string"
    
    if len(smiles) > 5000:
        return False, f"SMILES too long ({len(smiles)} chars). Maximum supported: 5000"
    
    # Check bracket balance
    if smiles.count('[') != smiles.count(']'):
        return False, f"Unbalanced square brackets: {smiles.count('[')} '[' vs {smiles.count(']')} ']'"
    
    if smiles.count('(') != smiles.count(')'):
        return False, f"Unbalanced parentheses: {smiles.count('(')} '(' vs {smiles.count(')')} ')'"
    
    # Check for empty brackets
    if '[]' in smiles:
        return False, "Empty brackets '[]' found in SMILES"
    
    # Check for double bonds/rings issues
    if '==' in smiles:
        return False, "Invalid double bond notation '==' found"
    
    if '##' in smiles:
        return False, "Invalid triple bond notation '##' found"
    
    # Check ring closure numbers are reasonable
    ring_numbers = re.findall(r'%(\d+)', smiles)
    for num in ring_numbers:
        if int(num) > 99:
            return False, f"Ring closure number %{num} is too large (max 99)"
    
    return True, None


async def _validate_and_canonicalize_smiles(
    async_config: AsyncDatabaseConfig,
    smiles: str,
) -> SmilesValidationResult:
    """
    Validate and canonicalize SMILES using PostgreSQL RDKit cartridge.
    
    This function:
    1. Preprocesses the input string (removes prefixes, quotes, etc.)
    2. Performs basic syntax checks
    3. Uses RDKit to parse and canonicalize
    4. Returns detailed error information if invalid
    
    Args:
        async_config: Database connection
        smiles: Input SMILES string
        
    Returns:
        SmilesValidationResult with validation status and details
    """
    original_smiles = smiles
    
    # Step 1: Preprocess
    cleaned_smiles, preprocess_warnings = _preprocess_smiles_string(smiles)
    
    # Step 2: Basic syntax check
    syntax_ok, syntax_error = _check_smiles_syntax(cleaned_smiles)
    if not syntax_ok:
        return SmilesValidationResult(
            is_valid=False,
            original_smiles=original_smiles,
            error_message=syntax_error,
            error_type="syntax",
            warnings=preprocess_warnings,
        )
    
    # Step 3: RDKit validation and canonicalization
    # We use multiple queries to get detailed error information
    try:
        # First, try to parse the molecule
        parse_sql = """
            SELECT 
                mol_from_smiles($1::cstring) IS NOT NULL AS is_valid
        """
        rows = await async_config.execute_query(parse_sql, cleaned_smiles)
        
        if not rows or not rows[0]["is_valid"]:
            # Molecule failed to parse - try to get more details
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original_smiles,
                error_message=_generate_detailed_smiles_error(cleaned_smiles),
                error_type="parse_error",
                warnings=preprocess_warnings,
            )
        
        # Molecule parsed successfully - get canonical form and properties
        canonical_sql = """
            SELECT 
                mol_to_smiles(mol_from_smiles($1::cstring)) AS canonical_smiles,
                mol_formula(mol_from_smiles($1::cstring)) AS formula,
                mol_amw(mol_from_smiles($1::cstring)) AS mol_weight,
                mol_numheavyatoms(mol_from_smiles($1::cstring)) AS heavy_atoms
        """
        rows = await async_config.execute_query(canonical_sql, cleaned_smiles)
        
        if rows and rows[0]["canonical_smiles"]:
            canonical = rows[0]["canonical_smiles"]
            
            # Add info warnings if structure looks unusual
            mol_weight = rows[0].get("mol_weight")
            heavy_atoms = rows[0].get("heavy_atoms")
            
            if mol_weight and mol_weight > 2000:
                preprocess_warnings.append(f"Large molecule detected (MW: {mol_weight:.1f})")
            
            if heavy_atoms and heavy_atoms < 3:
                preprocess_warnings.append(f"Very small molecule ({heavy_atoms} heavy atoms)")
            
            # Check if canonical differs significantly from input
            if canonical != cleaned_smiles:
                preprocess_warnings.append("SMILES was canonicalized (this is normal)")
            
            return SmilesValidationResult(
                is_valid=True,
                canonical_smiles=canonical,
                original_smiles=original_smiles,
                warnings=preprocess_warnings,
            )
        else:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original_smiles,
                error_message="RDKit could not canonicalize the molecule",
                error_type="canonicalization_error",
                warnings=preprocess_warnings,
            )
            
    except Exception as e:
        error_str = str(e).lower()
        
        # Parse PostgreSQL/RDKit error messages for user-friendly output
        if "mol_from_smiles" in error_str or "cstring" in error_str:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original_smiles,
                error_message=_generate_detailed_smiles_error(cleaned_smiles, str(e)),
                error_type="rdkit_error",
                warnings=preprocess_warnings,
            )
        else:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original_smiles,
                error_message=f"Database error during validation: {e}",
                error_type="database_error",
                warnings=preprocess_warnings,
            )


def _generate_detailed_smiles_error(smiles: str, rdkit_error: str | None = None) -> str:
    """
    Generate a detailed, user-friendly error message for invalid SMILES.
    
    Analyzes the SMILES to identify likely issues.
    """
    issues = []
    
    # Check for common issues
    
    # 1. Lowercase elements that should be uppercase
    lowercase_issues = []
    for match in re.finditer(r'\b([cnosp])\b(?![a-z])', smiles):
        char = match.group(1)
        # c, n, o, s are valid in aromatic context, but might be errors
        # Only flag if they appear in non-ring context
        pass  # This is actually valid aromatic notation
    
    # 2. Invalid element symbols
    invalid_elements = re.findall(r'\[([A-Za-z]{3,})\]', smiles)
    if invalid_elements:
        issues.append(f"Possibly invalid element symbols: {invalid_elements}")
    
    # 3. Stereochemistry issues
    if smiles.count('@') > 10:
        issues.append("Excessive stereochemistry annotations (@) - verify structure")
    
    # 4. Check for common typos
    typos = {
        'CL': 'Cl (chlorine)',
        'BR': 'Br (bromine)', 
        'NA': 'Na (sodium)',
        'CA': 'Ca (calcium)',
    }
    for typo, correction in typos.items():
        if typo in smiles and f'[{typo}]' not in smiles:
            issues.append(f"'{typo}' should likely be '{correction}'")
    
    # 5. Check charge notation
    if '+-' in smiles or '-+' in smiles:
        issues.append("Invalid charge notation '+-' or '-+' found")
    
    # 6. Ring closure issues
    ring_digits = re.findall(r'(?<![%\d])(\d)', smiles)
    digit_counts = {}
    for d in ring_digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1
    odd_rings = [d for d, count in digit_counts.items() if count % 2 != 0]
    if odd_rings:
        issues.append(f"Unmatched ring closure numbers: {odd_rings}")
    
    # Build error message
    base_msg = "Invalid SMILES structure"
    
    if rdkit_error:
        # Extract useful part of RDKit error
        if "SMILES Parse Error" in rdkit_error:
            base_msg = "SMILES parsing failed"
        elif "sanitize" in rdkit_error.lower():
            base_msg = "SMILES failed chemical validation (sanitization)"
        elif "valence" in rdkit_error.lower():
            base_msg = "Invalid valence detected in structure"
    
    if issues:
        return f"{base_msg}. Potential issues: {'; '.join(issues)}"
    else:
        # Generic helpful message
        return (
            f"{base_msg}. Please verify: "
            "1) Bracket balance [], () "
            "2) Element symbols are correct (Cl not CL, Br not BR) "
            "3) Ring closures are paired "
            "4) Aromatic atoms are lowercase (c, n, o, s) only in aromatic rings"
        )


async def _validate_smarts_pattern(
    async_config: AsyncDatabaseConfig,
    smarts: str,
) -> SmilesValidationResult:
    """
    Validate a SMARTS pattern using PostgreSQL RDKit cartridge.
    
    SMARTS is more permissive than SMILES, so validation is different.
    """
    original = smarts
    warnings = []
    
    # Basic preprocessing
    smarts = smarts.strip()
    
    # Remove common prefixes
    for prefix in ["SMARTS:", "smarts:", "SMARTS=", "smarts="]:
        if smarts.startswith(prefix):
            smarts = smarts[len(prefix):].strip()
            warnings.append(f"Removed prefix '{prefix.strip()}'")
            break
    
    if not smarts:
        return SmilesValidationResult(
            is_valid=False,
            original_smiles=original,
            error_message="Empty SMARTS pattern",
            error_type="empty",
            warnings=warnings,
        )
    
    # Check bracket balance
    if smarts.count('[') != smarts.count(']'):
        return SmilesValidationResult(
            is_valid=False,
            original_smiles=original,
            error_message=f"Unbalanced brackets in SMARTS: {smarts.count('[')} '[' vs {smarts.count(']')} ']'",
            error_type="syntax",
            warnings=warnings,
        )
    
    # Validate with RDKit
    try:
        sql = """
            SELECT qmol_from_smarts($1::cstring) IS NOT NULL AS is_valid
        """
        rows = await async_config.execute_query(sql, smarts)
        
        if rows and rows[0]["is_valid"]:
            return SmilesValidationResult(
                is_valid=True,
                canonical_smiles=smarts,  # SMARTS doesn't have a canonical form
                original_smiles=original,
                warnings=warnings,
            )
        else:
            return SmilesValidationResult(
                is_valid=False,
                original_smiles=original,
                error_message=_generate_detailed_smarts_error(smarts),
                error_type="parse_error",
                warnings=warnings,
            )
            
    except Exception as e:
        return SmilesValidationResult(
            is_valid=False,
            original_smiles=original,
            error_message=f"SMARTS validation failed: {e}",
            error_type="rdkit_error",
            warnings=warnings,
        )


def _generate_detailed_smarts_error(smarts: str) -> str:
    """Generate detailed error message for invalid SMARTS."""
    issues = []
    
    # Check for common SMARTS issues
    
    # 1. Invalid atom queries
    invalid_queries = re.findall(r'\[([^\]]+)\]', smarts)
    for query in invalid_queries:
        if query.startswith('!') and len(query) == 1:
            issues.append("Empty negation '!' in atom query")
        if '&' in query and query.endswith('&'):
            issues.append("Incomplete AND expression in atom query")
        if ',' in query and query.endswith(','):
            issues.append("Incomplete OR expression in atom query")
    
    # 2. Recursive SMARTS issues
    if '$(' in smarts:
        if smarts.count('$(') != smarts.count(')'):
            issues.append("Unbalanced recursive SMARTS $(...)")
    
    if issues:
        return f"Invalid SMARTS pattern. Issues: {'; '.join(issues)}"
    else:
        return (
            "Invalid SMARTS pattern. Please verify: "
            "1) Atom queries are valid [#6], [C,N], etc. "
            "2) Logical operators &, |, ! are used correctly "
            "3) Recursive SMARTS $(...) are balanced"
        )


# =============================================================================
# Input Model
# =============================================================================

class MoleculeTrialSearchInput(BaseModel):
    mode: Literal[
        "trials_by_molecule",
        "molecules_by_condition",
        "trials_by_target",
        "trials_by_structure",
        "trials_by_substructure",
        "trials_by_sequence",
    ]
    
    # Text-based inputs
    molecule_name: str | None = None
    inchi_key: str | None = None
    target_gene: str | None = None
    condition: str | None = None
    sequence: str | None = None
    
    # Structure-based inputs
    smiles: str | None = None
    smarts: str | None = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Filters
    min_pchembl: float = Field(default=6.0, ge=0.0, le=15.0)
    phase: list[str] | None = None
    status: list[str] | None = None
    molecule_type: Literal["all", "small_molecule", "biotherapeutic"] = "all"
    biotherapeutic_type: Literal["antibody", "enzyme", "all"] = "all"
    # trials_by_target: default uses dm_drug_mechanism only; set True to also include dm_compound_target_activity
    include_activity_data: bool = False

    # Trial output filters (rag_study_search columns)
    start_date_from: date | None = None
    start_date_to: date | None = None
    completion_date_from: date | None = None
    completion_date_to: date | None = None
    min_enrollment: int | None = None
    max_enrollment: int | None = None
    lead_sponsor: str | None = None
    country: list[str] | None = None
    has_results: bool | None = None
    is_fda_regulated: bool | None = None

    # Optional full trial summary (join rag_study_corpus, render study_json)
    include_study_details: bool = False
    # Optional grouping (currently applied to trials_by_molecule and trials_by_target)
    group_by: Literal["none", "condition", "molecule_concept", "intervention"] = "none"

    # Pagination
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator(
        "molecule_name",
        "inchi_key",
        "target_gene",
        "condition",
        "smiles",
        "smarts",
        "sequence",
        mode="before",
    )
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
        elif self.mode == "trials_by_structure":
            if not self.smiles:
                errors.append("'smiles' is required for structure similarity search")
        elif self.mode == "trials_by_substructure":
            if not self.smiles and not self.smarts:
                errors.append("'smiles' or 'smarts' is required for substructure search")
        elif self.mode == "trials_by_sequence":
            if not self.sequence:
                errors.append("'sequence' is required for sequence search")
        return errors


# =============================================================================
# Output Models
# =============================================================================

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
    similarity_score: float | None = None
    structure_match_type: str | None = None
    # Extended trial info (from rag_study_search)
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    lead_sponsor: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    rendered_summary: str | None = None

    def pretty_print(self, console: Console | None = None, show_summary: bool = False) -> None:
        """Pretty print a single trial hit."""
        console = console or Console()
        status_colors = {
            "RECRUITING": "green",
            "ACTIVE_NOT_RECRUITING": "yellow",
            "COMPLETED": "blue",
            "TERMINATED": "red",
            "WITHDRAWN": "red",
            "NOT_YET_RECRUITING": "cyan",
            "SUSPENDED": "orange3",
        }
        status_color = status_colors.get(self.status or "", "white")
        conf = f" (conf: {self.confidence:.2f})" if self.confidence is not None else ""
        title = f"[bold cyan]{self.nct_id}[/bold cyan][dim]{conf}[/dim]"
        tree = Tree(title)
        tree.add(f"[bold]{_truncate(self.brief_title or '', 100)}[/bold]")
        meta = tree.add("[bold yellow]Metadata[/bold yellow]")
        meta.add(f"[dim]Phase:[/dim] {self.phase or 'N/A'}")
        meta.add(f"[dim]Status:[/dim] [{status_color}]{self.status or 'N/A'}[/{status_color}]")
        if self.enrollment is not None:
            meta.add(f"[dim]Enrollment:[/dim] {self.enrollment:,}")
        if self.start_date:
            meta.add(f"[dim]Start:[/dim] {self.start_date}")
        if self.completion_date:
            meta.add(f"[dim]Completion:[/dim] {self.completion_date}")
        if self.lead_sponsor:
            meta.add(f"[dim]Sponsor:[/dim] {_truncate(self.lead_sponsor, 50)}")
        mol = tree.add("[bold blue]Molecule[/bold blue]")
        mol.add(f"[dim]Concept:[/dim] {self.concept_name or 'N/A'}")
        if self.molecule_name:
            mol.add(f"[dim]Name:[/dim] {self.molecule_name}")
        if self.match_type:
            mol.add(f"[dim]Match:[/dim] {self.match_type}")
        if self.conditions:
            cond_branch = tree.add(f"[bold green]Conditions ({len(self.conditions)})[/bold green]")
            for c in self.conditions[:5]:
                cond_branch.add(f"  {_truncate(c, 60)}")
            if len(self.conditions) > 5:
                cond_branch.add(f"[dim]... and {len(self.conditions) - 5} more[/dim]")
        if self.interventions:
            int_branch = tree.add(f"[bold blue]Interventions ({len(self.interventions)})[/bold blue]")
            for i in self.interventions[:5]:
                int_branch.add(f"  {_truncate(i, 60)}")
            if len(self.interventions) > 5:
                int_branch.add(f"[dim]... and {len(self.interventions) - 5} more[/dim]")
        console.print(tree)
        if show_summary and self.rendered_summary:
            summary = self.rendered_summary[:2000] + ("..." if len(self.rendered_summary) > 2000 else "")
            console.print(Panel(summary, title="Study summary", border_style="dim"))


class TrialsGroupHit(BaseModel):
    group_key: str
    group_label: str
    n_trials: int = 0
    trials: list[Any] = Field(default_factory=list)

    def pretty_print(self, console: Console | None = None, show_summary: bool = False) -> None:
        """Pretty print a grouped set of trial hits."""
        console = console or Console()
        tree = Tree(
            f"[bold magenta]Group:[/bold magenta] {self.group_label} "
            f"[dim]| Trials: {self.n_trials}[/dim]"
        )
        for trial in self.trials:
            conf = f" (conf: {trial.confidence:.2f})" if trial.confidence is not None else ""
            trial_title = _truncate(trial.brief_title or "", 90)
            node = tree.add(f"[bold cyan]{trial.nct_id}[/bold cyan][dim]{conf}[/dim] {trial_title}")
            meta_parts: list[str] = []
            if trial.phase:
                meta_parts.append(f"Phase: {trial.phase}")
            if trial.status:
                meta_parts.append(f"Status: {trial.status}")
            if trial.enrollment is not None:
                meta_parts.append(f"Enrollment: {trial.enrollment:,}")
            if meta_parts:
                node.add(f"[dim]{' | '.join(meta_parts)}[/dim]")
        console.print(tree)
        if show_summary:
            for trial in self.trials:
                if trial.rendered_summary:
                    summary = trial.rendered_summary[:2000] + ("..." if len(trial.rendered_summary) > 2000 else "")
                    console.print(
                        Panel(
                            summary,
                            title=f"Study summary: {trial.nct_id}",
                            border_style="dim",
                        )
                    )


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
    # Extended trial info (from rag_study_search)
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    lead_sponsor: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    target_evidence: list[str] = Field(default_factory=list)
    rendered_summary: str | None = None

    def pretty_print(self, console: Console | None = None, show_summary: bool = False) -> None:
        """Pretty print a single trial-by-target hit."""
        console = console or Console()
        status_colors = {
            "RECRUITING": "green",
            "ACTIVE_NOT_RECRUITING": "yellow",
            "COMPLETED": "blue",
            "TERMINATED": "red",
            "WITHDRAWN": "red",
            "NOT_YET_RECRUITING": "cyan",
            "SUSPENDED": "orange3",
        }
        status_color = status_colors.get(self.status or "", "white")
        conf = f" (conf: {self.confidence:.2f})" if self.confidence is not None else ""
        title = f"[bold cyan]{self.nct_id}[/bold cyan][dim]{conf}[/dim]"
        tree = Tree(title)
        tree.add(f"[bold]{_truncate(self.brief_title or '', 100)}[/bold]")
        meta = tree.add("[bold yellow]Metadata[/bold yellow]")
        meta.add(f"[dim]Phase:[/dim] {self.phase or 'N/A'}")
        meta.add(f"[dim]Status:[/dim] [{status_color}]{self.status or 'N/A'}[/{status_color}]")
        if self.enrollment is not None:
            meta.add(f"[dim]Enrollment:[/dim] {self.enrollment:,}")
        if self.start_date:
            meta.add(f"[dim]Start:[/dim] {self.start_date}")
        if self.completion_date:
            meta.add(f"[dim]Completion:[/dim] {self.completion_date}")
        if self.lead_sponsor:
            meta.add(f"[dim]Sponsor:[/dim] {_truncate(self.lead_sponsor, 50)}")
        mol = tree.add("[bold blue]Molecule[/bold blue]")
        mol.add(f"[dim]Concept:[/dim] {self.concept_name or 'N/A'}")
        if self.match_type:
            mol.add(f"[dim]Match:[/dim] {self.match_type}")
        if self.conditions:
            cond_branch = tree.add(f"[bold green]Conditions ({len(self.conditions)})[/bold green]")
            for c in self.conditions[:5]:
                cond_branch.add(f"  {_truncate(c, 60)}")
            if len(self.conditions) > 5:
                cond_branch.add(f"[dim]... and {len(self.conditions) - 5} more[/dim]")
        if self.interventions:
            int_branch = tree.add(f"[bold blue]Interventions ({len(self.interventions)})[/bold blue]")
            for i in self.interventions[:5]:
                int_branch.add(f"  {_truncate(i, 60)}")
            if len(self.interventions) > 5:
                int_branch.add(f"[dim]... and {len(self.interventions) - 5} more[/dim]")
        if self.target_evidence:
            tree.add(f"[dim]Target evidence:[/dim] {', '.join(self.target_evidence)}")
        console.print(tree)
        if show_summary and self.rendered_summary:
            summary = self.rendered_summary[:2000] + ("..." if len(self.rendered_summary) > 2000 else "")
            console.print(Panel(summary, title="Study summary", border_style="dim"))


class StructureMatchedMolecule(BaseModel):
    """Summary of molecules matched by structure search."""
    concept_id: int
    concept_name: str | None = None
    chembl_id: str | None = None
    canonical_smiles: str | None = None
    similarity_score: float | None = None
    n_trials: int = 0


class StructureValidationInfo(BaseModel):
    """Information about structure preprocessing/validation."""
    original_input: str
    canonical_smiles: str | None = None
    was_modified: bool = False
    preprocessing_notes: list[str] = Field(default_factory=list)


class MoleculeTrialSearchOutput(BaseModel):
    status: Literal["success", "not_found", "error", "invalid_input", "invalid_structure"]
    mode: str
    total_hits: int = 0
    hits: list[Any] = Field(default_factory=list)
    matched_molecules: list[StructureMatchedMolecule] = Field(default_factory=list)
    
    # Structure validation info (for structure modes)
    structure_info: StructureValidationInfo | None = None
    
    error: str | None = None
    query_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    def pretty_print(
        self,
        console: Console | None = None,
        show_summary: bool = False,
    ) -> None:
        """Pretty print all hits."""
        console = console or Console()
        if self.query_summary:
            console.print(f"[bold]Query:[/bold] {self.query_summary}")
        console.print(f"[dim]Mode: {self.mode} | Total: {self.total_hits}[/dim]\n")
        for i, hit in enumerate(self.hits, 1):
            is_group_hit = hasattr(hit, "group_label") and hasattr(hit, "trials")
            if hasattr(hit, "pretty_print") and callable(hit.pretty_print):
                label = "Group" if is_group_hit else "Hit"
                console.print(f"[bold]--- {label} {i}/{len(self.hits)} ---[/bold]")
                hit.pretty_print(console=console, show_summary=show_summary)
            else:
                console.print(f"[bold]--- Hit {i}[/bold] {hit}")


# =============================================================================
# Helper Functions
# =============================================================================

def _bio_type_filter(bio_type: str, param_idx: int) -> tuple[str, list[Any]]:
    if bio_type == "all":
        return "", []
    return f"AND LOWER(b.biotherapeutic_type) = ${param_idx}", [bio_type.lower()]


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return text[:max_len] + "..." if len(text) > max_len else text


def _as_list(val: Any) -> list:
    """Convert PG array or None to list."""
    if val is None:
        return []
    if isinstance(val, list):
        return list(val)
    return []


def _study_json_to_dict(val: Any) -> dict[str, Any]:
    """Convert study_json (dict or JSON string) to dict for render_study_text_full."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}


def _row_to_trial_by_molecule_hit(
    row: dict[str, Any],
    include_study_details: bool = False,
    extra: dict[str, Any] | None = None,
) -> TrialByMoleculeHit:
    """Build TrialByMoleculeHit from a query row (e.g. from _get_trials_for_concepts)."""
    rendered = None
    if include_study_details and row.get("study_json"):
        rendered = render_study_text_full(
            _study_json_to_dict(row["study_json"]),
            nct_id=row.get("nct_id"),
        )
    kwargs: dict[str, Any] = {
        "nct_id": row["nct_id"],
        "brief_title": row.get("brief_title"),
        "phase": row.get("phase"),
        "status": row.get("overall_status"),
        "match_type": row.get("match_type"),
        "confidence": row.get("confidence"),
        "mol_id": row.get("mol_id"),
        "concept_id": row.get("concept_id"),
        "concept_name": row.get("concept_name"),
        "molecule_name": row.get("molecule_name"),
        "inchi_key": row.get("inchi_key"),
        "canonical_smiles": row.get("canonical_smiles"),
        "enrollment": row.get("enrollment"),
        "start_date": str(row["start_date"]) if row.get("start_date") is not None else None,
        "completion_date": str(row["completion_date"]) if row.get("completion_date") is not None else None,
        "lead_sponsor": row.get("lead_sponsor"),
        "conditions": _as_list(row.get("conditions")),
        "interventions": _as_list(row.get("interventions")),
        "countries": _as_list(row.get("countries")),
        "rendered_summary": rendered,
    }
    if extra:
        kwargs.update(extra)
    return TrialByMoleculeHit(**kwargs)


def _group_trial_hits(
    trial_hits: list[Any],
    group_by: Literal["none", "condition", "molecule_concept", "intervention"],
) -> list[TrialsGroupHit]:
    """Group trial hits by a selected dimension."""
    if group_by == "none":
        return []

    grouped: dict[str, dict[str, Any]] = {}

    for hit in trial_hits:
        keys_for_hit: list[tuple[str, str]] = []
        if group_by == "condition":
            if hit.conditions:
                for condition in hit.conditions:
                    label = (condition or "").strip()
                    if not label:
                        continue
                    keys_for_hit.append((f"condition:{label.lower()}", label))
            if not keys_for_hit:
                keys_for_hit.append(("condition:unknown", "Unknown condition"))
        elif group_by == "intervention":
            if hit.interventions:
                for intervention in hit.interventions:
                    label = (intervention or "").strip()
                    if not label:
                        continue
                    keys_for_hit.append((f"intervention:{label.lower()}", label))
            if not keys_for_hit:
                keys_for_hit.append(("intervention:unknown", "Unknown intervention"))
        elif group_by == "molecule_concept":
            if hit.concept_id is not None:
                label = (hit.concept_name or hit.molecule_name or f"Concept {hit.concept_id}").strip()
                keys_for_hit.append((f"molecule_concept:{hit.concept_id}", label))
            else:
                label = (hit.concept_name or hit.molecule_name or "").strip() or "Unknown molecule concept"
                keys_for_hit.append((f"molecule_concept:{label.lower()}", label))

        for group_key, group_label in keys_for_hit:
            bucket = grouped.setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "group_label": group_label,
                    "nct_seen": set(),
                    "trials": [],
                },
            )
            if hit.nct_id in bucket["nct_seen"]:
                continue
            bucket["nct_seen"].add(hit.nct_id)
            bucket["trials"].append(hit)

    groups = [
        TrialsGroupHit(
            group_key=data["group_key"],
            group_label=data["group_label"],
            n_trials=len(data["trials"]),
            trials=data["trials"],
        )
        for data in grouped.values()
    ]
    groups.sort(key=lambda g: (-g.n_trials, g.group_label.lower()))
    return groups


def _build_trial_filters(search_input: MoleculeTrialSearchInput, param_start: int) -> tuple[str, list[Any]]:
    """Build WHERE clauses and params for rag_study_search filters. Returns (sql_fragment, params_to_append)."""
    parts: list[str] = []
    extra: list[Any] = []
    idx = param_start
    inp = search_input
    if inp.start_date_from is not None:
        parts.append(f"AND rs.start_date_parsed >= ${idx}")
        extra.append(inp.start_date_from)
        idx += 1
    if inp.start_date_to is not None:
        parts.append(f"AND rs.start_date_parsed <= ${idx}")
        extra.append(inp.start_date_to)
        idx += 1
    if inp.completion_date_from is not None:
        parts.append(f"AND rs.completion_date_parsed >= ${idx}")
        extra.append(inp.completion_date_from)
        idx += 1
    if inp.completion_date_to is not None:
        parts.append(f"AND rs.completion_date_parsed <= ${idx}")
        extra.append(inp.completion_date_to)
        idx += 1
    if inp.min_enrollment is not None:
        parts.append(f"AND rs.enrollment >= ${idx}")
        extra.append(inp.min_enrollment)
        idx += 1
    if inp.max_enrollment is not None:
        parts.append(f"AND rs.enrollment <= ${idx}")
        extra.append(inp.max_enrollment)
        idx += 1
    if inp.lead_sponsor is not None and inp.lead_sponsor.strip():
        parts.append(f"AND rs.lead_sponsor ILIKE ${idx}")
        extra.append(f"%{inp.lead_sponsor.strip()}%")
        idx += 1
    if inp.country:
        parts.append(f"AND rs.countries && ${idx}::text[]")
        extra.append([c.strip() for c in inp.country if c.strip()])
        idx += 1
    if inp.has_results is True:
        parts.append("AND rs.results_first_submitted_date IS NOT NULL")
    elif inp.has_results is False:
        parts.append("AND rs.results_first_submitted_date IS NULL")
    if inp.is_fda_regulated is True:
        parts.append("AND rs.is_fda_regulated_drug = TRUE")
    elif inp.is_fda_regulated is False:
        parts.append("AND rs.is_fda_regulated_drug = FALSE")
    return " ".join(parts), extra


async def _find_molecule_ids(
    async_config: AsyncDatabaseConfig,
    molecule_name: str | None,
    inchi_key: str | None,
    limit: int,
    molecule_type: str,
) -> list[dict[str, Any]]:
    """Find molecules by name or InChIKey."""
    rows: list[dict[str, Any]] = []

    if molecule_type in ("all", "small_molecule"):
        params: list[Any] = []
        where_parts: list[str] = []

        if inchi_key:
            params.append(inchi_key)
            where_parts.append("m.inchi_key = $1")

        if molecule_name:
            name_param = molecule_name.lower()
            pattern = f"%{molecule_name}%"
            # Fast path: resolve by synonym first (uses index on synonym_lower).
            # Avoids full scan of dm_molecule when the name is a biotherapeutic or not found.
            sql_syn = """
                SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
                       m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
                FROM dm_molecule_synonyms s
                JOIN dm_molecule m ON s.mol_id = m.mol_id
                LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                WHERE s.synonym_lower = $1 OR s.synonym_lower ILIKE $2
                ORDER BY m.mol_id
                LIMIT $3
            """
            syn_rows = await async_config.execute_query(sql_syn, name_param, pattern, limit)
            rows.extend(syn_rows)
            # Only run pref_name/preferred_name lookups if synonym path returned few rows;
            # use two index-friendly queries (GIN on pref_name, GIN on preferred_name) instead of
            # one OR that triggers a full table scan.
            if len(syn_rows) < limit:
                name_rows = []
                # Query 1: by dm_molecule.pref_name (uses idx_dm_mol_pref_name_trgm)
                sql_pref = """
                    SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
                           m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
                    FROM dm_molecule m
                    LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                    WHERE m.pref_name ILIKE $1
                    ORDER BY m.mol_id
                    LIMIT $2
                """
                if inchi_key:
                    sql_pref = """
                    SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
                           m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
                    FROM dm_molecule m
                    LEFT JOIN dm_molecule_concept mc ON m.concept_id = mc.concept_id
                    WHERE m.pref_name ILIKE $2 AND m.inchi_key = $1
                    ORDER BY m.mol_id
                    LIMIT $3
                    """
                params_pref: list[Any] = [pattern, limit] if not inchi_key else [inchi_key, pattern, limit]
                name_rows.extend(await async_config.execute_query(sql_pref, *params_pref))
                # Query 2: by dm_molecule_concept.preferred_name (uses idx_concept_name_trgm)
                sql_concept = """
                    SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
                           m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
                    FROM dm_molecule_concept mc
                    JOIN dm_molecule m ON m.concept_id = mc.concept_id
                    WHERE mc.preferred_name ILIKE $1
                    ORDER BY m.mol_id
                    LIMIT $2
                """
                if inchi_key:
                    sql_concept = """
                    SELECT m.mol_id, m.concept_id, m.pref_name AS molecule_name,
                           m.inchi_key, m.canonical_smiles, mc.preferred_name AS concept_name
                    FROM dm_molecule_concept mc
                    JOIN dm_molecule m ON m.concept_id = mc.concept_id
                    WHERE mc.preferred_name ILIKE $2 AND m.inchi_key = $1
                    ORDER BY m.mol_id
                    LIMIT $3
                    """
                params_concept: list[Any] = [pattern, limit] if not inchi_key else [inchi_key, pattern, limit]
                name_rows.extend(await async_config.execute_query(sql_concept, *params_concept))
                rows.extend(name_rows)
        elif where_parts:
            # inchi_key only (no molecule_name)
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
            rows.extend(await async_config.execute_query(sql, *params))

    if molecule_type in ("all", "biotherapeutic") and molecule_name:
        name_param = molecule_name.lower()
        params = [name_param, f"%{molecule_name}%", limit]
        # Use exact match or substring on synonyms; trigram similarity (%)
        # matches similar names (e.g. ruplizumab, atezolizumab for pembrolizumab).
        sql = """
            SELECT NULL::bigint AS mol_id,
                   dbt.concept_id,
                   dbt.pref_name AS molecule_name,
                   NULL AS inchi_key,
                   NULL AS canonical_smiles,
                   mc.preferred_name AS concept_name
            FROM dm_biotherapeutic dbt
            LEFT JOIN dm_molecule_concept mc ON dbt.concept_id = mc.concept_id
            WHERE dbt.pref_name ILIKE $2
               OR mc.preferred_name ILIKE $2
               OR EXISTS (
                   SELECT 1 FROM dm_biotherapeutic_synonyms s
                   WHERE s.bio_id = dbt.bio_id
                     AND (s.synonym_lower = $1 OR s.synonym_lower ILIKE $2)
               )
            ORDER BY dbt.bio_id
            LIMIT $3
        """
        rows.extend(await async_config.execute_query(sql, *params))

    if not rows:
        return []


    seen: set[tuple[int | None, int | None]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("mol_id"), row.get("concept_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


async def _find_similar_molecules_with_trials(
    async_config: AsyncDatabaseConfig,
    smiles: str,
    similarity_threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Find molecules similar to query SMILES that have clinical trial associations."""
    
    sql = """
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
                dm.inchi_key,
                tanimoto_sml(q.fp, dm.mfp2)::float AS similarity
            FROM dm_molecule dm
            CROSS JOIN query_fp q
            WHERE dm.mfp2 % q.fp
              AND tanimoto_sml(q.fp, dm.mfp2) >= $2
            ORDER BY similarity DESC
            LIMIT 500
        ),
        mols_with_trials AS (
            SELECT DISTINCT sm.concept_id
            FROM similar_mols sm
            JOIN map_ctgov_molecules map ON map.concept_id = sm.concept_id
        )
        SELECT DISTINCT ON (sm.concept_id)
            sm.mol_id,
            sm.concept_id,
            COALESCE(mc.preferred_name, sm.molecule_name) AS concept_name,
            sm.canonical_smiles,
            sm.chembl_id,
            sm.inchi_key,
            sm.similarity
        FROM similar_mols sm
        JOIN mols_with_trials mwt ON sm.concept_id = mwt.concept_id
        LEFT JOIN dm_molecule_concept mc ON sm.concept_id = mc.concept_id
        ORDER BY sm.concept_id, sm.similarity DESC
        LIMIT $3
    """
    
    return await async_config.execute_query(sql, smiles, similarity_threshold, limit)


async def _find_substructure_molecules_with_trials(
    async_config: AsyncDatabaseConfig,
    pattern: str,
    is_smarts: bool,
    limit: int,
) -> list[dict[str, Any]]:
    """Find molecules containing substructure that have clinical trial associations."""
    
    if is_smarts:
        match_clause = "dm.mol @> qmol_from_smarts($1::cstring)"
    else:
        match_clause = "dm.mol @> mol_from_smiles($1::cstring)"
    
    sql = f"""
        WITH matching_mols AS (
            SELECT 
                dm.mol_id,
                dm.concept_id,
                dm.pref_name AS molecule_name,
                dm.canonical_smiles,
                dm.chembl_id,
                dm.inchi_key
            FROM dm_molecule dm
            WHERE {match_clause}
            LIMIT 1000
        ),
        mols_with_trials AS (
            SELECT DISTINCT mm.concept_id
            FROM matching_mols mm
            JOIN map_ctgov_molecules map ON map.concept_id = mm.concept_id
        )
        SELECT DISTINCT ON (mm.concept_id)
            mm.mol_id,
            mm.concept_id,
            COALESCE(mc.preferred_name, mm.molecule_name) AS concept_name,
            mm.canonical_smiles,
            mm.chembl_id,
            mm.inchi_key
        FROM matching_mols mm
        JOIN mols_with_trials mwt ON mm.concept_id = mwt.concept_id
        LEFT JOIN dm_molecule_concept mc ON mm.concept_id = mc.concept_id
        ORDER BY mm.concept_id, mm.mol_id
        LIMIT $2
    """
    
    return await async_config.execute_query(sql, pattern, limit)


async def _get_trials_for_concepts(
    async_config: AsyncDatabaseConfig,
    concept_ids: list[int],
    phase: list[str] | None,
    status: list[str] | None,
    limit: int,
    offset: int,
    search_input: MoleculeTrialSearchInput | None = None,
) -> list[dict[str, Any]]:
    """Get trials for a list of concept IDs with optional filters."""
    
    params: list[Any] = [concept_ids]
    param_idx = 1
    
    where_clauses = ["map.concept_id = ANY($1::bigint[])"]
    
    if phase:
        param_idx += 1
        params.append(phase)
        where_clauses.append(f"rs.phase = ANY(${param_idx}::text[])")
    
    if status:
        param_idx += 1
        params.append(status)
        where_clauses.append(f"rs.overall_status = ANY(${param_idx}::text[])")
    
    if search_input:
        trial_frag, trial_params = _build_trial_filters(search_input, param_idx + 1)
        if trial_frag:
            params.extend(trial_params)
            clause = trial_frag.strip()
            if clause.upper().startswith("AND "):
                clause = clause[4:]
            where_clauses.append(clause)
            param_idx += len(trial_params)
    
    params.extend([limit, offset])
    
    study_join = ""
    study_col = ""
    if search_input and getattr(search_input, "include_study_details", False):
        study_join = "LEFT JOIN rag_study_corpus c ON rs.nct_id = c.nct_id"
        study_col = ", rs.enrollment, rs.start_date, rs.completion_date, rs.lead_sponsor, rs.conditions, rs.interventions, rs.countries, c.study_json"
    else:
        study_col = ", rs.enrollment, rs.start_date, rs.completion_date, rs.lead_sponsor, rs.conditions, rs.interventions, rs.countries"
    
    sql = f"""
        SELECT 
            rs.nct_id,
            rs.brief_title,
            rs.phase,
            rs.overall_status,
            map.match_type,
            map.confidence,
            map.mol_id,
            map.concept_id,
            mc.preferred_name AS concept_name,
            m.pref_name AS molecule_name,
            m.inchi_key,
            m.canonical_smiles
            {study_col}
        FROM map_ctgov_molecules map
        JOIN rag_study_search rs ON rs.nct_id = map.nct_id
        {study_join}
        LEFT JOIN dm_molecule m ON map.mol_id = m.mol_id
        LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
        LIMIT ${param_idx + 1} OFFSET ${param_idx + 2}
    """
    
    return await async_config.execute_query(sql, *params)


async def _get_semantic_condition_terms(
    async_config: AsyncDatabaseConfig,
    condition: str,
    top_k: int = 3,
) -> list[str]:
    """Fetch semantically related indication names for condition fallback."""
    try:
        query_vector = encode_query_vector(condition)
        rows = await async_config.execute_query(
            """
            SELECT preferred_name
            FROM dm_indication
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            query_vector,
            top_k + 1,
        )
        base = normalize_semantic_text(condition)
        expanded: list[str] = []
        for row in rows:
            candidate = (row.get("preferred_name") or "").strip()
            if not candidate:
                continue
            if normalize_semantic_text(candidate) == base:
                continue
            expanded.append(candidate)
            if len(expanded) >= top_k:
                break
        return expanded
    except Exception:
        return []


def _build_structure_error_response(
    mode: str,
    validation_result: SmilesValidationResult,
    is_smarts: bool = False,
) -> MoleculeTrialSearchOutput:
    """Build a detailed error response for invalid structure input."""
    
    structure_type = "SMARTS" if is_smarts else "SMILES"
    original = validation_result.original_smiles or ""
    display = original[:50] + "..." if len(original) > 50 else original
    
    suggestions = []
    
    if validation_result.error_type == "syntax":
        suggestions.extend([
            "Check bracket balance: [] and ()",
            "Verify ring closure numbers are paired (e.g., C1CC1)",
            "Use a molecule editor to generate valid SMILES",
        ])
    elif validation_result.error_type == "parse_error":
        suggestions.extend([
            "Verify element symbols: Cl (not CL), Br (not BR)",
            "Aromatic atoms should be lowercase: c, n, o, s",
            "Use a chemical structure tool to validate your SMILES",
        ])
    elif validation_result.error_type == "empty":
        suggestions.append("Provide a non-empty structure string")
    
    if is_smarts:
        suggestions.append("SMARTS reference: https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html")
    else:
        suggestions.append("SMILES reference: https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html")
    
    return MoleculeTrialSearchOutput(
        status="invalid_structure",
        mode=mode,
        error=validation_result.error_message,
        query_summary=f"Invalid {structure_type}: '{display}'",
        warnings=validation_result.warnings,
        suggestions=suggestions,
        structure_info=StructureValidationInfo(
            original_input=original,
            was_modified=bool(validation_result.warnings),
            preprocessing_notes=validation_result.warnings,
        ),
    )


# =============================================================================
# Main Search Function
# =============================================================================

async def molecule_trial_search_async(
    db_config: DatabaseConfig,
    search_input: MoleculeTrialSearchInput,
) -> MoleculeTrialSearchOutput:
    """
    Search trial connectivity for molecules, conditions, targets, and structures.

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

        # =====================================================================
        # MODE: trials_by_molecule
        # =====================================================================
        if search_input.mode == "trials_by_molecule":

            mol_rows = await _find_molecule_ids(
                async_config,
                search_input.molecule_name,
                search_input.inchi_key,
                limit=min(200, search_input.limit * 5),
            molecule_type=search_input.molecule_type,
            )
            if not mol_rows:
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=search_input.molecule_name or search_input.inchi_key or "",
                )

            mol_ids = [row["mol_id"] for row in mol_rows if row.get("mol_id") is not None]
            concept_ids = [row["concept_id"] for row in mol_rows if row.get("concept_id") is not None]
            params: list[Any] = [mol_ids, concept_ids]
            trial_frag, trial_params = _build_trial_filters(search_input, 3)
            params.extend(trial_params)
            params.extend([search_input.limit, search_input.offset])
            study_join = "LEFT JOIN rag_study_corpus c ON rs.nct_id = c.nct_id" if search_input.include_study_details else ""
            study_col = ", c.study_json" if search_input.include_study_details else ""
            sql = f"""
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.mol_id, map.concept_id,
                       mc.preferred_name AS concept_name, m.pref_name AS molecule_name,
                       m.inchi_key, m.canonical_smiles,
                       rs.enrollment, rs.start_date, rs.completion_date, rs.lead_sponsor,
                       rs.conditions, rs.interventions, rs.countries
                       {study_col}
                FROM map_ctgov_molecules map
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                {study_join}
                LEFT JOIN dm_molecule m ON map.mol_id = m.mol_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE (map.mol_id = ANY($1) OR map.concept_id = ANY($2))
                {trial_frag}
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT ${len(params)-1} OFFSET ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            trial_hits = [
                _row_to_trial_by_molecule_hit(
                    row,
                    include_study_details=search_input.include_study_details,
                )
                for row in rows
            ]
            if search_input.group_by == "none":
                hits: list[Any] = trial_hits
            else:
                hits = _group_trial_hits(trial_hits, search_input.group_by)
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.molecule_name or search_input.inchi_key or "",
            )

        # =====================================================================
        # MODE: molecules_by_condition
        # =====================================================================
        if search_input.mode == "molecules_by_condition":
            condition = search_input.condition or ""
            semantic_terms = await _get_semantic_condition_terms(async_config, condition)
            terms: list[str] = [condition]
            terms.extend(semantic_terms)
            normalized_terms: list[str] = []
            seen_terms: set[str] = set()
            for term in terms:
                normalized = normalize_semantic_text(term)
                if not normalized or normalized in seen_terms:
                    continue
                seen_terms.add(normalized)
                normalized_terms.append(normalized)
            if not normalized_terms:
                normalized_terms = [normalize_semantic_text(condition)]

            term_values: list[str] = []
            params: list[Any] = []
            for term in normalized_terms[:5]:
                params.append(term)
                term_idx = len(params)
                params.append(_sanitize_tsquery(term))
                ts_idx = len(params)
                term_values.append(f"(${term_idx}, ${ts_idx})")

            type_filter = ""
            if search_input.molecule_type == "biotherapeutic":
                type_filter = "AND mc.is_biotherapeutic = TRUE"
            elif search_input.molecule_type == "small_molecule":
                type_filter = "AND (mc.is_biotherapeutic IS FALSE OR mc.is_biotherapeutic IS NULL)"
            trial_frag, trial_params = _build_trial_filters(search_input, len(params) + 1)
            params.extend(trial_params)
            params.extend([search_input.limit, search_input.offset])
            sql = f"""
                WITH condition_terms(term, tsquery) AS (
                    VALUES {", ".join(term_values)}
                ),
                matched_trials AS (
                    SELECT nct_id
                    FROM rag_study_search rs
                    JOIN condition_terms ct ON TRUE
                    WHERE (
                        rs.conditions_norm % ct.term
                        OR rs.conditions_norm ILIKE '%' || ct.term || '%'
                        OR rs.terms_tsv @@ to_tsquery('english', ct.tsquery)
                    )
                    {trial_frag}
                )
                SELECT mc.concept_id, mc.preferred_name AS concept_name,
                       COUNT(DISTINCT map.nct_id)::int AS n_trials
                FROM map_ctgov_molecules map
                JOIN matched_trials mt ON mt.nct_id = map.nct_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE 1=1 {type_filter}
                GROUP BY mc.concept_id, mc.preferred_name
                ORDER BY n_trials DESC, mc.preferred_name
                LIMIT ${len(params)-1} OFFSET ${len(params)}
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

        # =====================================================================
        # MODE: trials_by_target
        # =====================================================================
        # Default: dm_drug_mechanism only (curated). Optional: include_activity_data adds dm_compound_target_activity.
        # Match strategy is strict by gene symbol / target identifiers to avoid false positives
        # from short-symbol substring matches (e.g., MET in "metabolism", "metalloproteinase").
        if search_input.mode == "trials_by_target":
            target_gene = (search_input.target_gene or "").upper()
            params: list[Any] = [target_gene]
            molecule_type_filter = ""
            if search_input.molecule_type == "biotherapeutic":
                molecule_type_filter = "AND mcf.is_biotherapeutic = TRUE"
            elif search_input.molecule_type == "small_molecule":
                molecule_type_filter = "AND (mcf.is_biotherapeutic IS FALSE OR mcf.is_biotherapeutic IS NULL)"
            activity_cte = """
                , activity_concepts AS (
                    SELECT NULL::bigint AS concept_id, FALSE::boolean AS activity_match
                    WHERE FALSE
                )
            """
            if search_input.include_activity_data:
                params.append(search_input.min_pchembl)
                pchembl_idx = len(params)
                activity_cte = f"""
                , activity_concepts AS (
                    SELECT
                        act.concept_id,
                        TRUE::boolean AS activity_match
                    FROM dm_compound_target_activity act
                    LEFT JOIN dm_molecule_concept mcf ON act.concept_id = mcf.concept_id
                    WHERE act.concept_id IS NOT NULL
                      AND (
                        LOWER(TRIM(act.gene_symbol)) IN (SELECT alias FROM target_aliases)
                        OR act.target_id IN (SELECT target_id FROM resolved_targets)
                      )
                      AND (act.pchembl_value IS NULL OR act.pchembl_value >= ${pchembl_idx})
                      AND act.concept_id IS NOT NULL
                      {molecule_type_filter}
                    GROUP BY act.concept_id
                )
                """
            phase_filter = ""
            if search_input.phase:
                params.append(search_input.phase)
                phase_filter = f"AND rs.phase = ANY(${len(params)}::text[])"

            trial_frag, trial_params = _build_trial_filters(search_input, len(params) + 1)
            params.extend(trial_params)
            params.extend([search_input.limit, search_input.offset])
            n_params = len(params)
            limit_ph = n_params - 1
            offset_ph = n_params
            study_join = "LEFT JOIN rag_study_corpus c ON rs.nct_id = c.nct_id" if search_input.include_study_details else ""
            study_col = ", c.study_json" if search_input.include_study_details else ""
            # Bind gene once in a CTE to avoid asyncpg param binding issues in nested subqueries
            sql = f"""
                WITH gene_param AS (SELECT $1::text AS gene),
                resolved_targets AS (
                    SELECT DISTINCT
                        dt.target_id,
                        dt.chembl_tid,
                        dt.chembl_id,
                        LOWER(TRIM(dt.gene_symbol)) AS gene_symbol_norm
                    FROM dm_target dt
                    CROSS JOIN gene_param gp
                    WHERE LOWER(TRIM(dt.gene_symbol)) = LOWER(TRIM(gp.gene))
                ),
                synonym_resolved_targets AS (
                    SELECT DISTINCT
                        dt.target_id,
                        dt.chembl_tid,
                        dt.chembl_id,
                        LOWER(TRIM(dt.gene_symbol)) AS gene_symbol_norm
                    FROM dm_target_gene_synonyms s
                    JOIN dm_target dt ON dt.target_id = s.target_id
                    CROSS JOIN gene_param gp
                    WHERE s.symbol_type = 'SYNONYM'
                      AND LOWER(TRIM(s.gene_symbol)) = LOWER(TRIM(gp.gene))
                ),
                strict_targets AS (
                    SELECT * FROM resolved_targets
                    UNION
                    SELECT * FROM synonym_resolved_targets
                ),
                target_aliases AS (
                    SELECT DISTINCT LOWER(TRIM(gene)) AS alias
                    FROM gene_param
                    UNION
                    SELECT DISTINCT gene_symbol_norm AS alias
                    FROM strict_targets
                    UNION
                    SELECT DISTINCT LOWER(TRIM(s.gene_symbol)) AS alias
                    FROM dm_target_gene_synonyms s
                    JOIN strict_targets st ON st.target_id = s.target_id
                    WHERE s.symbol_type = 'SYNONYM'
                      AND s.gene_symbol IS NOT NULL
                ),
                mechanism_concepts AS (
                    SELECT
                        ddm.concept_id,
                        BOOL_OR(
                            ddm.gene_symbols IS NOT NULL
                            AND EXISTS (
                                SELECT 1
                                FROM unnest(ddm.gene_symbols) AS g
                                WHERE LOWER(TRIM(g)) IN (SELECT alias FROM target_aliases)
                            )
                        ) AS gene_symbols_exact,
                        BOOL_OR(ddm.tid IN (SELECT chembl_tid FROM strict_targets)) AS tid_match,
                        BOOL_OR(ddm.target_chembl_id IN (SELECT chembl_id FROM strict_targets)) AS target_chembl_match,
                        TRUE::boolean AS mechanism_match
                    FROM dm_drug_mechanism ddm
                    LEFT JOIN dm_molecule_concept mcf ON ddm.concept_id = mcf.concept_id
                    WHERE ddm.concept_id IS NOT NULL
                      AND (
                          (ddm.gene_symbols IS NOT NULL AND EXISTS (
                              SELECT 1 FROM unnest(ddm.gene_symbols) AS g
                              WHERE LOWER(TRIM(g)) IN (SELECT alias FROM target_aliases)
                          ))
                          OR ddm.tid IN (SELECT chembl_tid FROM strict_targets)
                          OR ddm.target_chembl_id IN (SELECT chembl_id FROM strict_targets)
                      )
                      {molecule_type_filter}
                    GROUP BY ddm.concept_id
                )
                {activity_cte},
                target_concepts AS (
                    SELECT
                        concept_id,
                        BOOL_OR(gene_symbols_exact) AS gene_symbols_exact,
                        BOOL_OR(tid_match) AS tid_match,
                        BOOL_OR(target_chembl_match) AS target_chembl_match,
                        BOOL_OR(mechanism_match) AS mechanism_match,
                        BOOL_OR(activity_match) AS activity_match
                    FROM (
                        SELECT
                            mc.concept_id,
                            mc.gene_symbols_exact,
                            mc.tid_match,
                            mc.target_chembl_match,
                            mc.mechanism_match,
                            FALSE::boolean AS activity_match
                        FROM mechanism_concepts mc
                        UNION ALL
                        SELECT
                            ac.concept_id,
                            FALSE::boolean AS gene_symbols_exact,
                            FALSE::boolean AS tid_match,
                            FALSE::boolean AS target_chembl_match,
                            FALSE::boolean AS mechanism_match,
                            ac.activity_match
                        FROM activity_concepts ac
                    ) combined
                    GROUP BY concept_id
                ),
                dedup_map AS (
                    SELECT DISTINCT ON (map.nct_id, map.concept_id)
                        map.nct_id,
                        map.concept_id,
                        map.match_type,
                        map.confidence,
                        tc.gene_symbols_exact,
                        tc.tid_match,
                        tc.target_chembl_match,
                        tc.mechanism_match,
                        tc.activity_match
                    FROM target_concepts tc
                    JOIN map_ctgov_molecules map ON map.concept_id = tc.concept_id
                    ORDER BY
                        map.nct_id,
                        map.concept_id,
                        tc.mechanism_match DESC,
                        tc.gene_symbols_exact DESC,
                        tc.tid_match DESC,
                        tc.target_chembl_match DESC,
                        map.confidence DESC NULLS LAST,
                        CASE map.match_type
                            WHEN 'EXACT' THEN 1
                            WHEN 'SALT_STRIPPED' THEN 2
                            WHEN 'COMBO_PART' THEN 3
                            WHEN 'FUZZY' THEN 4
                            ELSE 9
                        END,
                        map.id
                )
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.concept_id,
                       map.gene_symbols_exact, map.tid_match, map.target_chembl_match,
                       map.mechanism_match, map.activity_match,
                       mc.preferred_name AS concept_name,
                       rs.enrollment, rs.start_date, rs.completion_date, rs.lead_sponsor,
                       rs.conditions, rs.interventions, rs.countries
                       {study_col}
                FROM dedup_map map
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                {study_join}
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE 1=1 {phase_filter} {trial_frag}
                ORDER BY
                    map.mechanism_match DESC,
                    map.gene_symbols_exact DESC,
                    map.tid_match DESC,
                    map.target_chembl_match DESC,
                    map.confidence DESC NULLS LAST,
                    rs.nct_id
                LIMIT ${limit_ph} OFFSET ${offset_ph}
            """
            rows = await async_config.execute_query(sql, *params)
            trial_hits: list[TrialByTargetHit] = []
            for row in rows:
                rendered = None
                if search_input.include_study_details and row.get("study_json"):
                    rendered = render_study_text_full(
                        _study_json_to_dict(row["study_json"]),
                        nct_id=row.get("nct_id"),
                    )
                target_evidence: list[str] = []
                if row.get("gene_symbols_exact"):
                    target_evidence.append("gene_symbols_exact")
                if row.get("tid_match"):
                    target_evidence.append("tid_match")
                if row.get("target_chembl_match"):
                    target_evidence.append("target_chembl_match")
                if row.get("mechanism_match"):
                    target_evidence.append("mechanism_match")
                if row.get("activity_match"):
                    target_evidence.append("activity_match")
                trial_hits.append(TrialByTargetHit(
                    nct_id=row["nct_id"],
                    brief_title=row.get("brief_title"),
                    phase=row.get("phase"),
                    status=row.get("overall_status"),
                    match_type=row.get("match_type"),
                    confidence=row.get("confidence"),
                    concept_id=row.get("concept_id"),
                    concept_name=row.get("concept_name"),
                    enrollment=row.get("enrollment"),
                    start_date=str(row["start_date"]) if row.get("start_date") is not None else None,
                    completion_date=str(row["completion_date"]) if row.get("completion_date") is not None else None,
                    lead_sponsor=row.get("lead_sponsor"),
                    conditions=_as_list(row.get("conditions")),
                    interventions=_as_list(row.get("interventions")),
                    countries=_as_list(row.get("countries")),
                    target_evidence=target_evidence,
                    rendered_summary=rendered,
                ))
            if search_input.group_by in ("condition", "intervention", "molecule_concept"):
                hits: list[Any] = _group_trial_hits(trial_hits, search_input.group_by)
            else:
                hits = trial_hits
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.target_gene or "",
            )

        # =====================================================================
        # MODE: trials_by_sequence
        # =====================================================================
        if search_input.mode == "trials_by_sequence":
            if search_input.molecule_type == "small_molecule":
                return MoleculeTrialSearchOutput(
                    status="invalid_input",
                    mode=search_input.mode,
                    error="molecule_type='small_molecule' is not valid for sequence-based trials",
                    query_summary=search_input.sequence or "",
                    suggestions=["Use trials_by_structure or trials_by_substructure for small molecules."],
                )
            sequence = search_input.sequence or ""
            seq_pattern = f"%{sequence}%"
            params = [seq_pattern]
            type_clause, type_params = _bio_type_filter(search_input.biotherapeutic_type, 2)
            params.extend(type_params)
            params.append(min(200, search_input.limit * 5))
            sql = f"""
                SELECT DISTINCT b.concept_id
                FROM dm_biotherapeutic b
                JOIN dm_biotherapeutic_component c ON b.bio_id = c.bio_id
                WHERE c.sequence ILIKE $1
                {type_clause}
                LIMIT ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            concept_ids = [row["concept_id"] for row in rows if row.get("concept_id") is not None]
            if not concept_ids:
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=sequence,
                )
            trial_rows = await _get_trials_for_concepts(
                async_config,
                concept_ids,
                search_input.phase,
                search_input.status,
                search_input.limit,
                search_input.offset,
                search_input,
            )
            hits = [
                _row_to_trial_by_molecule_hit(row, search_input.include_study_details)
                for row in trial_rows
            ]
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=sequence,
            )

        # =====================================================================
        # MODE: trials_by_structure (similarity search)
        # =====================================================================
        if search_input.mode == "trials_by_structure":
            if search_input.molecule_type == "biotherapeutic":
                return MoleculeTrialSearchOutput(
                    status="invalid_input",
                    mode=search_input.mode,
                    error="molecule_type='biotherapeutic' is not supported for structure-based trials",
                    query_summary=search_input.smiles or "",
                    suggestions=["Use trials_by_sequence for biologics."],
                )
            smiles = search_input.smiles
            
            # Validate and canonicalize SMILES
            validation = await _validate_and_canonicalize_smiles(async_config, smiles)
            
            if not validation.is_valid:
                return _build_structure_error_response(
                    search_input.mode, validation, is_smarts=False
                )
            
            canonical_smiles = validation.canonical_smiles
            
            # Find similar molecules with trial associations
            mol_rows = await _find_similar_molecules_with_trials(
                async_config,
                canonical_smiles,
                search_input.similarity_threshold,
                limit=100,
            )
            
            if not mol_rows:
                smiles_display = smiles[:30] + "..." if len(smiles) > 30 else smiles
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=f"similarity >= {search_input.similarity_threshold}",
                    warnings=[
                        f"No molecules with trial associations found at similarity >= {search_input.similarity_threshold}",
                    ] + validation.warnings,
                    suggestions=[
                        f"Try lowering similarity_threshold (current: {search_input.similarity_threshold})",
                        "Try 0.5 or 0.6 for broader structural matches",
                        "The query structure may not have analogs in clinical development",
                    ],
                    structure_info=StructureValidationInfo(
                        original_input=smiles,
                        canonical_smiles=canonical_smiles,
                        was_modified=smiles != canonical_smiles,
                        preprocessing_notes=validation.warnings,
                    ),
                )
            
            # Get trial details
            concept_ids = [r["concept_id"] for r in mol_rows if r.get("concept_id")]
            similarity_lookup = {r["concept_id"]: r["similarity"] for r in mol_rows}
            
            trial_rows = await _get_trials_for_concepts(
                async_config,
                concept_ids,
                search_input.phase,
                search_input.status,
                search_input.limit,
                search_input.offset,
                search_input,
            )
            
            # Count trials per molecule
            trial_counts: dict[int, int] = {}
            for row in trial_rows:
                cid = row.get("concept_id")
                if cid:
                    trial_counts[cid] = trial_counts.get(cid, 0) + 1
            
            # Build hits
            hits = []
            for row in trial_rows:
                concept_id = row.get("concept_id")
                hit = _row_to_trial_by_molecule_hit(
                    row,
                    search_input.include_study_details,
                    extra={
                        "similarity_score": similarity_lookup.get(concept_id),
                        "structure_match_type": "similarity",
                    },
                )
                hits.append(hit)
            
            # Build matched molecules summary
            matched_molecules = [
                StructureMatchedMolecule(
                    concept_id=r["concept_id"],
                    concept_name=r.get("concept_name"),
                    chembl_id=r.get("chembl_id"),
                    canonical_smiles=r.get("canonical_smiles"),
                    similarity_score=r.get("similarity"),
                    n_trials=trial_counts.get(r["concept_id"], 0),
                )
                for r in mol_rows
                if r.get("concept_id") in trial_counts
            ]
            matched_molecules.sort(key=lambda m: -(m.similarity_score or 0))
            
            smiles_display = smiles[:30] + "..." if len(smiles) > 30 else smiles
            
            return MoleculeTrialSearchOutput(
                status="success" if hits else "not_found",
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                matched_molecules=matched_molecules[:20],
                query_summary=f"'{smiles_display}' (>= {search_input.similarity_threshold} sim, {len(matched_molecules)} mols)",
                warnings=validation.warnings if validation.warnings else [],
                structure_info=StructureValidationInfo(
                    original_input=smiles,
                    canonical_smiles=canonical_smiles,
                    was_modified=smiles != canonical_smiles,
                    preprocessing_notes=validation.warnings,
                ),
            )

        # =====================================================================
        # MODE: trials_by_substructure
        # =====================================================================
        if search_input.mode == "trials_by_substructure":
            if search_input.molecule_type == "biotherapeutic":
                return MoleculeTrialSearchOutput(
                    status="invalid_input",
                    mode=search_input.mode,
                    error="molecule_type='biotherapeutic' is not supported for substructure trials",
                    query_summary=search_input.smiles or search_input.smarts or "",
                    suggestions=["Use trials_by_sequence for biologics."],
                )
            is_smarts = search_input.smarts is not None
            pattern = search_input.smarts if is_smarts else search_input.smiles
            
            # Validate structure
            if is_smarts:
                validation = await _validate_smarts_pattern(async_config, pattern)
            else:
                validation = await _validate_and_canonicalize_smiles(async_config, pattern)
            
            if not validation.is_valid:
                return _build_structure_error_response(
                    search_input.mode, validation, is_smarts=is_smarts
                )
            
            # Use canonical form for search
            search_pattern = validation.canonical_smiles or pattern
            
            # Find molecules containing substructure with trial associations
            mol_rows = await _find_substructure_molecules_with_trials(
                async_config,
                search_pattern,
                is_smarts,
                limit=100,
            )
            
            if not mol_rows:
                pattern_display = pattern[:30] + "..." if len(pattern) > 30 else pattern
                pattern_type = "SMARTS" if is_smarts else "SMILES"
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=f"substructure '{pattern_display}'",
                    warnings=[
                        f"No molecules with trial associations contain this substructure",
                    ] + validation.warnings,
                    suggestions=[
                        "Try a simpler or more common substructure pattern",
                        "SMARTS patterns are more flexible than SMILES for substructure search",
                        f"Example SMARTS: 'c1ccccc1' (benzene), 'C(=O)N' (amide)",
                    ],
                    structure_info=StructureValidationInfo(
                        original_input=pattern,
                        canonical_smiles=validation.canonical_smiles,
                        was_modified=bool(validation.warnings),
                        preprocessing_notes=validation.warnings,
                    ),
                )
            
            # Get trial details
            concept_ids = [r["concept_id"] for r in mol_rows if r.get("concept_id")]
            
            trial_rows = await _get_trials_for_concepts(
                async_config,
                concept_ids,
                search_input.phase,
                search_input.status,
                search_input.limit,
                search_input.offset,
                search_input,
            )
            
            # Count trials per molecule
            trial_counts: dict[int, int] = {}
            for row in trial_rows:
                cid = row.get("concept_id")
                if cid:
                    trial_counts[cid] = trial_counts.get(cid, 0) + 1
            
            # Build hits
            hits = [
                _row_to_trial_by_molecule_hit(
                    row,
                    search_input.include_study_details,
                    extra={"structure_match_type": "substructure"},
                )
                for row in trial_rows
            ]
            
            # Build matched molecules summary
            matched_molecules = [
                StructureMatchedMolecule(
                    concept_id=r["concept_id"],
                    concept_name=r.get("concept_name"),
                    chembl_id=r.get("chembl_id"),
                    canonical_smiles=r.get("canonical_smiles"),
                    n_trials=trial_counts.get(r["concept_id"], 0),
                )
                for r in mol_rows
                if r.get("concept_id") in trial_counts
            ]
            
            pattern_display = pattern[:30] + "..." if len(pattern) > 30 else pattern
            pattern_type = "SMARTS" if is_smarts else "SMILES"
            
            return MoleculeTrialSearchOutput(
                status="success" if hits else "not_found",
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                matched_molecules=matched_molecules[:20],
                query_summary=f"{pattern_type} '{pattern_display}' ({len(matched_molecules)} molecules)",
                warnings=validation.warnings if validation.warnings else [],
                structure_info=StructureValidationInfo(
                    original_input=pattern,
                    canonical_smiles=validation.canonical_smiles,
                    was_modified=bool(validation.warnings),
                    preprocessing_notes=validation.warnings,
                ),
            )

        # =====================================================================
        # Unknown mode
        # =====================================================================
        return MoleculeTrialSearchOutput(
            status="invalid_input",
            mode=search_input.mode,
            error="Unsupported mode",
            query_summary="invalid mode",
            suggestions=[
                "Supported modes: trials_by_molecule, molecules_by_condition, "
                "trials_by_target, trials_by_structure, trials_by_substructure"
            ],
        )

    except Exception as exc:
        return MoleculeTrialSearchOutput(
            status="error",
            mode=search_input.mode,
            error=f"{type(exc).__name__}: {exc}",
            query_summary="error",
        )