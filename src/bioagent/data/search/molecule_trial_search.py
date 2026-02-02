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

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig


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
    ]
    
    # Text-based inputs
    molecule_name: str | None = None
    inchi_key: str | None = None
    target_gene: str | None = None
    condition: str | None = None
    
    # Structure-based inputs
    smiles: str | None = None
    smarts: str | None = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Filters
    min_pchembl: float = Field(default=6.0, ge=0.0, le=15.0)
    phase: list[str] | None = None
    status: list[str] | None = None
    
    # Pagination
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @field_validator("molecule_name", "inchi_key", "target_gene", "condition", "smiles", "smarts", mode="before")
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


# =============================================================================
# Helper Functions
# =============================================================================

async def _find_molecule_ids(
    async_config: AsyncDatabaseConfig,
    molecule_name: str | None,
    inchi_key: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Find molecules by name or InChIKey."""
    params: list[Any] = []
    where_parts: list[str] = []

    if inchi_key:
        params.append(inchi_key)
        where_parts.append("m.inchi_key = $1")

    if molecule_name:
        name_param = molecule_name.lower()
        params.append(name_param)
        name_param_idx = len(params)
        params.append(f"%{molecule_name}%")
        pattern_param_idx = len(params)
        where_parts.append(
            f"""(
                m.pref_name ILIKE ${pattern_param_idx}
                OR mc.preferred_name ILIKE ${pattern_param_idx}
                OR EXISTS (
                    SELECT 1 FROM dm_molecule_synonyms s
                    WHERE s.mol_id = m.mol_id AND s.synonym_lower % ${name_param_idx}
                )
            )"""
        )

    if not where_parts:
        return []

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
    return await async_config.execute_query(sql, *params)


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
    
    params.extend([limit, offset])
    
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
        FROM map_ctgov_molecules map
        JOIN rag_study_search rs ON rs.nct_id = map.nct_id
        LEFT JOIN dm_molecule m ON map.mol_id = m.mol_id
        LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
        LIMIT ${param_idx + 1} OFFSET ${param_idx + 2}
    """
    
    return await async_config.execute_query(sql, *params)


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
            )
            if not mol_rows:
                return MoleculeTrialSearchOutput(
                    status="not_found",
                    mode=search_input.mode,
                    query_summary=search_input.molecule_name or search_input.inchi_key or "",
                )

            mol_ids = [row["mol_id"] for row in mol_rows if row.get("mol_id") is not None]
            concept_ids = [row["concept_id"] for row in mol_rows if row.get("concept_id") is not None]
            params: list[Any] = [mol_ids, concept_ids, search_input.limit, search_input.offset]

            sql = """
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.mol_id, map.concept_id,
                       mc.preferred_name AS concept_name, m.pref_name AS molecule_name,
                       m.inchi_key, m.canonical_smiles
                FROM map_ctgov_molecules map
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                LEFT JOIN dm_molecule m ON map.mol_id = m.mol_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE (map.mol_id = ANY($1) OR map.concept_id = ANY($2))
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT $3 OFFSET $4
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [TrialByMoleculeHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                match_type=row.get("match_type"),
                confidence=row.get("confidence"),
                mol_id=row.get("mol_id"),
                concept_id=row.get("concept_id"),
                concept_name=row.get("concept_name"),
                molecule_name=row.get("molecule_name"),
                inchi_key=row.get("inchi_key"),
                canonical_smiles=row.get("canonical_smiles"),
            ) for row in rows]
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
            condition_norm = condition.lower()
            tsquery = _sanitize_tsquery(condition)
            params: list[Any] = [condition_norm, f"%{condition}%", tsquery, search_input.limit, search_input.offset]
            sql = """
                WITH matched_trials AS (
                    SELECT nct_id
                    FROM rag_study_search
                    WHERE conditions_norm % $1
                       OR conditions_norm ILIKE $2
                       OR terms_tsv @@ to_tsquery('english', $3)
                )
                SELECT mc.concept_id, mc.preferred_name AS concept_name,
                       COUNT(DISTINCT map.nct_id)::int AS n_trials
                FROM map_ctgov_molecules map
                JOIN matched_trials mt ON mt.nct_id = map.nct_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                GROUP BY mc.concept_id, mc.preferred_name
                ORDER BY n_trials DESC, mc.preferred_name
                LIMIT $4 OFFSET $5
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
        if search_input.mode == "trials_by_target":
            target_gene = (search_input.target_gene or "").upper()
            params: list[Any] = [target_gene, search_input.min_pchembl]
            phase_filter = ""
            if search_input.phase:
                params.append(search_input.phase)
                phase_filter = f"AND rs.phase = ANY(${len(params)}::text[])"

            params.extend([search_input.limit, search_input.offset])
            sql = f"""
                WITH target_concepts AS (
                    SELECT DISTINCT concept_id
                    FROM dm_compound_target_activity
                    WHERE gene_symbol = $1
                      AND (pchembl_value IS NULL OR pchembl_value >= $2)
                      AND concept_id IS NOT NULL
                )
                SELECT rs.nct_id, rs.brief_title, rs.phase, rs.overall_status,
                       map.match_type, map.confidence, map.concept_id,
                       mc.preferred_name AS concept_name
                FROM target_concepts tc
                JOIN map_ctgov_molecules map ON map.concept_id = tc.concept_id
                JOIN rag_study_search rs ON rs.nct_id = map.nct_id
                LEFT JOIN dm_molecule_concept mc ON map.concept_id = mc.concept_id
                WHERE 1=1 {phase_filter}
                ORDER BY map.confidence DESC NULLS LAST, rs.nct_id
                LIMIT ${len(params)-1} OFFSET ${len(params)}
            """
            rows = await async_config.execute_query(sql, *params)
            hits = [TrialByTargetHit(
                nct_id=row["nct_id"],
                brief_title=row.get("brief_title"),
                phase=row.get("phase"),
                status=row.get("overall_status"),
                match_type=row.get("match_type"),
                confidence=row.get("confidence"),
                concept_id=row.get("concept_id"),
                concept_name=row.get("concept_name"),
            ) for row in rows]
            status = "success" if hits else "not_found"
            return MoleculeTrialSearchOutput(
                status=status,
                mode=search_input.mode,
                total_hits=len(hits),
                hits=hits,
                query_summary=search_input.target_gene or "",
            )

        # =====================================================================
        # MODE: trials_by_structure (similarity search)
        # =====================================================================
        if search_input.mode == "trials_by_structure":
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
                hits.append(TrialByMoleculeHit(
                    nct_id=row["nct_id"],
                    brief_title=row.get("brief_title"),
                    phase=row.get("phase"),
                    status=row.get("overall_status"),
                    match_type=row.get("match_type"),
                    confidence=row.get("confidence"),
                    mol_id=row.get("mol_id"),
                    concept_id=concept_id,
                    concept_name=row.get("concept_name"),
                    molecule_name=row.get("molecule_name"),
                    inchi_key=row.get("inchi_key"),
                    canonical_smiles=row.get("canonical_smiles"),
                    similarity_score=similarity_lookup.get(concept_id),
                    structure_match_type="similarity",
                ))
            
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
            )
            
            # Count trials per molecule
            trial_counts: dict[int, int] = {}
            for row in trial_rows:
                cid = row.get("concept_id")
                if cid:
                    trial_counts[cid] = trial_counts.get(cid, 0) + 1
            
            # Build hits
            hits = [
                TrialByMoleculeHit(
                    nct_id=row["nct_id"],
                    brief_title=row.get("brief_title"),
                    phase=row.get("phase"),
                    status=row.get("overall_status"),
                    match_type=row.get("match_type"),
                    confidence=row.get("confidence"),
                    mol_id=row.get("mol_id"),
                    concept_id=row.get("concept_id"),
                    concept_name=row.get("concept_name"),
                    molecule_name=row.get("molecule_name"),
                    inchi_key=row.get("inchi_key"),
                    canonical_smiles=row.get("canonical_smiles"),
                    structure_match_type="substructure",
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