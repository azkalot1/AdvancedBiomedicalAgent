#!/usr/bin/env python3
# agent_target_search.py
"""
Agent-facing pharmacology search tools with proper formatting.

This module provides LLM-friendly tools for searching drug-target interactions,
mechanisms of action, clinical trials, and molecular similarity.
"""

from langchain_core.tools import tool
from bioagent.data.ingest.config import DEFAULT_CONFIG
from .tool_utils import robust_unwrap_llm_inputs

from bioagent.data.search.target_search import (
    PharmacologySearch,
    TargetSearchInput,
    TargetSearchOutput,
    SearchMode,
    DataSource,
    CompoundTargetProfile,
    DrugForTargetHit,
    DrugProfileResult,
    ClinicalTrialHit,
    MoleculeForm,
)


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def _get_potency_label(pchembl: float | None) -> str:
    """Get human-readable potency label from pChEMBL value."""
    if pchembl is None:
        return "unknown"
    if pchembl >= 9:
        return "very potent (<1nM)"
    elif pchembl >= 8:
        return "potent (1-10nM)"
    elif pchembl >= 7:
        return "good (10-100nM)"
    elif pchembl >= 6:
        return "moderate (100nM-1μM)"
    elif pchembl >= 5:
        return "weak (1-10μM)"
    else:
        return "very weak (>10μM)"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return text[:max_len] + "..." if len(text) > max_len else text


def _format_compound_profile(rank: int, hit: CompoundTargetProfile) -> list[str]:
    """Format a compound target profile for agent output."""
    lines = []
    
    # Header
    header_parts = [f"[{rank}] {hit.concept_name}"]
    if hit.tanimoto_similarity:
        header_parts.append(f"(similarity: {hit.tanimoto_similarity:.2f})")
    lines.append(" ".join(header_parts))
    
    # Identifiers
    if hit.chembl_id:
        lines.append(f"    ChEMBL: {hit.chembl_id}")
    if hit.best_pchembl:
        lines.append(f"    Best potency: pChEMBL {hit.best_pchembl:.1f} ({_get_potency_label(hit.best_pchembl)})")
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    # Mechanisms of Action (curated)
    if hit.mechanisms:
        lines.append("")
        lines.append(f"    MECHANISMS OF ACTION ({len(hit.mechanisms)}):")
        for m in hit.mechanisms:
            genes = ", ".join(m.gene_symbols) if m.gene_symbols else "unknown target"
            lines.append(f"      • {m.action_type}: {m.mechanism_of_action}")
            lines.append(f"        Target: {m.target_name or 'N/A'} ({genes})")
            
            # Flags
            flags = []
            if m.disease_efficacy:
                flags.append("efficacy established")
            if m.direct_interaction:
                flags.append("direct interaction")
            if flags:
                lines.append(f"        [{', '.join(flags)}]")
    
    # Activity Data (quantitative)
    if hit.activities:
        lines.append("")
        lines.append(f"    ACTIVITY DATA ({len(hit.activities)} target(s)):")
        for a in hit.activities:
            lines.append(f"      • {a.gene_symbol}: {a.activity_type} = {a.activity_value_nm:.1f} nM")
            lines.append(f"        pChEMBL: {a.pchembl:.2f} ({_get_potency_label(a.pchembl)})" if a.pchembl else "        pChEMBL: N/A")
            if a.target_name:
                lines.append(f"        Target: {a.target_name}")
            lines.append(f"        Confidence: {a.data_confidence or 'N/A'}, measurements: {a.n_measurements}")
    
    # No data message
    if not hit.mechanisms and not hit.activities:
        lines.append("")
        lines.append("    No target or mechanism data available for this compound.")
    
    return lines


def _format_drug_for_target(rank: int, hit: DrugForTargetHit) -> list[str]:
    """Format a drug-for-target hit."""
    lines = []
    
    # Header
    lines.append(f"[{rank}] {hit.concept_name}")
    
    # Identifiers
    if hit.chembl_id:
        lines.append(f"    ChEMBL: {hit.chembl_id}")
    
    # Activity data
    if hit.activity_value_nm is not None:
        lines.append(f"    Activity: {hit.activity_type} = {hit.activity_value_nm:.1f} nM")
        if hit.pchembl:
            lines.append(f"    pChEMBL: {hit.pchembl:.2f} ({_get_potency_label(hit.pchembl)})")
        lines.append(f"    Confidence: {hit.data_confidence or 'N/A'}, measurements: {hit.n_measurements}")
    
    # Mechanism data
    if hit.mechanism_of_action:
        lines.append(f"    Mechanism: {hit.action_type} - {hit.mechanism_of_action}")
    
    # Selectivity
    if hit.selectivity_fold:
        lines.append(f"    Selectivity: {hit.selectivity_fold:.1f}x over off-targets")
    
    # SMILES
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    return lines


def _format_drug_profile_result(rank: int, hit: DrugProfileResult) -> list[str]:
    """Format a drug profile result."""
    lines = []
    
    lines.append(f"[{rank}] {hit.concept_name}")
    lines.append(f"    Concept ID: {hit.concept_id}")
    
    # Summary stats
    salt_info = " (includes salts)" if hit.has_salt_forms else ""
    lines.append(f"    Molecular forms: {hit.n_forms}{salt_info}")
    lines.append(f"    Activity targets: {hit.n_activity_targets}")
    lines.append(f"    Mechanism targets: {hit.n_mechanism_targets}")
    lines.append(f"    Clinical trials: {hit.n_clinical_trials}")
    
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    # Target profile details
    if hit.target_profile:
        lines.append("")
        lines.append("    --- TARGET PROFILE ---")
        profile_lines = _format_compound_profile(0, hit.target_profile)
        # Indent and skip the header line
        for line in profile_lines[1:]:
            lines.append(f"    {line.strip()}")
    
    # Recent trials
    if hit.recent_trials:
        lines.append("")
        lines.append(f"    --- CLINICAL TRIALS ({len(hit.recent_trials)}) ---")
        for t in hit.recent_trials[:5]:
            lines.append(f"      • {t.nct_id} | Phase: {t.phase or 'N/A'} | Status: {t.trial_status or 'Unknown'}")
            lines.append(f"        {_truncate(t.trial_title, 70)}")
        if len(hit.recent_trials) > 5:
            lines.append(f"      ... and {len(hit.recent_trials) - 5} more trials")
    
    # Molecular forms
    if hit.forms:
        lines.append("")
        lines.append(f"    --- MOLECULAR FORMS ({len(hit.forms)}) ---")
        for f in hit.forms[:5]:
            name = f.form_name or f.chembl_id or f"MOL_{f.mol_id}"
            salt = f" ({f.salt_form})" if f.salt_form else ""
            stereo = f" [{f.stereo_type}]" if f.stereo_type else ""
            lines.append(f"      • {name}{salt}{stereo}")
        if len(hit.forms) > 5:
            lines.append(f"      ... and {len(hit.forms) - 5} more forms")
    
    return lines


def _format_clinical_trial(rank: int, hit: ClinicalTrialHit) -> list[str]:
    """Format a clinical trial hit."""
    lines = []
    
    lines.append(f"[{rank}] {hit.nct_id}")
    lines.append(f"    Phase: {hit.phase or 'N/A'}")
    lines.append(f"    Status: {hit.trial_status or 'Unknown'}")
    lines.append(f"    Title: {_truncate(hit.trial_title, 100)}")
    
    if hit.concept_name:
        lines.append(f"    Drug: {hit.concept_name}")
    if hit.molecule_form:
        lines.append(f"    Form: {hit.molecule_form}")
    if hit.match_type:
        conf = f", confidence: {hit.confidence:.2f}" if hit.confidence else ""
        lines.append(f"    Match type: {hit.match_type}{conf}")
    
    return lines


def _format_molecule_form(rank: int, hit: MoleculeForm) -> list[str]:
    """Format a molecule form."""
    lines = []
    
    name = hit.form_name or hit.chembl_id or f"MOL_{hit.mol_id}"
    lines.append(f"[{rank}] {name}")
    
    if hit.chembl_id and hit.chembl_id != name:
        lines.append(f"    ChEMBL: {hit.chembl_id}")
    if hit.is_salt:
        lines.append(f"    Salt form: {hit.salt_form or 'Yes'}")
    if hit.stereo_type:
        lines.append(f"    Stereochemistry: {hit.stereo_type}")
    if hit.inchi_key:
        lines.append(f"    InChIKey: {hit.inchi_key}")
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    return lines


def _format_pharmacology_output(result: TargetSearchOutput) -> str:
    """Format pharmacology search results for LLM consumption."""
    lines = []
    
    # Status header
    if result.status == "success":
        lines.append(f"✓ Search: {result.query_summary}")
        lines.append(f"  Found {result.total_hits} result(s)")
    elif result.status == "not_found":
        lines.append(f"○ Search: {result.query_summary}")
        lines.append(f"  No results found")
    elif result.status == "error":
        lines.append(f"✗ Search failed: {result.query_summary}")
        lines.append(f"  Error: {result.error}")
    elif result.status == "invalid_input":
        lines.append(f"✗ Invalid input")
        lines.append(f"  Error: {result.error}")
    
    # Timing
    if result.execution_time_ms:
        lines.append(f"  (Query time: {result.execution_time_ms:.0f}ms)")
    
    # Warnings
    if result.warnings:
        lines.append("")
        for w in result.warnings:
            lines.append(f"⚠ {w}")
    
    # Early return for errors
    if result.status in ["error", "invalid_input"]:
        lines.append("")
        lines.append("Suggestions:")
        lines.append("  • Check input parameters and spelling")
        lines.append("  • Verify drug/gene names are correct")
        lines.append("  • For SMILES input, ensure the structure is valid")
        return "\n".join(lines)
    
    # Not found suggestions
    if result.status == "not_found":
        lines.append("")
        lines.append("Suggestions:")
        if result.mode in [SearchMode.TARGETS_FOR_DRUG, SearchMode.DRUG_PROFILE, 
                          SearchMode.DRUG_FORMS, SearchMode.TRIALS_FOR_DRUG]:
            lines.append("  • Check drug name spelling")
            lines.append("  • Try generic name instead of brand name")
            lines.append("  • Try alternative synonyms")
        elif result.mode == SearchMode.DRUGS_FOR_TARGET:
            lines.append("  • Verify gene symbol (use official HGNC symbol)")
            lines.append("  • Try lowering min_pchembl threshold (e.g., 5.0)")
        elif result.mode in [SearchMode.SIMILAR_MOLECULES, SearchMode.EXACT_STRUCTURE]:
            lines.append("  • Verify SMILES string is valid")
            lines.append("  • Try lowering similarity_threshold")
        return "\n".join(lines)
    
    # No hits
    if not result.hits:
        return "\n".join(lines)
    
    lines.append("")
    lines.append("=" * 60)
    
    # Format each hit based on type
    for i, hit in enumerate(result.hits[:30], 1):  # Limit to 30 for readability
        lines.append("")
        
        if isinstance(hit, CompoundTargetProfile):
            lines.extend(_format_compound_profile(i, hit))
        elif isinstance(hit, DrugForTargetHit):
            lines.extend(_format_drug_for_target(i, hit))
        elif isinstance(hit, DrugProfileResult):
            lines.extend(_format_drug_profile_result(i, hit))
        elif isinstance(hit, ClinicalTrialHit):
            lines.extend(_format_clinical_trial(i, hit))
        elif isinstance(hit, MoleculeForm):
            lines.extend(_format_molecule_form(i, hit))
        else:
            # Fallback for unknown types
            lines.append(f"[{i}] {type(hit).__name__}: {hit}")
    
    # Truncation notice
    if result.total_hits > 30:
        lines.append("")
        lines.append(f"... showing 30 of {result.total_hits} results. Use 'limit' parameter for more.")
    
    return "\n".join(lines)


# =============================================================================
# INPUT VALIDATION HELPERS  
# =============================================================================

def _validate_drug_name(drug_name: str | None) -> tuple[str | None, str | None]:
    """Validate drug name input. Returns (cleaned_name, error_message)."""
    if not drug_name:
        return None, "drug_name is required. Example: search_drug_targets(drug_name='imatinib')"
    
    drug_name = drug_name.strip()
    
    if len(drug_name) < 2:
        return None, f"drug_name '{drug_name}' is too short. Please provide a valid drug name."
    
    if len(drug_name) > 200:
        return None, f"drug_name is too long ({len(drug_name)} chars). Maximum is 200 characters."
    
    return drug_name, None


def _validate_gene_symbol(gene_symbol: str | None) -> tuple[str | None, str | None]:
    """Validate gene symbol input. Returns (cleaned_symbol, error_message)."""
    if not gene_symbol:
        return None, "gene_symbol is required. Example: search_target_drugs(gene_symbol='EGFR')"
    
    gene_symbol = gene_symbol.strip().upper()
    
    if len(gene_symbol) < 2:
        return None, f"gene_symbol '{gene_symbol}' is too short. Use official HGNC symbols (e.g., 'EGFR', 'ABL1')."
    
    if len(gene_symbol) > 20:
        return None, f"gene_symbol '{gene_symbol}' is too long. Use official HGNC symbols."
    
    # Basic pattern check
    if not gene_symbol.replace("-", "").replace("_", "").isalnum():
        return None, f"gene_symbol '{gene_symbol}' contains invalid characters. Use official HGNC symbols."
    
    return gene_symbol, None


def _validate_smiles(smiles: str | None) -> tuple[str | None, str | None]:
    """Validate SMILES input. Returns (cleaned_smiles, error_message)."""
    if not smiles:
        return None, "smiles is required. Provide a valid SMILES string."
    
    smiles = smiles.strip()
    
    if len(smiles) < 1:
        return None, "smiles cannot be empty."
    
    if len(smiles) > 5000:
        return None, f"smiles is too long ({len(smiles)} chars). Maximum is 5000 characters."
    
    # Basic character check (SMILES alphabet)
    valid_chars = set("CNOPSFIBrcnopsfib[]()=#@+-\\/%.0123456789")
    invalid = set(smiles) - valid_chars
    if invalid:
        return None, f"smiles contains invalid characters: {invalid}. Check SMILES syntax."
    
    return smiles, None


def _validate_data_source(data_source: str) -> tuple[DataSource | None, str | None]:
    """Validate data_source parameter. Returns (DataSource, error_message)."""
    valid_sources = {
        "both": DataSource.BOTH,
        "activity": DataSource.ACTIVITY,
        "mechanism": DataSource.MECHANISM,
    }
    
    ds = data_source.lower().strip()
    
    if ds not in valid_sources:
        return None, (
            f"Invalid data_source '{data_source}'. "
            f"Valid options: 'both', 'activity', 'mechanism'\n"
            "  • 'both' (default): Activity data + curated mechanisms\n"
            "  • 'activity': Only quantitative assay data (IC50, Ki, etc.)\n"
            "  • 'mechanism': Only curated mechanism of action data"
        )
    
    return valid_sources[ds], None


def _validate_pchembl(min_pchembl: float) -> tuple[float | None, str | None]:
    """Validate min_pchembl parameter. Returns (value, error_message)."""
    if not isinstance(min_pchembl, (int, float)):
        return None, f"min_pchembl must be a number, got {type(min_pchembl).__name__}"
    
    if min_pchembl < 0 or min_pchembl > 15:
        return None, (
            f"min_pchembl={min_pchembl} is out of range (0.0 to 15.0).\n"
            "Common values:\n"
            "  • 5.0 = 10 μM (weak, more results)\n"
            "  • 6.0 = 1 μM (moderate)\n"
            "  • 7.0 = 100 nM (good)\n"
            "  • 8.0 = 10 nM (potent, fewer results)"
        )
    
    return float(min_pchembl), None


def _validate_similarity(threshold: float) -> tuple[float | None, str | None]:
    """Validate similarity_threshold parameter. Returns (value, error_message)."""
    if not isinstance(threshold, (int, float)):
        return None, f"similarity_threshold must be a number, got {type(threshold).__name__}"
    
    if threshold < 0.0 or threshold > 1.0:
        return None, (
            f"similarity_threshold={threshold} is out of range (0.0 to 1.0).\n"
            "Common values:\n"
            "  • 0.5 = distant analogs\n"
            "  • 0.7 = similar scaffold (default)\n"
            "  • 0.85 = close analogs\n"
            "  • 0.95 = very similar"
        )
    
    return float(threshold), None


def _validate_limit(limit: int) -> tuple[int | None, str | None]:
    """Validate limit parameter. Returns (value, error_message)."""
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            return None, f"limit must be an integer, got {type(limit).__name__}"
    
    if limit < 1:
        return None, "limit must be at least 1"
    
    if limit > 500:
        return 500, None  # Cap at 500, don't error
    
    return limit, None


# =============================================================================
# PHARMACOLOGY TOOLS
# =============================================================================

@tool("search_drug_targets", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_targets(
    drug_name: str,
    min_pchembl: float = 5.0,
    data_source: str = "both",
    include_all_organisms: bool = False,
    limit: int = 50,
) -> str:
    """
    Find all known protein targets for a drug.
    
    Returns target proteins that the drug binds to or inhibits, with 
    activity measurements (IC50, Ki, Kd, EC50) from ChEMBL and curated
    mechanism of action data from DrugCentral.
    
    Args:
        drug_name: Name of the drug to search (brand name, generic, or synonym).
            Examples: "imatinib", "Gleevec", "aspirin", "trastuzumab"
        min_pchembl: Minimum pChEMBL value (potency threshold). 
            - 5.0 = 10 μM (weak, default - more results)
            - 6.0 = 1 μM (moderate)
            - 7.0 = 100 nM (good)
            - 8.0 = 10 nM (potent - fewer results)
        data_source: Type of target data to return.
            - "both" (default): Activity data + curated mechanisms
            - "activity": Only quantitative assay data (IC50, Ki, etc.)
            - "mechanism": Only curated mechanism of action data
        include_all_organisms: If True, include targets from all species.
            Default (False) returns only human targets.
        limit: Maximum number of results (default: 50, max: 500).
    
    Returns:
        Formatted list of targets with gene symbols, activity values, mechanisms,
        and data quality indicators.
    
    Examples:
        search_drug_targets("imatinib")
        search_drug_targets("aspirin", min_pchembl=6.0)
        search_drug_targets("diazepam", data_source="mechanism")
    """
    # Validate inputs
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    ds, error = _validate_data_source(data_source)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query=drug_name,
            min_pchembl=min_pchembl,
            data_source=ds,
            include_all_organisms=include_all_organisms,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error searching drug targets: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}', min_pchembl={min_pchembl}, data_source='{data_source}'\n\n"
            "Suggestions:\n"
            "  • Check drug name spelling\n"
            "  • Try generic name instead of brand name\n"
            "  • For biologics (antibodies), try the INN name"
        )


@tool("search_target_drugs", return_direct=False)
@robust_unwrap_llm_inputs
async def search_target_drugs(
    gene_symbol: str,
    min_pchembl: float = 5.0,
    data_source: str = "both",
    limit: int = 50,
) -> str:
    """
    Find all drugs/compounds that modulate a specific protein target.
    
    Use this to discover inhibitors, agonists, or modulators of a gene/protein.
    Returns both quantitative activity data and curated mechanism annotations.
    
    Args:
        gene_symbol: Gene symbol of the target protein (HGNC official symbol).
            Examples: "EGFR", "ABL1", "JAK2", "BRAF", "CDK4", "ROS1", "NTRK1"
        min_pchembl: Minimum pChEMBL value (potency threshold).
            - 5.0 = 10 μM (weak, default - more results)
            - 6.0 = 1 μM (moderate)  
            - 7.0 = 100 nM (good)
            - 8.0 = 10 nM (potent - fewer results)
        data_source: Type of data to return.
            - "both" (default): Activity + mechanism data
            - "activity": Only quantitative assay data
            - "mechanism": Only curated mechanisms
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        Formatted list of drugs/compounds with activity data and/or mechanisms.
    
    Examples:
        search_target_drugs("EGFR")
        search_target_drugs("JAK2", min_pchembl=7.0)
        search_target_drugs("ROS1", data_source="mechanism")
    """
    # Validate inputs
    gene_symbol, error = _validate_gene_symbol(gene_symbol)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    ds, error = _validate_data_source(data_source)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_TARGET,
            query=gene_symbol,
            min_pchembl=min_pchembl,
            data_source=ds,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error searching target drugs: {type(e).__name__}: {e}\n\n"
            f"Input: gene_symbol='{gene_symbol}', min_pchembl={min_pchembl}\n\n"
            "Suggestions:\n"
            "  • Verify gene symbol (use official HGNC symbol)\n"
            "  • Common symbols: EGFR, BRAF, ABL1, JAK2, CDK4, ERBB2\n"
            "  • Try lowering min_pchembl to get more results"
        )


@tool("search_similar_molecules", return_direct=False)
@robust_unwrap_llm_inputs
async def search_similar_molecules(
    smiles: str,
    similarity_threshold: float = 0.7,
    min_pchembl: float = 5.0,
    limit: int = 50,
) -> str:
    """
    Find molecules structurally similar to a query compound.
    
    Uses Tanimoto similarity on Morgan fingerprints to find analogs.
    Useful for finding related compounds or understanding structure-activity relationships.
    
    Args:
        smiles: SMILES string of the query molecule.
            Example: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" (imatinib)
        similarity_threshold: Minimum Tanimoto similarity (0.0 to 1.0).
            - 0.5 = distant analogs (many results)
            - 0.7 = similar scaffold (default)
            - 0.85 = close analogs
            - 0.95 = very similar (stereoisomers, salts)
        min_pchembl: Minimum potency for activity data (default: 5.0).
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        List of similar molecules with similarity scores and activity data.
    
    Examples:
        search_similar_molecules("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin analogs
        search_similar_molecules(imatinib_smiles, similarity_threshold=0.85)
    """
    # Validate inputs
    smiles, error = _validate_smiles(smiles)
    if error:
        return f"✗ Input error: {error}"
    
    similarity_threshold, error = _validate_similarity(similarity_threshold)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.SIMILAR_MOLECULES,
            smiles=smiles,
            similarity_threshold=similarity_threshold,
            min_pchembl=min_pchembl,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        smiles_preview = _truncate(smiles, 50)
        return (
            f"✗ Error searching similar molecules: {type(e).__name__}: {e}\n\n"
            f"Input SMILES: {smiles_preview}\n\n"
            "Suggestions:\n"
            "  • Verify the SMILES string is valid\n"
            "  • Use a SMILES validator or chemical drawing tool\n"
            "  • Try lowering similarity_threshold for more results"
        )


@tool("search_exact_structure", return_direct=False)
@robust_unwrap_llm_inputs
async def search_exact_structure(
    smiles: str,
    min_pchembl: float = 5.0,
) -> str:
    """
    Find an exact structure match for a molecule and retrieve its data.
    
    Use this when you have a SMILES and want to identify the compound
    and retrieve its target activity and mechanism data.
    
    Args:
        smiles: SMILES string of the molecule to identify.
        min_pchembl: Minimum potency for activity data (default: 5.0).
    
    Returns:
        Compound identification with activity and mechanism data if found.
    
    Examples:
        search_exact_structure("CC(=O)Oc1ccccc1C(=O)O")  # Identify aspirin
        search_exact_structure("Cc1nc(CNC(=O)NC2CCN(c3ncccc3Cl)C2)oc1C")
    """
    # Validate inputs
    smiles, error = _validate_smiles(smiles)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.EXACT_STRUCTURE,
            smiles=smiles,
            min_pchembl=min_pchembl,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        smiles_preview = _truncate(smiles, 50)
        return (
            f"✗ Error in exact structure search: {type(e).__name__}: {e}\n\n"
            f"Input SMILES: {smiles_preview}\n\n"
            "Suggestions:\n"
            "  • Verify the SMILES string is valid\n"
            "  • Try search_similar_molecules with high threshold instead"
        )


@tool("search_substructure", return_direct=False)
@robust_unwrap_llm_inputs
async def search_substructure(
    pattern: str,
    limit: int = 50,
) -> str:
    """
    Find molecules containing a specific substructure.
    
    Use SMILES or SMARTS patterns to find compounds with specific 
    chemical features (e.g., a particular ring system, functional group).
    
    Note: This search can be slow for very common substructures (e.g., benzene).
    
    Args:
        pattern: SMILES or SMARTS pattern to search for.
            Examples:
            - "c1ccccc1" (benzene ring - very common, slow)
            - "c1ccc2[nH]ccc2c1" (indole)
            - "C(F)(F)F" (trifluoromethyl)
            - "C(=O)N" (amide)
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        List of molecules containing the substructure.
    
    Examples:
        search_substructure("c1ccc2[nH]ccc2c1")  # Indole compounds
        search_substructure("C(F)(F)F")  # CF3-containing compounds
    """
    # Validate pattern (similar to SMILES validation)
    if not pattern or not pattern.strip():
        return "✗ Input error: pattern is required. Provide a SMILES or SMARTS pattern."
    
    pattern = pattern.strip()
    
    if len(pattern) < 1:
        return "✗ Input error: pattern cannot be empty."
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.SUBSTRUCTURE,
            smarts=pattern,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error in substructure search: {type(e).__name__}: {e}\n\n"
            f"Input pattern: {pattern}\n\n"
            "Suggestions:\n"
            "  • Check SMILES/SMARTS syntax\n"
            "  • Very common substructures (benzene) may timeout\n"
            "  • Try a more specific pattern"
        )


@tool("get_drug_profile", return_direct=False)
@robust_unwrap_llm_inputs
async def get_drug_profile(
    drug_name: str,
    include_trials: bool = True,
    include_forms: bool = True,
    min_pchembl: float = 5.0,
) -> str:
    """
    Get a comprehensive profile for a drug.
    
    Returns aggregated information including:
    - All molecular forms (salts, stereoisomers)
    - Known protein targets with activity data
    - Curated mechanism of action
    - Associated clinical trials
    - Chemical identifiers (ChEMBL ID, SMILES)
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
        include_trials: Include clinical trial associations (default: True).
        include_forms: Include all salt/stereo forms (default: True).
        min_pchembl: Minimum potency for target data (default: 5.0).
    
    Returns:
        Comprehensive drug profile with all available information.
    
    Examples:
        get_drug_profile("imatinib")
        get_drug_profile("aspirin", include_trials=False)
    """
    # Validate inputs
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUG_PROFILE,
            query=drug_name,
            include_trials=include_trials,
            include_forms=include_forms,
            min_pchembl=min_pchembl,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error getting drug profile: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}'\n\n"
            "Suggestions:\n"
            "  • Check drug name spelling\n"
            "  • Try alternative names or synonyms"
        )


@tool("get_drug_forms", return_direct=False)
@robust_unwrap_llm_inputs
async def get_drug_forms(
    drug_name: str,
    limit: int = 50,
) -> str:
    """
    Get all molecular forms of a drug (salts, stereoisomers, etc.).
    
    Use this to see all registered forms of a drug substance,
    including different salt forms and stereochemical variants.
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
        limit: Maximum results (default: 50).
    
    Returns:
        List of all molecular forms with identifiers and SMILES.
    
    Examples:
        get_drug_forms("imatinib")  # Shows imatinib mesylate, etc.
        get_drug_forms("metformin")  # Shows metformin HCl, etc.
    """
    # Validate inputs
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUG_FORMS,
            query=drug_name,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return f"✗ Error getting drug forms: {type(e).__name__}: {e}"


@tool("search_drug_trials", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_trials(
    drug_name: str,
    limit: int = 50,
) -> str:
    """
    Find clinical trials for a drug.
    
    Searches for trials involving the specified drug from ClinicalTrials.gov.
    Works for both small molecules and biologics.
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
            Examples: "imatinib", "pembrolizumab", "Keytruda"
        limit: Maximum results (default: 50).
    
    Returns:
        List of clinical trials with NCT IDs, titles, phases, and status.
    
    Examples:
        search_drug_trials("imatinib")
        search_drug_trials("pembrolizumab")
    """
    # Validate inputs
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.TRIALS_FOR_DRUG,
            query=drug_name,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error searching drug trials: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}'\n\n"
            "Suggestions:\n"
            "  • Check drug name spelling\n"
            "  • For biologics, try the INN name (e.g., 'pembrolizumab' not 'Keytruda')"
        )


@tool("compare_drugs_on_target", return_direct=False)
@robust_unwrap_llm_inputs
async def compare_drugs_on_target(
    target: str,
    drug_names: list[str] | str,
) -> str:
    """
    Compare multiple drugs' activity against a single target.
    
    Useful for understanding relative potency of different drugs
    against the same protein target.
    
    Args:
        target: Gene symbol of the target (e.g., "EGFR", "ABL1").
        drug_names: List of drug names to compare.
            Examples: ["imatinib", "dasatinib", "nilotinib"]
    
    Returns:
        Comparison table showing relative potency and fold-differences.
    
    Examples:
        compare_drugs_on_target("ABL1", ["imatinib", "dasatinib", "nilotinib"])
        compare_drugs_on_target("EGFR", ["erlotinib", "gefitinib", "osimertinib"])
    """
    # Validate target
    target, error = _validate_gene_symbol(target)
    if error:
        return f"✗ Input error (target): {error}"
    
    # Normalize drug_names to list
    if isinstance(drug_names, str):
        drug_names = [drug_names]
    
    if not drug_names or len(drug_names) == 0:
        return "✗ Input error: drug_names is required. Provide a list of drug names to compare."
    
    if len(drug_names) < 2:
        return "✗ Input error: Provide at least 2 drug names to compare."
    
    if len(drug_names) > 20:
        return "✗ Input error: Too many drugs. Maximum is 20 for comparison."
    
    # Validate each drug name
    validated_names = []
    for name in drug_names:
        clean_name, error = _validate_drug_name(name)
        if error:
            return f"✗ Input error (drug '{name}'): {error}"
        validated_names.append(clean_name)
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.COMPARE_DRUGS,
            target=target,
            drug_names=validated_names,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error comparing drugs: {type(e).__name__}: {e}\n\n"
            f"Input: target='{target}', drugs={validated_names}"
        )


@tool("search_selective_drugs", return_direct=False)
@robust_unwrap_llm_inputs
async def search_selective_drugs(
    target: str,
    off_targets: list[str] | str,
    min_selectivity_fold: float = 10.0,
    min_pchembl: float = 6.0,
    limit: int = 50,
) -> str:
    """
    Find drugs that are selective for one target over others.
    
    Useful for finding compounds that hit a desired target but 
    spare related proteins (e.g., selective kinase inhibitors).
    
    Args:
        target: Primary target gene symbol (the desired target).
            Example: "JAK2"
        off_targets: Gene symbols of targets to avoid.
            Example: ["JAK1", "JAK3"] to find JAK2-selective compounds
        min_selectivity_fold: Minimum fold-selectivity required (default: 10x).
            A value of 10 means the drug must be 10x more potent on 
            the primary target vs off-targets.
        min_pchembl: Minimum potency on primary target (default: 6.0 = 1 μM).
        limit: Maximum results (default: 50).
    
    Returns:
        List of selective compounds with selectivity ratios.
    
    Examples:
        search_selective_drugs("JAK2", ["JAK1", "JAK3"])
        search_selective_drugs("CDK4", ["CDK6"], min_selectivity_fold=100)
        search_selective_drugs("EGFR", ["ERBB2", "ERBB4"])
    """
    # Validate target
    target, error = _validate_gene_symbol(target)
    if error:
        return f"✗ Input error (target): {error}"
    
    # Normalize off_targets to list
    if isinstance(off_targets, str):
        off_targets = [off_targets]
    
    if not off_targets or len(off_targets) == 0:
        return (
            "✗ Input error: off_targets is required.\n"
            "Provide gene symbols of targets to avoid.\n"
            "Example: search_selective_drugs('JAK2', ['JAK1', 'JAK3'])"
        )
    
    if len(off_targets) > 10:
        return "✗ Input error: Too many off-targets. Maximum is 10."
    
    # Validate each off-target
    validated_off_targets = []
    for ot in off_targets:
        clean_ot, error = _validate_gene_symbol(ot)
        if error:
            return f"✗ Input error (off_target '{ot}'): {error}"
        validated_off_targets.append(clean_ot)
    
    # Validate numeric params
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    if not isinstance(min_selectivity_fold, (int, float)) or min_selectivity_fold < 1:
        return "✗ Input error: min_selectivity_fold must be >= 1.0"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.SELECTIVE_DRUGS,
            target=target,
            off_targets=validated_off_targets,
            min_selectivity_fold=float(min_selectivity_fold),
            min_pchembl=min_pchembl,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return (
            f"✗ Error searching selective drugs: {type(e).__name__}: {e}\n\n"
            f"Input: target='{target}', off_targets={validated_off_targets}"
        )