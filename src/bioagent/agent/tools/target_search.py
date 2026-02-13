#!/usr/bin/env python3
# agent_target_search.py
"""
Agent-facing pharmacology search tools with proper formatting.
"""

from langchain_core.tools import tool

from bioagent.data.ingest.config import DEFAULT_CONFIG
from .tool_utils import robust_unwrap_llm_inputs, build_handoff_signals

from bioagent.data.search.target_search import (
    PharmacologySearch,
    TargetSearchInput,
    TargetSearchOutput,
    SearchMode,
    ActivityType,
    DataConfidence,
    DataSource,
    CompoundTargetProfile,
    DrugForTargetHit,
    DrugProfileResult,
    DrugIndicationHit,
    TargetPathwayHit,
    ClinicalTrialHit,
    MoleculeForm,
    SearchDiagnostics,
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


def _format_diagnostics(diagnostics: SearchDiagnostics | None, verbose: bool = False) -> list[str]:
    """Format diagnostics information for output."""
    if not diagnostics:
        return []
    
    lines = []
    
    # Search steps (what was tried)
    if diagnostics.search_steps and verbose:
        lines.append("")
        lines.append("Search steps attempted:")
        for step in diagnostics.search_steps:
            status = "✓" if step.get("found", 0) > 0 else "✗"
            step_name = step.get("step", "unknown").replace("_", " ")
            found = step.get("found", 0)
            lines.append(f"  {status} {step_name}: {found} match(es)")
            if step.get("matches"):
                matches_preview = ", ".join(step["matches"][:3])
                lines.append(f"      → {matches_preview}")
    
    # Suggestions (always show)
    if diagnostics.suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for suggestion in diagnostics.suggestions:
            lines.append(f"  → {suggestion}")
    
    return lines


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
    
    lines.append(f"[{rank}] {hit.concept_name}")
    
    if hit.chembl_id:
        lines.append(f"    ChEMBL: {hit.chembl_id}")
    
    if hit.activity_value_nm is not None:
        lines.append(f"    Activity: {hit.activity_type} = {hit.activity_value_nm:.1f} nM")
        if hit.pchembl:
            lines.append(f"    pChEMBL: {hit.pchembl:.2f} ({_get_potency_label(hit.pchembl)})")
        lines.append(f"    Confidence: {hit.data_confidence or 'N/A'}, measurements: {hit.n_measurements}")
    
    if hit.mechanism_of_action:
        lines.append(f"    Mechanism: {hit.action_type} - {hit.mechanism_of_action}")
    
    if hit.selectivity_fold:
        lines.append(f"    Selectivity: {hit.selectivity_fold:.1f}x over off-targets")
    
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    return lines


def _format_drug_profile_result(
    rank: int,
    hit: DrugProfileResult,
    detail_level: str = "standard",
) -> list[str]:
    """Format a drug profile result."""
    lines = []
    
    lines.append(f"[{rank}] {hit.concept_name}")
    lines.append(f"    Concept ID: {hit.concept_id}")
    
    salt_info = " (includes salts)" if hit.has_salt_forms else ""
    lines.append(f"    Molecular forms: {hit.n_forms}{salt_info}")
    lines.append(f"    Activity targets: {hit.n_activity_targets}")
    lines.append(f"    Mechanism targets: {hit.n_mechanism_targets}")
    lines.append(f"    Clinical trials: {hit.n_clinical_trials}")
    
    if hit.canonical_smiles:
        lines.append(f"    SMILES: {_truncate(hit.canonical_smiles, 70)}")
    
    if hit.target_profile:
        lines.append("")
        lines.append("    --- TARGET PROFILE ---")
        profile_lines = _format_compound_profile(0, hit.target_profile)
        for line in profile_lines[1:]:
            lines.append(f"    {line.strip()}")
    
    if hit.recent_trials:
        lines.append("")
        lines.append(f"    --- CLINICAL TRIALS ({len(hit.recent_trials)}) ---")
        max_trials = len(hit.recent_trials) if detail_level == "comprehensive" else 5
        for t in hit.recent_trials[:max_trials]:
            lines.append(f"      • {t.nct_id} | Phase: {t.phase or 'N/A'} | Status: {t.trial_status or 'Unknown'}")
            lines.append(f"        {_truncate(t.trial_title, 70)}")
        if detail_level != "comprehensive" and len(hit.recent_trials) > 5:
            lines.append(f"      ... and {len(hit.recent_trials) - 5} more trials")
    
    if hit.forms:
        lines.append("")
        lines.append(f"    --- MOLECULAR FORMS ({len(hit.forms)}) ---")
        max_forms = len(hit.forms) if detail_level == "comprehensive" else 5
        for f in hit.forms[:max_forms]:
            name = f.form_name or f.chembl_id or f"MOL_{f.mol_id}"
            salt = f" ({f.salt_form})" if f.salt_form else ""
            stereo = f" [{f.stereo_type}]" if f.stereo_type else ""
            lines.append(f"      • {name}{salt}{stereo}")
        if detail_level != "comprehensive" and len(hit.forms) > 5:
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


def _format_drug_indication(rank: int, hit: DrugIndicationHit) -> list[str]:
    """Format a drug-indication hit."""
    lines = []

    phase = f"phase {hit.max_phase}" if hit.max_phase is not None else "phase N/A"
    status = "APPROVED" if hit.is_approved else "investigational"
    drug_type = ""
    if hit.is_biotherapeutic:
        drug_type = f" [{hit.molecule_type or 'Biotherapeutic'}]"
    else:
        drug_type = " [small molecule]"

    lines.append(f"[{rank}] {hit.concept_name}{drug_type} | {status}, {phase}")

    if hit.matched_indication:
        lines.append(f"    Indication: {hit.matched_indication}")
    if hit.match_reason:
        lines.append(f"    Matched via: {hit.match_reason}")
    if hit.sources:
        lines.append(f"    Sources: {', '.join(hit.sources)}")

    return lines


def _format_target_pathway(rank: int, hit: TargetPathwayHit) -> list[str]:
    """Format a target-pathway hit."""
    lines = []
    pathway_desc = hit.protein_class_desc or "unknown pathway class"
    lines.append(f"[{rank}] {hit.gene_symbol} is associated with a signaling pathway")
    lines.append(f"    Pathway class: {pathway_desc}")
    if hit.synonyms:
        lines.append(f"    Synonyms: {', '.join(hit.synonyms[:5])}")
    return lines


def _format_pharmacology_output(
    result: TargetSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format pharmacology search results for LLM consumption."""
    lines = []
    
    # === STATUS HEADER ===
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
    
    # === WARNINGS ===
    if result.warnings:
        lines.append("")
        for w in result.warnings:
            lines.append(f"⚠ {w}")
    
    # === DIAGNOSTICS FOR ERRORS AND NOT_FOUND ===
    if result.status in ["error", "invalid_input", "not_found"]:
        # Use diagnostics from the search
        if result.diagnostics:
            lines.extend(_format_diagnostics(result.diagnostics, verbose=(result.status == "error")))
        else:
            # Fallback suggestions if no diagnostics provided
            lines.append("")
            lines.append("Suggestions:")
            if result.status == "invalid_input":
                lines.append("  → Check required parameters for this search mode")
                lines.append("  → Refer to the tool documentation for examples")
            elif result.status == "error":
                lines.append("  → Check input parameters and try again")
                lines.append("  → If SMILES error, verify structure is valid")
            elif result.status == "not_found":
                lines.append("  → Check spelling of drug/gene names")
                lines.append("  → Try alternative names or synonyms")
                lines.append("  → Try lowering min_pchembl threshold")
        
        results_text = "\n".join(lines)
        signals = build_handoff_signals(source_tool or "", result_context or {})
        if not signals:
            signals = "[AGENT_SIGNALS]\n---\nRelated searches:\n  -> None"
        return f"[RESULTS]\n{results_text}\n\n{signals}"
    
    # === RESULTS ===
    if not result.hits:
        results_text = "\n".join(lines)
        signals = build_handoff_signals(source_tool or "", result_context or {})
        if not signals:
            signals = "[AGENT_SIGNALS]\n---\nRelated searches:\n  -> None"
        return f"[RESULTS]\n{results_text}\n\n{signals}"
    
    lines.append("")
    lines.append("=" * 60)
    
    max_hits = 100 if detail_level == "comprehensive" else 30
    for i, hit in enumerate(result.hits[:max_hits], 1):
        lines.append("")
        if detail_level == "brief":
            if isinstance(hit, CompoundTargetProfile):
                best = f"{hit.best_pchembl:.1f}" if hit.best_pchembl else "N/A"
                chembl = hit.chembl_id or "N/A"
                lines.append(f"[{i}] {hit.concept_name} | ChEMBL: {chembl} | best pChEMBL: {best}")
            elif isinstance(hit, DrugForTargetHit):
                activity = f"{hit.activity_type}={hit.activity_value_nm:.1f} nM" if hit.activity_value_nm else "activity N/A"
                lines.append(f"[{i}] {hit.concept_name} | {activity}")
            elif isinstance(hit, DrugProfileResult):
                lines.append(
                    f"[{i}] {hit.concept_name} | forms: {hit.n_forms} | targets: {hit.n_activity_targets} | trials: {hit.n_clinical_trials}"
                )
            elif isinstance(hit, ClinicalTrialHit):
                lines.append(f"[{i}] {hit.nct_id} | Phase: {hit.phase or 'N/A'} | Status: {hit.trial_status or 'Unknown'}")
            elif isinstance(hit, MoleculeForm):
                name = hit.form_name or hit.chembl_id or f"MOL_{hit.mol_id}"
                lines.append(f"[{i}] {name} | ChEMBL: {hit.chembl_id or 'N/A'}")
            elif isinstance(hit, DrugIndicationHit):
                status = "APPROVED" if hit.is_approved else "investigational"
                phase = f"phase {hit.max_phase}" if hit.max_phase is not None else "phase N/A"
                drug_type = hit.molecule_type or ("biotherapeutic" if hit.is_biotherapeutic else "small molecule")
                lines.append(f"[{i}] {hit.concept_name} | {status}, {phase} | {drug_type}")
            elif isinstance(hit, TargetPathwayHit):
                pathway_desc = hit.protein_class_desc or "unknown pathway class"
                lines.append(f"[{i}] {hit.gene_symbol} | pathway: {pathway_desc}")
            else:
                lines.append(f"[{i}] {type(hit).__name__}: {hit}")
        else:
            if isinstance(hit, CompoundTargetProfile):
                lines.extend(_format_compound_profile(i, hit))
            elif isinstance(hit, DrugForTargetHit):
                lines.extend(_format_drug_for_target(i, hit))
            elif isinstance(hit, DrugProfileResult):
                lines.extend(_format_drug_profile_result(i, hit, detail_level=detail_level))
            elif isinstance(hit, ClinicalTrialHit):
                lines.extend(_format_clinical_trial(i, hit))
            elif isinstance(hit, MoleculeForm):
                lines.extend(_format_molecule_form(i, hit))
            elif isinstance(hit, DrugIndicationHit):
                lines.extend(_format_drug_indication(i, hit))
            elif isinstance(hit, TargetPathwayHit):
                lines.extend(_format_target_pathway(i, hit))
            else:
                lines.append(f"[{i}] {type(hit).__name__}: {hit}")
    
    if result.total_hits > max_hits:
        lines.append("")
        lines.append(f"... showing {max_hits} of {result.total_hits} results. Use 'limit' parameter for more.")
    
    # === DIAGNOSTICS FOR SUCCESS (if there are suggestions) ===
    if result.diagnostics and result.diagnostics.suggestions:
        lines.append("")
        lines.append("Notes:")
        for suggestion in result.diagnostics.suggestions[:3]:
            lines.append(f"  ℹ {suggestion}")

    if detail_level == "comprehensive" and result.diagnostics:
        lines.extend(_format_diagnostics(result.diagnostics, verbose=True))
    
    results_text = "\n".join(lines)
    signals = build_handoff_signals(source_tool or "", result_context or {})
    if not signals:
        signals = "[AGENT_SIGNALS]\n---\nRelated searches:\n  -> None"
    return f"[RESULTS]\n{results_text}\n\n{signals}"


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
    
    if not gene_symbol.replace("-", "").replace("_", "").isalnum():
        return None, f"gene_symbol '{gene_symbol}' contains invalid characters. Use official HGNC symbols."
    
    return gene_symbol, None


def _validate_text_query(value: str | None, label: str) -> tuple[str | None, str | None]:
    """Validate a generic text query parameter."""
    if not value or not isinstance(value, str) or not value.strip():
        return None, f"{label} is required and cannot be empty"
    return value.strip(), None


def _validate_smiles(smiles: str | None) -> tuple[str | None, str | None]:
    """Validate SMILES input. Returns (cleaned_smiles, error_message)."""
    if not smiles:
        return None, "smiles is required. Provide a valid SMILES string."
    
    smiles = smiles.strip()
    
    if len(smiles) < 1:
        return None, "smiles cannot be empty."
    
    if len(smiles) > 5000:
        return None, f"smiles is too long ({len(smiles)} chars). Maximum is 5000 characters."

    # Prefer RDKit validation at the tool boundary so malformed SMILES are
    # rejected consistently before dispatching to deeper search handlers.
    try:
        from rdkit import Chem  # pyright: ignore[reportMissingImports]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid SMILES syntax. Check parentheses/ring closures/aromatic tokens."
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical or smiles, None
    except ImportError:
        # Fallback only if RDKit is unavailable in the current runtime.
        valid_chars = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            "[]()=#@+-\\/%.:*"
        )
        invalid = set(smiles) - valid_chars
        if invalid:
            return None, f"smiles contains invalid characters: {invalid}. Check SMILES syntax."
        return smiles, None
    except Exception as e:
        return None, f"invalid SMILES syntax: {type(e).__name__}: {e}"


def _validate_activity_type(activity_type: str) -> tuple[ActivityType | None, str | None]:
    """Validate activity_type parameter. Returns (value, error_message)."""
    if not isinstance(activity_type, str) or not activity_type.strip():
        return ActivityType.ALL, None

    key = activity_type.strip().upper()
    mapping = {
        "IC50": ActivityType.IC50,
        "KI": ActivityType.KI,
        "KD": ActivityType.KD,
        "EC50": ActivityType.EC50,
        "ALL": ActivityType.ALL,
    }
    if key not in mapping:
        allowed = ", ".join(mapping.keys())
        return None, f"activity_type must be one of: {allowed}"
    return mapping[key], None


def _validate_confidence(confidence: str) -> tuple[DataConfidence | None, str | None]:
    """Validate min_confidence parameter. Returns (value, error_message)."""
    if not isinstance(confidence, str) or not confidence.strip():
        return DataConfidence.ANY, None

    key = confidence.strip().upper()
    mapping = {
        "HIGH": DataConfidence.HIGH,
        "MEDIUM": DataConfidence.MEDIUM,
        "LOW": DataConfidence.LOW,
        "ANY": DataConfidence.ANY,
    }
    if key not in mapping:
        allowed = ", ".join(mapping.keys())
        return None, f"min_confidence must be one of: {allowed}"
    return mapping[key], None


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
        return 500, None
    
    return limit, None


def _validate_detail_level(detail_level: str | None) -> tuple[str | None, str | None]:
    """Validate detail_level parameter. Returns (value, error_message)."""
    if not detail_level or not isinstance(detail_level, str):
        return "standard", None
    value = detail_level.strip().lower()
    allowed = {"brief", "standard", "comprehensive"}
    if value not in allowed:
        return None, "detail_level must be one of: brief, standard, comprehensive"
    return value, None


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
    detail_level: str = "standard",
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

    detail_level, error = _validate_detail_level(detail_level)
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
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_targets",
            result_context={"drug_name": drug_name},
        )
    
    except Exception as e:
        return (
            f"✗ Error searching drug targets: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}', min_pchembl={min_pchembl}, data_source='{data_source}'\n\n"
            "Suggestions:\n"
            "  → Check drug name spelling\n"
            "  → Try generic name instead of brand name\n"
            "  → For biologics (antibodies), try the INN name"
        )


@tool("search_target_drugs", return_direct=False)
@robust_unwrap_llm_inputs
async def search_target_drugs(
    gene_symbol: str,
    min_pchembl: float = 5.0,
    data_source: str = "both",
    molecule_type: str = "all",
    biotherapeutic_subtype: str = "all",
    limit: int = 50,
    detail_level: str = "standard",
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
        molecule_type: Filter by molecule type.
            - "all" (default): Small molecules + biotherapeutics
            - "small_molecule": Only small molecules
            - "biotherapeutic": Only biologics
        biotherapeutic_subtype: Filter biologics by subtype (only when molecule_type="biotherapeutic").
            - "all" (default): All biologic subtypes
            - "antibody": Antibodies
            - "enzyme": Enzymes
            - "protein": Proteins
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        Formatted list of drugs/compounds with activity data and/or mechanisms.
    """
    gene_symbol, error = _validate_gene_symbol(gene_symbol)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    ds, error = _validate_data_source(data_source)
    if error:
        return f"✗ Input error: {error}"

    allowed_types = {"all", "small_molecule", "biotherapeutic"}
    if molecule_type not in allowed_types:
        return f"✗ Input error: molecule_type must be one of {sorted(allowed_types)}"

    allowed_subtypes = {"all", "antibody", "enzyme", "protein"}
    if biotherapeutic_subtype not in allowed_subtypes:
        return f"✗ Input error: biotherapeutic_subtype must be one of {sorted(allowed_subtypes)}"
    if molecule_type != "biotherapeutic" and biotherapeutic_subtype != "all":
        return "✗ Input error: biotherapeutic_subtype requires molecule_type='biotherapeutic'"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_TARGET,
            query=gene_symbol,
            min_pchembl=min_pchembl,
            data_source=ds,
            molecule_type_filter=molecule_type,
            biotherapeutic_subtype=biotherapeutic_subtype,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_target_drugs",
            result_context={"gene_symbol": gene_symbol},
        )
    
    except Exception as e:
        return (
            f"✗ Error searching target drugs: {type(e).__name__}: {e}\n\n"
            f"Input: gene_symbol='{gene_symbol}', min_pchembl={min_pchembl}\n\n"
            "Suggestions:\n"
            "  → Verify gene symbol (use official HGNC symbol)\n"
            "  → Common symbols: EGFR, BRAF, ABL1, JAK2, CDK4, ERBB2\n"
            "  → Try lowering min_pchembl to get more results"
        )


@tool("search_similar_molecules", return_direct=False)
@robust_unwrap_llm_inputs
async def search_similar_molecules(
    smiles: str,
    similarity_threshold: float = 0.7,
    min_pchembl: float = 5.0,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Find molecules structurally similar to a query compound.
    
    Uses Tanimoto similarity on Morgan fingerprints to find analogs.
    Useful for finding related compounds or understanding structure-activity relationships.
    
    Args:
        smiles: SMILES string of the query molecule.
        similarity_threshold: Minimum Tanimoto similarity (0.0 to 1.0).
            - 0.5 = distant analogs (many results)
            - 0.7 = similar scaffold (default)
            - 0.85 = close analogs
            - 0.95 = very similar (stereoisomers, salts)
        min_pchembl: Minimum potency for activity data (default: 5.0).
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        List of similar molecules with similarity scores and activity data.
    """
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

    detail_level, error = _validate_detail_level(detail_level)
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
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_similar_molecules",
        )
    
    except Exception as e:
        smiles_preview = _truncate(smiles, 50)
        return (
            f"✗ Error searching similar molecules: {type(e).__name__}: {e}\n\n"
            f"Input SMILES: {smiles_preview}\n\n"
            "Suggestions:\n"
            "  → Verify the SMILES string is valid\n"
            "  → Use a chemical structure editor to generate valid SMILES\n"
            "  → Try lowering similarity_threshold for more results"
        )


@tool("search_exact_structure", return_direct=False)
@robust_unwrap_llm_inputs
async def search_exact_structure(
    smiles: str,
    min_pchembl: float = 5.0,
    detail_level: str = "standard",
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
    """
    smiles, error = _validate_smiles(smiles)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.EXACT_STRUCTURE,
            smiles=smiles,
            min_pchembl=min_pchembl,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_exact_structure",
        )
    
    except Exception as e:
        smiles_preview = _truncate(smiles, 50)
        return (
            f"✗ Error in exact structure search: {type(e).__name__}: {e}\n\n"
            f"Input SMILES: {smiles_preview}\n\n"
            "Suggestions:\n"
            "  → Verify the SMILES string is valid\n"
            "  → Try search_similar_molecules with high threshold instead"
        )


@tool("search_substructure", return_direct=False)
@robust_unwrap_llm_inputs
async def search_substructure(
    pattern: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Find molecules containing a specific substructure.
    
    Use SMILES or SMARTS patterns to find compounds with specific 
    chemical features (e.g., a particular ring system, functional group).
    
    Note: This search can be slow for very common substructures (e.g., benzene).
    
    Args:
        pattern: SMILES or SMARTS pattern to search for.
        limit: Maximum results (default: 50, max: 500).
    
    Returns:
        List of molecules containing the substructure.
    """
    if not pattern or not pattern.strip():
        return "✗ Input error: pattern is required. Provide a SMILES or SMARTS pattern."
    
    pattern = pattern.strip()
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.SUBSTRUCTURE,
            smarts=pattern,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_substructure",
        )
    
    except Exception as e:
        return (
            f"✗ Error in substructure search: {type(e).__name__}: {e}\n\n"
            f"Input pattern: {pattern}\n\n"
            "Suggestions:\n"
            "  → Check SMILES/SMARTS syntax\n"
            "  → Very common substructures (benzene) may timeout\n"
            "  → Try a more specific pattern"
        )


@tool("get_drug_profile", return_direct=False)
@robust_unwrap_llm_inputs
async def get_drug_profile(
    drug_name: str,
    include_trials: bool = True,
    include_forms: bool = True,
    min_pchembl: float = 5.0,
    detail_level: str = "standard",
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
    """
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
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
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="get_drug_profile",
            result_context={"drug_name": drug_name, "search_type": "drug_profile"},
        )
    
    except Exception as e:
        return (
            f"✗ Error getting drug profile: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}'\n\n"
            "Suggestions:\n"
            "  → Check drug name spelling\n"
            "  → Try alternative names or synonyms"
        )


@tool("get_drug_forms", return_direct=False)
@robust_unwrap_llm_inputs
async def get_drug_forms(
    drug_name: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Get all molecular forms of a drug (salts, stereoisomers, etc.).
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
        limit: Maximum results (default: 50).
    
    Returns:
        List of all molecular forms with identifiers and SMILES.
    """
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUG_FORMS,
            query=drug_name,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="get_drug_forms",
            result_context={"drug_name": drug_name},
        )
    
    except Exception as e:
        return f"✗ Error getting drug forms: {type(e).__name__}: {e}"


@tool("search_drug_trials", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_trials(
    drug_name: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Find clinical trials for a drug.
    
    Searches for trials involving the specified drug from ClinicalTrials.gov.
    Works for both small molecules and biologics.
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
        limit: Maximum results (default: 50).
    
    Returns:
        List of clinical trials with NCT IDs, titles, phases, and status.
    """
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.TRIALS_FOR_DRUG,
            query=drug_name,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_trials",
            result_context={"drug_name": drug_name},
        )
    
    except Exception as e:
        return (
            f"✗ Error searching drug trials: {type(e).__name__}: {e}\n\n"
            f"Input: drug_name='{drug_name}'\n\n"
            "Suggestions:\n"
            "  → Check drug name spelling\n"
            "  → For biologics, try the INN name (e.g., 'pembrolizumab' not 'Keytruda')"
        )


@tool("compare_drugs_on_target", return_direct=False)
@robust_unwrap_llm_inputs
async def compare_drugs_on_target(
    target: str,
    drug_names: list[str] | str,
    detail_level: str = "standard",
) -> str:
    """
    Compare multiple drugs' activity against a single target.
    
    Args:
        target: Gene symbol of the target (e.g., "EGFR", "ABL1").
        drug_names: List of drug names to compare (at least 2).
    
    Returns:
        Comparison table showing relative potency and fold-differences.
    """
    target, error = _validate_gene_symbol(target)
    if error:
        return f"✗ Input error (target): {error}"
    
    if isinstance(drug_names, str):
        drug_names = [drug_names]
    
    if not drug_names or len(drug_names) == 0:
        return "✗ Input error: drug_names is required. Provide a list of drug names to compare."
    
    if len(drug_names) < 2:
        return "✗ Input error: Provide at least 2 drug names to compare."
    
    if len(drug_names) > 20:
        return "✗ Input error: Too many drugs. Maximum is 20 for comparison."
    
    validated_names = []
    for name in drug_names:
        clean_name, error = _validate_drug_name(name)
        if error:
            return f"✗ Input error (drug '{name}'): {error}"
        validated_names.append(clean_name)

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    try:
        search_input = TargetSearchInput(
            mode=SearchMode.COMPARE_DRUGS,
            target=target,
            drug_names=validated_names,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="compare_drugs_on_target",
            result_context={"gene_symbol": target},
        )
    
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
    detail_level: str = "standard",
) -> str:
    """
    Find drugs that are selective for one target over others.
    
    Args:
        target: Primary target gene symbol (the desired target).
        off_targets: Gene symbols of targets to avoid.
        min_selectivity_fold: Minimum fold-selectivity required (default: 10x).
        min_pchembl: Minimum potency on primary target (default: 6.0 = 1 μM).
        limit: Maximum results (default: 50).
    
    Returns:
        List of selective compounds with selectivity ratios.
    """
    target, error = _validate_gene_symbol(target)
    if error:
        return f"✗ Input error (target): {error}"
    
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
    
    validated_off_targets = []
    for ot in off_targets:
        clean_ot, error = _validate_gene_symbol(ot)
        if error:
            return f"✗ Input error (off_target '{ot}'): {error}"
        validated_off_targets.append(clean_ot)
    
    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"
    
    if not isinstance(min_selectivity_fold, (int, float)) or min_selectivity_fold < 1:
        return "✗ Input error: min_selectivity_fold must be >= 1.0"
    
    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
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
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_selective_drugs",
            result_context={"gene_symbol": target},
        )
    
    except Exception as e:
        return (
            f"✗ Error searching selective drugs: {type(e).__name__}: {e}\n\n"
            f"Input: target='{target}', off_targets={validated_off_targets}"
        )


# =============================================================================
# ADDITIONAL PHARMACOLOGY MODES
# =============================================================================

@tool("search_drug_activities", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_activities(
    drug_name: str,
    min_pchembl: float = 5.0,
    activity_type: str = "all",
    min_confidence: str = "any",
    data_source: str = "activity",
    include_all_organisms: bool = False,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Retrieve quantitative activity measurements for a drug.

    Useful for viewing IC50/Ki/Kd/EC50 measurements across targets.
    """
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"

    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"

    activity_enum, error = _validate_activity_type(activity_type)
    if error:
        return f"✗ Input error: {error}"

    confidence_enum, error = _validate_confidence(min_confidence)
    if error:
        return f"✗ Input error: {error}"

    ds, error = _validate_data_source(data_source)
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.ACTIVITIES_FOR_DRUG,
            query=drug_name,
            min_pchembl=min_pchembl,
            activity_type=activity_enum,
            min_confidence=confidence_enum,
            data_source=ds,
            include_all_organisms=include_all_organisms,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_activities",
            result_context={"drug_name": drug_name},
        )

    except Exception as e:
        return f"✗ Error searching drug activities: {type(e).__name__}: {e}"


@tool("search_target_activities", return_direct=False)
@robust_unwrap_llm_inputs
async def search_target_activities(
    gene_symbol: str,
    min_pchembl: float = 5.0,
    activity_type: str = "all",
    min_confidence: str = "any",
    data_source: str = "activity",
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """Retrieve quantitative activity measurements for a target gene."""
    gene_symbol, error = _validate_gene_symbol(gene_symbol)
    if error:
        return f"✗ Input error: {error}"

    min_pchembl, error = _validate_pchembl(min_pchembl)
    if error:
        return f"✗ Input error: {error}"

    activity_enum, error = _validate_activity_type(activity_type)
    if error:
        return f"✗ Input error: {error}"

    confidence_enum, error = _validate_confidence(min_confidence)
    if error:
        return f"✗ Input error: {error}"

    ds, error = _validate_data_source(data_source)
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.ACTIVITIES_FOR_TARGET,
            query=gene_symbol,
            min_pchembl=min_pchembl,
            activity_type=activity_enum,
            min_confidence=confidence_enum,
            data_source=ds,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_target_activities",
            result_context={"gene_symbol": gene_symbol},
        )

    except Exception as e:
        return f"✗ Error searching target activities: {type(e).__name__}: {e}"


@tool("search_drug_indications", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_indications(
    drug_name: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """Find indications (approved or reported) for a drug."""
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.INDICATIONS_FOR_DRUG,
            query=drug_name,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_indications",
            result_context={"drug_name": drug_name},
        )

    except Exception as e:
        return f"✗ Error searching drug indications: {type(e).__name__}: {e}"


@tool("search_indication_drugs", return_direct=False)
@robust_unwrap_llm_inputs
async def search_indication_drugs(
    indication: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """Find drugs associated with an indication or disease term."""
    indication, error = _validate_text_query(indication, "indication")
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_INDICATION,
            query=indication,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_indication_drugs",
        )

    except Exception as e:
        return f"✗ Error searching drugs for indication: {type(e).__name__}: {e}"


@tool("search_target_pathways", return_direct=False)
@robust_unwrap_llm_inputs
async def search_target_pathways(
    gene_symbol: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """Find pathway annotations for a target gene."""
    gene_symbol, error = _validate_gene_symbol(gene_symbol)
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.TARGET_PATHWAYS,
            query=gene_symbol,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_target_pathways",
            result_context={"gene_symbol": gene_symbol},
        )

    except Exception as e:
        return f"✗ Error searching target pathways: {type(e).__name__}: {e}"


@tool("search_drug_interactions", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_interactions(
    drug_name: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """Find known drug-drug interactions for a drug."""
    drug_name, error = _validate_drug_name(drug_name)
    if error:
        return f"✗ Input error: {error}"

    limit, error = _validate_limit(limit)
    if error:
        return f"✗ Input error: {error}"

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"

    try:
        search_input = TargetSearchInput(
            mode=SearchMode.DRUG_INTERACTIONS,
            query=drug_name,
            limit=limit,
        )

        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_interactions",
            result_context={"drug_name": drug_name},
        )

    except Exception as e:
        return f"✗ Error searching drug interactions: {type(e).__name__}: {e}"


# =============================================================================
# UNIFIED PHARMACOLOGY SEARCH TOOL
# =============================================================================

@tool("pharmacology_search", return_direct=False)
@robust_unwrap_llm_inputs
async def pharmacology_search(
    search_type: str,
    drug_name: str | None = None,
    gene_symbol: str | None = None,
    smiles: str | None = None,
    drug_names: list[str] | None = None,
    off_targets: list[str] | None = None,
    indication: str | None = None,
    min_pchembl: float = 5.0,
    similarity_threshold: float = 0.7,
    min_selectivity_fold: float = 10.0,
    data_source: str = "both",
    activity_type: str = "all",
    min_confidence: str = "any",
    include_all_organisms: bool = False,
    limit: int = 50,
    detail_level: str = "standard",
) -> str:
    """
    Unified pharmacology search tool for drug-target interactions, mechanisms, 
    clinical trials, and molecular similarity.
    
    Args:
        search_type: The type of search to perform. Required. One of:
            - "drug_targets": Find all protein targets for a drug (requires drug_name)
            - "target_drugs": Find all drugs that hit a target (requires gene_symbol)
            - "drug_profile": Get comprehensive drug profile (requires drug_name)
            - "drug_forms": Get all salt/stereo forms of a drug (requires drug_name)
            - "drug_trials": Find clinical trials for a drug (requires drug_name)
            - "similar_molecules": Find structurally similar molecules (requires smiles)
            - "exact_structure": Identify a molecule from SMILES (requires smiles)
            - "substructure": Find molecules containing a substructure (requires smiles)
            - "compare_drugs": Compare drugs on a target (requires gene_symbol + drug_names)
            - "selective_drugs": Find selective drugs (requires gene_symbol + off_targets)
            - "drug_activities": Activity measurements for a drug (requires drug_name)
            - "target_activities": Activity measurements for a target (requires gene_symbol)
            - "drug_indications": Indications for a drug (requires drug_name)
            - "indication_drugs": Drugs for an indication (requires indication)
            - "target_pathways": Pathways for a target (requires gene_symbol)
            - "drug_interactions": Drug-drug interactions (requires drug_name)
        
        drug_name: Name of the drug (brand, generic, or synonym).
            Required for: drug_targets, drug_profile, drug_forms, drug_trials
            Examples: "imatinib", "Gleevec", "aspirin", "pembrolizumab"
        
        gene_symbol: Gene symbol of the target protein (HGNC official symbol).
            Required for: target_drugs, compare_drugs, selective_drugs
            Examples: "EGFR", "ABL1", "JAK2", "BRAF", "CDK4", "ROS1"
        
        smiles: SMILES string of a molecule.
            Required for: similar_molecules, exact_structure, substructure
            Example: "CC(=O)Oc1ccccc1C(=O)O" (aspirin)
        
        drug_names: List of drug names for comparison.
            Required for: compare_drugs
            Example: ["imatinib", "dasatinib", "nilotinib"]
        
        off_targets: List of gene symbols to avoid (for selectivity search).
            Required for: selective_drugs
            Example: ["JAK1", "JAK3"] (to find JAK2-selective compounds)

        indication: Indication or disease term for indication_drugs mode.
            Required for: indication_drugs
            Example: "chronic myeloid leukemia"
        
        min_pchembl: Minimum potency threshold (pChEMBL value). Default: 5.0
            - 5.0 = 10 μM (weak, more results)
            - 6.0 = 1 μM (moderate)
            - 7.0 = 100 nM (good)
            - 8.0 = 10 nM (potent, fewer results)
        
        similarity_threshold: Minimum Tanimoto similarity for structure search. Default: 0.7
            - 0.5 = distant analogs
            - 0.7 = similar scaffold
            - 0.85 = close analogs
            - 0.95 = very similar
        
        min_selectivity_fold: Minimum fold-selectivity for selective_drugs. Default: 10.0
        
        data_source: Type of target data to return. Default: "both"
            - "both": Activity data + curated mechanisms
            - "activity": Only quantitative assay data (IC50, Ki, etc.)
            - "mechanism": Only curated mechanism of action data

        activity_type: Activity type filter for *_activities modes. Default: "all"
        min_confidence: Minimum confidence for activity data (high/medium/low/any).
        include_all_organisms: Include non-human targets (default: False)
        
        limit: Maximum number of results. Default: 50, max: 500
    
    Returns:
        Formatted search results with target/mechanism data, activity values,
        clinical trials, or molecular similarity scores depending on search_type.
    
    Examples:
        # Find targets for a drug
        pharmacology_search(search_type="drug_targets", drug_name="imatinib")
        
        # Find drugs for a target
        pharmacology_search(search_type="target_drugs", gene_symbol="EGFR", min_pchembl=7.0)
        
        # Get comprehensive drug profile
        pharmacology_search(search_type="drug_profile", drug_name="aspirin")
        
        # Find similar molecules
        pharmacology_search(search_type="similar_molecules", smiles="CC(=O)Oc1ccccc1C(=O)O")
        
        # Compare drugs on a target
        pharmacology_search(
            search_type="compare_drugs",
            gene_symbol="ABL1",
            drug_names=["imatinib", "dasatinib", "nilotinib"]
        )
        
        # Find selective drugs
        pharmacology_search(
            search_type="selective_drugs",
            gene_symbol="JAK2",
            off_targets=["JAK1", "JAK3"],
            min_selectivity_fold=10.0
        )

        # Find drug indications
        pharmacology_search(
            search_type="drug_indications",
            drug_name="imatinib"
        )

        # Find drugs for an indication
        pharmacology_search(
            search_type="indication_drugs",
            indication="chronic myeloid leukemia"
        )

        # Find target pathways
        pharmacology_search(
            search_type="target_pathways",
            gene_symbol="EGFR"
        )

        # Find drug interactions
        pharmacology_search(
            search_type="drug_interactions",
            drug_name="warfarin"
        )
    """
    
    # === VALIDATE SEARCH TYPE ===
    valid_search_types = {
        "drug_targets": SearchMode.TARGETS_FOR_DRUG,
        "target_drugs": SearchMode.DRUGS_FOR_TARGET,
        "drug_profile": SearchMode.DRUG_PROFILE,
        "drug_forms": SearchMode.DRUG_FORMS,
        "drug_trials": SearchMode.TRIALS_FOR_DRUG,
        "similar_molecules": SearchMode.SIMILAR_MOLECULES,
        "exact_structure": SearchMode.EXACT_STRUCTURE,
        "substructure": SearchMode.SUBSTRUCTURE,
        "compare_drugs": SearchMode.COMPARE_DRUGS,
        "selective_drugs": SearchMode.SELECTIVE_DRUGS,
        "drug_activities": SearchMode.ACTIVITIES_FOR_DRUG,
        "target_activities": SearchMode.ACTIVITIES_FOR_TARGET,
        "drug_indications": SearchMode.INDICATIONS_FOR_DRUG,
        "indication_drugs": SearchMode.DRUGS_FOR_INDICATION,
        "target_pathways": SearchMode.TARGET_PATHWAYS,
        "drug_interactions": SearchMode.DRUG_INTERACTIONS,
    }
    
    if not search_type:
        return (
            "✗ Input error: search_type is required.\n\n"
            "Valid search types:\n"
            "  • drug_targets - Find protein targets for a drug\n"
            "  • target_drugs - Find drugs that hit a target\n"
            "  • drug_profile - Get comprehensive drug profile\n"
            "  • drug_forms - Get all salt/stereo forms\n"
            "  • drug_trials - Find clinical trials\n"
            "  • similar_molecules - Find structurally similar molecules\n"
            "  • exact_structure - Identify a molecule from SMILES\n"
            "  • substructure - Find molecules containing a substructure\n"
            "  • compare_drugs - Compare drugs on a target\n"
            "  • selective_drugs - Find selective drugs\n"
            "  • drug_activities - Activity measurements for a drug\n"
            "  • target_activities - Activity measurements for a target\n"
            "  • drug_indications - Indications for a drug\n"
            "  • indication_drugs - Drugs for an indication\n"
            "  • target_pathways - Pathways for a target\n"
            "  • drug_interactions - Drug-drug interactions"
        )
    
    search_type_lower = search_type.lower().strip().replace("-", "_").replace(" ", "_")
    
    if search_type_lower not in valid_search_types:
        return (
            f"✗ Input error: Unknown search_type '{search_type}'.\n\n"
            "Valid search types:\n"
            "  • drug_targets - Find protein targets for a drug (requires drug_name)\n"
            "  • target_drugs - Find drugs that hit a target (requires gene_symbol)\n"
            "  • drug_profile - Get comprehensive drug profile (requires drug_name)\n"
            "  • drug_forms - Get all salt/stereo forms (requires drug_name)\n"
            "  • drug_trials - Find clinical trials (requires drug_name)\n"
            "  • similar_molecules - Find structurally similar molecules (requires smiles)\n"
            "  • exact_structure - Identify a molecule from SMILES (requires smiles)\n"
            "  • substructure - Find molecules with substructure (requires smiles)\n"
            "  • compare_drugs - Compare drugs on a target (requires gene_symbol + drug_names)\n"
            "  • selective_drugs - Find selective drugs (requires gene_symbol + off_targets)\n"
            "  • drug_activities - Activity measurements for a drug (requires drug_name)\n"
            "  • target_activities - Activity measurements for a target (requires gene_symbol)\n"
            "  • drug_indications - Indications for a drug (requires drug_name)\n"
            "  • indication_drugs - Drugs for an indication (requires indication)\n"
            "  • target_pathways - Pathways for a target (requires gene_symbol)\n"
            "  • drug_interactions - Drug-drug interactions (requires drug_name)"
        )
    
    mode = valid_search_types[search_type_lower]
    
    # === VALIDATE REQUIRED PARAMETERS FOR EACH SEARCH TYPE ===
    
    # Drug name required
    if mode in [SearchMode.TARGETS_FOR_DRUG, SearchMode.DRUG_PROFILE, 
                SearchMode.DRUG_FORMS, SearchMode.TRIALS_FOR_DRUG,
                SearchMode.ACTIVITIES_FOR_DRUG, SearchMode.INDICATIONS_FOR_DRUG,
                SearchMode.DRUG_INTERACTIONS]:
        if not drug_name or not drug_name.strip():
            return (
                f"✗ Input error: drug_name is required for search_type='{search_type}'.\n\n"
                "Example:\n"
                f"  pharmacology_search(search_type='{search_type}', drug_name='imatinib')"
            )
        drug_name = drug_name.strip()
        if len(drug_name) < 2:
            return f"✗ Input error: drug_name '{drug_name}' is too short."
    
    # Gene symbol required
    if mode in [SearchMode.DRUGS_FOR_TARGET, SearchMode.COMPARE_DRUGS, SearchMode.SELECTIVE_DRUGS]:
        if not gene_symbol or not gene_symbol.strip():
            return (
                f"✗ Input error: gene_symbol is required for search_type='{search_type}'.\n\n"
                "Example:\n"
                f"  pharmacology_search(search_type='{search_type}', gene_symbol='EGFR')\n\n"
                "Common gene symbols: EGFR, ABL1, JAK2, BRAF, CDK4, ROS1, NTRK1"
            )
        gene_symbol = gene_symbol.strip().upper()
        if len(gene_symbol) < 2 or len(gene_symbol) > 20:
            return f"✗ Input error: gene_symbol '{gene_symbol}' is invalid. Use official HGNC symbols."

    if mode in [SearchMode.ACTIVITIES_FOR_TARGET, SearchMode.TARGET_PATHWAYS]:
        if not gene_symbol or not gene_symbol.strip():
            return (
                f"✗ Input error: gene_symbol is required for search_type='{search_type}'.\n\n"
                "Example:\n"
                f"  pharmacology_search(search_type='{search_type}', gene_symbol='EGFR')"
            )
        gene_symbol = gene_symbol.strip().upper()
    
    # SMILES required
    if mode in [SearchMode.SIMILAR_MOLECULES, SearchMode.EXACT_STRUCTURE, SearchMode.SUBSTRUCTURE]:
        if not smiles or not smiles.strip():
            return (
                f"✗ Input error: smiles is required for search_type='{search_type}'.\n\n"
                "Example:\n"
                f"  pharmacology_search(search_type='{search_type}', smiles='CC(=O)Oc1ccccc1C(=O)O')"
            )
        smiles = smiles.strip()
        if len(smiles) > 5000:
            return f"✗ Input error: smiles is too long ({len(smiles)} chars). Maximum is 5000."
    
    # Drug names list required for compare
    if mode == SearchMode.COMPARE_DRUGS:
        if not drug_names or len(drug_names) < 2:
            return (
                "✗ Input error: drug_names list with at least 2 drugs is required for compare_drugs.\n\n"
                "Example:\n"
                "  pharmacology_search(\n"
                "      search_type='compare_drugs',\n"
                "      gene_symbol='ABL1',\n"
                "      drug_names=['imatinib', 'dasatinib', 'nilotinib']\n"
                "  )"
            )
        if isinstance(drug_names, str):
            drug_names = [drug_names]
        drug_names = [d.strip() for d in drug_names if d and d.strip()]
        if len(drug_names) < 2:
            return "✗ Input error: Provide at least 2 valid drug names to compare."
    
    # Off-targets required for selective
    if mode == SearchMode.SELECTIVE_DRUGS:
        if not off_targets or len(off_targets) == 0:
            return (
                "✗ Input error: off_targets list is required for selective_drugs.\n\n"
                "Example:\n"
                "  pharmacology_search(\n"
                "      search_type='selective_drugs',\n"
                "      gene_symbol='JAK2',\n"
                "      off_targets=['JAK1', 'JAK3']\n"
                "  )"
            )
        if isinstance(off_targets, str):
            off_targets = [off_targets]
        off_targets = [t.strip().upper() for t in off_targets if t and t.strip()]
        if len(off_targets) == 0:
            return "✗ Input error: Provide at least 1 valid off-target gene symbol."
        if len(off_targets) > 10:
            return "✗ Input error: Too many off-targets. Maximum is 10."

    if mode == SearchMode.DRUGS_FOR_INDICATION:
        if not indication or not isinstance(indication, str) or not indication.strip():
            return (
                "✗ Input error: indication is required for search_type='indication_drugs'.\n\n"
                "Example:\n"
                "  pharmacology_search(search_type='indication_drugs', indication='chronic myeloid leukemia')"
            )
    
    # === VALIDATE OPTIONAL PARAMETERS ===
    
    # min_pchembl
    if not isinstance(min_pchembl, (int, float)):
        return f"✗ Input error: min_pchembl must be a number, got {type(min_pchembl).__name__}"
    if min_pchembl < 0 or min_pchembl > 15:
        return (
            f"✗ Input error: min_pchembl={min_pchembl} is out of range (0.0 to 15.0).\n"
            "Common values: 5.0 (weak), 6.0 (moderate), 7.0 (good), 8.0 (potent)"
        )
    
    # similarity_threshold
    if not isinstance(similarity_threshold, (int, float)):
        return f"✗ Input error: similarity_threshold must be a number"
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        return f"✗ Input error: similarity_threshold={similarity_threshold} must be between 0.0 and 1.0"
    
    # min_selectivity_fold
    if not isinstance(min_selectivity_fold, (int, float)) or min_selectivity_fold < 1:
        return "✗ Input error: min_selectivity_fold must be >= 1.0"
    
    # data_source
    data_source_map = {
        "both": DataSource.BOTH,
        "activity": DataSource.ACTIVITY,
        "mechanism": DataSource.MECHANISM,
    }
    ds_key = data_source.lower().strip() if data_source else "both"
    if ds_key not in data_source_map:
        return (
            f"✗ Input error: Invalid data_source '{data_source}'.\n"
            "Valid options: 'both', 'activity', 'mechanism'"
        )
    ds = data_source_map[ds_key]

    activity_enum, error = _validate_activity_type(activity_type)
    if error:
        return f"✗ Input error: {error}"

    confidence_enum, error = _validate_confidence(min_confidence)
    if error:
        return f"✗ Input error: {error}"
    
    # limit
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            return f"✗ Input error: limit must be an integer"
    limit = max(1, min(500, limit))

    detail_level, error = _validate_detail_level(detail_level)
    if error:
        return f"✗ Input error: {error}"
    
    # === BUILD SEARCH INPUT ===
    try:
        query_value = indication if mode == SearchMode.DRUGS_FOR_INDICATION else (drug_name or gene_symbol)
        search_input = TargetSearchInput(
            mode=mode,
            query=query_value,
            smiles=smiles,
            smarts=smiles if mode == SearchMode.SUBSTRUCTURE else None,
            similarity_threshold=float(similarity_threshold),
            min_pchembl=float(min_pchembl),
            data_source=ds,
            activity_type=activity_enum,
            min_confidence=confidence_enum,
            include_all_organisms=include_all_organisms,
            target=gene_symbol,
            off_targets=off_targets or [],
            min_selectivity_fold=float(min_selectivity_fold),
            drug_names=drug_names or [],
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=True)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(
            result,
            detail_level=detail_level,
            source_tool="pharmacology_search",
            result_context={
                "search_type": search_type_lower,
                "drug_name": drug_name,
                "gene_symbol": gene_symbol,
            },
        )
    
    except Exception as e:
        # Build context-specific error message
        context_parts = [f"search_type='{search_type}'"]
        if drug_name:
            context_parts.append(f"drug_name='{drug_name}'")
        if gene_symbol:
            context_parts.append(f"gene_symbol='{gene_symbol}'")
        if smiles:
            smiles_preview = _truncate(smiles, 40)
            context_parts.append(f"smiles='{smiles_preview}'")
        
        return (
            f"✗ Error in pharmacology search: {type(e).__name__}: {e}\n\n"
            f"Input: {', '.join(context_parts)}\n\n"
            "Suggestions:\n"
            "  → Check spelling of drug/gene names\n"
            "  → Verify SMILES string is valid\n"
            "  → Try lowering min_pchembl for more results"
        )


TARGET_SEARCH_TOOLS = [
    search_drug_targets,
    search_target_drugs,
    search_similar_molecules,
    search_exact_structure,
    search_substructure,
    get_drug_profile,
    get_drug_forms,
    search_drug_trials,
    compare_drugs_on_target,
    search_selective_drugs,
    search_drug_activities,
    search_target_activities,
    search_drug_indications,
    search_indication_drugs,
    search_target_pathways,
    search_drug_interactions,
    pharmacology_search,
]