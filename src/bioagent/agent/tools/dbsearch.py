# dbsearch.py
"""
Biomedical research tools for LangChain/LangGraph agents.

Exposes clinical trials, drug labels, and pharmacology data through
LLM-friendly interfaces with flat parameters and formatted outputs.
"""

from __future__ import annotations

import json
import inspect
from functools import wraps

from langchain_core.tools import tool

# Your existing imports
from bioagent.data.ingest.config import DEFAULT_CONFIG
from bioagent.data.search.clinical_trial_search import (
    ClinicalTrialsSearchInput,
    ClinicalTrialsSearchOutput,
    TrialSearchHit,
    clinical_trials_search_async,
    SearchStrategy, 
    SortField,
    SortOrder,
    _truncate
)
from bioagent.data.search.openfda_and_dailymed_search import (
    DailyMedAndOpenFDAInput,
    DailyMedAndOpenFDASearchOutput,
    dailymed_and_openfda_search_async,
)
from bioagent.data.search.molecule_trial_search import (
    MoleculeTrialSearchInput,
    MoleculeTrialSearchOutput,
    molecule_trial_search_async,
)
from bioagent.data.search.adverse_events_search import (
    AdverseEventsSearchInput,
    AdverseEventsSearchOutput,
    adverse_events_search_async,
)
from bioagent.data.search.outcomes_search import (
    OutcomesSearchInput,
    OutcomesSearchOutput,
    outcomes_search_async,
)
from bioagent.data.search.orange_book_search import (
    OrangeBookSearchInput,
    OrangeBookSearchOutput,
    orange_book_search_async,
)
from bioagent.data.search.cross_db_lookup import (
    CrossDatabaseLookupInput,
    CrossDatabaseLookupOutput,
    cross_database_lookup_async,
)
from bioagent.data.search.biotherapeutic_sequence_search import (
    BiotherapeuticSearchInput,
    BiotherapeuticSearchOutput,
    biotherapeutic_sequence_search_async,
)
from bioagent.data.search.target_search import (
    PharmacologySearch,
    TargetSearchInput,
    TargetSearchOutput,
    SearchMode,
    DataSource,
)


# =============================================================================
# DECORATOR
# =============================================================================

def robust_unwrap_llm_inputs(func):
    """
    Decorator to normalize LLM outputs into Python types.
    
    Handles:
    - JSON strings: '["a", "b"]' → ["a", "b"]
    - Null strings: "null", "None" → None
    - Wrapped dicts: {"value": [...]} → [...]
    - Single values: "aspirin" → ["aspirin"] (when list expected)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        def process_value(value, param_name=None):
            if value is None:
                return None
            
            # Unwrap {"value": ...} or {"type": ..., "value": ...}
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
                return process_value(value, param_name)
            
            # Handle string representations
            if isinstance(value, str):
                stripped = value.strip().lower()
                
                # Null-like strings
                if stripped in ("null", "none", ""):
                    return None
                
                # JSON arrays/objects
                if value.strip().startswith(("[", "{")):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        pass
                
                # Boolean strings
                if stripped == "true":
                    return True
                if stripped == "false":
                    return False
                
                # Return as-is
                return value
            
            return value

        processed_kwargs = {k: process_value(v, k) for k, v in kwargs.items()}
        
        if inspect.iscoroutinefunction(func):
            return await func(*args, **processed_kwargs)
        return func(*args, **processed_kwargs)

    return wrapper


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def _format_hit_brief(hit: TrialSearchHit, index: int) -> str:
    """Format a single hit as a brief summary (for list view)."""
    lines = [
        f"[{index}] {hit.nct_id} (score: {hit.score:.2f})",
        f"    Title: {_truncate(hit.brief_title, 80)}",
        f"    Phase: {hit.phase_display} | Status: {hit.status_display}",
    ]
    
    if hit.enrollment:
        lines.append(f"    Enrollment: {hit.enrollment:,}")
    
    if hit.lead_sponsor:
        lines.append(f"    Sponsor: {_truncate(hit.lead_sponsor, 50)}")
    
    if hit.conditions:
        conds = ", ".join(hit.conditions[:3])
        if len(hit.conditions) > 3:
            conds += f" (+{len(hit.conditions) - 3} more)"
        lines.append(f"    Conditions: {conds}")
    
    if hit.interventions:
        intv = ", ".join(hit.interventions[:3])
        if len(hit.interventions) > 3:
            intv += f" (+{len(hit.interventions) - 3} more)"
        lines.append(f"    Interventions: {intv}")
    
    if hit.match_reasons:
        lines.append(f"    Matched on: {', '.join(hit.match_reasons)}")
    
    return "\n".join(lines)


def _format_hit_full(hit: TrialSearchHit, index: int) -> str:
    """Format a single hit with full details including rendered summary."""
    lines = [
        f"[{index}] {hit.nct_id} (score: {hit.score:.2f})",
        "=" * 60,
    ]
    
    if hit.rendered_summary:
        lines.append(hit.rendered_summary)
    else:
        # Fallback if no rendered summary
        lines.append(f"Title: {hit.brief_title}")
        lines.append(f"Phase: {hit.phase_display} | Status: {hit.status_display}")
        if hit.enrollment:
            lines.append(f"Enrollment: {hit.enrollment:,}")
        if hit.start_date:
            lines.append(f"Start Date: {hit.start_date}")
        if hit.completion_date:
            lines.append(f"Completion Date: {hit.completion_date}")
        if hit.lead_sponsor:
            lines.append(f"Sponsor: {hit.lead_sponsor}")
        if hit.conditions:
            lines.append(f"Conditions: {', '.join(hit.conditions)}")
        if hit.interventions:
            lines.append(f"Interventions: {', '.join(hit.interventions)}")
        if hit.countries:
            lines.append(f"Countries: {', '.join(hit.countries)}")
    
    return "\n".join(lines)


def _format_clinical_trials_output(
    output: ClinicalTrialsSearchOutput,
    brief: bool = False,
) -> str:
    """
    Format clinical trials search results for LLM consumption.
    
    Args:
        output: The search output
        brief: If True, show brief summaries; if False, show full details
    """
    if output.status == "error":
        return f"❌ Search failed: {output.error}"
    
    if output.status == "not_found" or not output.hits:
        return (
            "🔍 No clinical trials found matching your criteria.\n\n"
            "Suggestions:\n"
            "- Try broader search terms\n"
            "- Check spelling of condition/drug names\n"
            "- Remove some filters to expand results"
        )
    
    lines = [
        f"✅ Found {output.total_hits} clinical trial(s)",
        f"Query: {output.query_summary}",
    ]
    
    if output.filters_applied:
        lines.append(f"Filters: {', '.join(output.filters_applied)}")
    
    if output.has_more:
        lines.append(f"(Showing {len(output.hits)} of {output.total_hits} - more available)")
    
    lines.append("=" * 60)
    
    for i, hit in enumerate(output.hits, 1):
        lines.append("")
        if brief:
            lines.append(_format_hit_brief(hit, i))
        else:
            lines.append(_format_hit_full(hit, i))
    
    return "\n".join(lines)

def _format_drug_labels_output(output: DailyMedAndOpenFDASearchOutput) -> str:
    """Format drug label search results for LLM consumption."""
    if output.status == "error":
        return f"Search failed: {output.error}"
    
    if output.status == "not_found" or not output.results:
        return "No drug information found. Try different drug names or search terms."
    
    lines = [
        f"Found information for {len(output.results)} drug(s):",
        "=" * 40,
    ]
    
    for result in output.results:
        lines.append(f"\n--- {result.product_name} ---")
        
        if result.properties and result.properties.name:
            p = result.properties
            lines.append(f"  Official Name: {p.name}")
            if p.formula:
                lines.append(f"  Formula: {p.formula}")
            if p.smiles:
                lines.append(f"  SMILES: {p.smiles}")
        
        if not result.sections:
            lines.append("  No matching sections found.")
            continue
        
        lines.append(f"\n  Label Sections ({len(result.sections)}):")
        for section in result.sections:
            text = " ".join(section.text.strip().split())
            # Truncate very long sections
            if len(text) > 1000:
                text = text[:1000] + "... [truncated]"
            lines.append(f"\n  [{section.section_name}] (source: {section.source})")
            lines.append(f"    {text}")
    
    return "\n".join(lines)


def _format_pharmacology_output(output: TargetSearchOutput) -> str:
    """Format pharmacology search results for LLM consumption."""
    if output.status == "error":
        return f"Search failed: {output.error}"
    
    if output.status == "invalid_input":
        return f"Invalid search parameters: {output.error}"
    
    if output.status == "not_found" or not output.hits:
        return f"No results found for: {output.query_summary}. Try different search terms or lower the potency threshold."
    
    lines = [
        f"Search: {output.query_summary}",
        f"Found {output.total_hits} result(s):",
        "=" * 40,
    ]
    
    for warning in output.warnings:
        lines.append(f"⚠ {warning}")
    
    for i, hit in enumerate(output.hits[:50], 1):  # Limit output
        # Handle different hit types
        if hasattr(hit, "gene_symbol") and hasattr(hit, "pchembl"):
            # DrugTargetHit
            name = hit.concept_name or hit.molecule_name or hit.chembl_id or f"MOL_{hit.mol_id}"
            target = hit.gene_symbol if hit.gene_symbol != "N/A" else ""
            
            parts = [f"[{i}] {name}"]
            if target:
                parts.append(f"→ {target}")
            if hit.pchembl:
                parts.append(f"| pChEMBL: {hit.pchembl:.2f}")
            if hit.activity_value_nm and hit.activity_value_nm > 0:
                parts.append(f"({hit.activity_type}: {hit.activity_value_nm:.1f} nM)")
            if hit.tanimoto_similarity:
                parts.append(f"[similarity: {hit.tanimoto_similarity:.2f}]")
            if hit.selectivity_fold:
                parts.append(f"[selectivity: {hit.selectivity_fold:.1f}x]")
            
            lines.append(" ".join(parts))
            
            if hit.canonical_smiles:
                smiles_preview = hit.canonical_smiles[:60] + "..." if len(hit.canonical_smiles) > 60 else hit.canonical_smiles
                lines.append(f"    SMILES: {smiles_preview}")
        
        elif hasattr(hit, "nct_id"):
            # ClinicalTrialHit
            lines.append(f"[{i}] {hit.nct_id} | {hit.phase or 'N/A'} | {hit.trial_status or 'N/A'}")
            lines.append(f"    {hit.trial_title[:100]}...")
        
        elif hasattr(hit, "fold_vs_best"):
            # DrugComparisonHit
            val = f"{hit.activity_value_nm:.1f} nM" if hit.activity_value_nm else "N/A"
            fold = f"{hit.fold_vs_best:.1f}x" if hit.fold_vs_best else "best"
            lines.append(f"[{i}] {hit.drug_name}: {val} (vs best: {fold}, n={hit.n_measurements})")
        
        elif hasattr(hit, "form_name"):
            # MoleculeForm
            name = hit.form_name or hit.chembl_id or f"MOL_{hit.mol_id}"
            salt = f" ({hit.salt_form})" if hit.salt_form else ""
            lines.append(f"[{i}] {name}{salt}")
            if hit.canonical_smiles:
                lines.append(f"    SMILES: {hit.canonical_smiles[:60]}...")
        
        elif hasattr(hit, "n_targets"):
            # DrugProfileResult
            lines.append(f"\nDrug Profile: {hit.concept_name}")
            lines.append(f"  - Forms: {hit.n_forms} ({hit.n_salt_forms} salts, {hit.n_stereo_variants} stereoisomers)")
            lines.append(f"  - Known targets: {hit.n_targets}")
            lines.append(f"  - Clinical trials: {hit.n_clinical_trials}")
            
            if hit.top_targets:
                lines.append("  - Top targets by potency:")
                for t in hit.top_targets[:5]:
                    val = f"{t.activity_value_nm:.1f} nM" if t.activity_value_nm else "N/A"
                    lines.append(f"      • {t.gene_symbol}: {val}")
        
        else:
            # Fallback
            lines.append(f"[{i}] {hit}")
    
    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")
    
    return "\n".join(lines)


def _format_molecule_trials_output(output: MoleculeTrialSearchOutput) -> str:
    """Format molecule-trial connectivity results."""
    if output.status == "error":
        return f"❌ Molecule-trial search failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found" or not output.hits:
        return f"🔍 No molecule-trial results found for: {output.query_summary}"

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    for i, hit in enumerate(output.hits[:50], 1):
        if output.mode == "molecules_by_condition":
            name = getattr(hit, "concept_name", None) or "Unknown"
            lines.append(f"[{i}] {name} | Trials: {getattr(hit, 'n_trials', 0)}")
            continue

        nct_id = getattr(hit, "nct_id", "N/A")
        phase = getattr(hit, "phase", None) or "N/A"
        status = getattr(hit, "status", None) or "N/A"
        lines.append(f"[{i}] {nct_id} | {phase} | {status}")

        title = getattr(hit, "brief_title", None)
        if title:
            lines.append(f"    {title}")

        concept_name = getattr(hit, "concept_name", None) or getattr(hit, "molecule_name", None)
        if concept_name:
            lines.append(f"    Molecule: {concept_name}")

        match_type = getattr(hit, "match_type", None)
        confidence = getattr(hit, "confidence", None)
        if match_type or confidence is not None:
            conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
            lines.append(f"    Match: {match_type or 'N/A'} | Confidence: {conf_text}")

        inchi_key = getattr(hit, "inchi_key", None)
        if inchi_key:
            lines.append(f"    InChIKey: {inchi_key}")

    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return "\n".join(lines)


def _format_adverse_events_output(output: AdverseEventsSearchOutput) -> str:
    """Format adverse events search results."""
    if output.status == "error":
        return f"❌ Adverse event search failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found" or not output.hits:
        return f"🔍 No adverse event results found for: {output.query_summary}"

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    for i, hit in enumerate(output.hits[:50], 1):
        if output.mode == "events_for_drug":
            term = getattr(hit, "adverse_event_term", None) or "Unknown event"
            event_type = getattr(hit, "event_type", None) or "N/A"
            affected = getattr(hit, "subjects_affected", None)
            at_risk = getattr(hit, "subjects_at_risk", None)
            n_trials = getattr(hit, "n_trials", None)
            lines.append(f"[{i}] {term} | {event_type}")
            lines.append(f"    Affected: {affected if affected is not None else 'N/A'} / {at_risk if at_risk is not None else 'N/A'}")
            if n_trials is not None:
                lines.append(f"    Trials: {n_trials}")
            continue

        if output.mode == "drugs_with_event":
            nct_id = getattr(hit, "nct_id", "N/A")
            title = getattr(hit, "brief_title", None)
            event_term = getattr(hit, "adverse_event_term", None)
            event_type = getattr(hit, "event_type", None)
            affected = getattr(hit, "subjects_affected", None)
            lines.append(f"[{i}] {nct_id} | {event_term or 'event'} | {event_type or 'N/A'}")
            if title:
                lines.append(f"    {title}")
            if affected is not None:
                lines.append(f"    Subjects affected: {affected}")
            continue

        # compare_safety
        drug_name = getattr(hit, "drug_name", None) or "Unknown drug"
        lines.append(f"[{i}] {drug_name}")
        top_events = getattr(hit, "top_events", []) or []
        for ev in top_events[:5]:
            term = getattr(ev, "adverse_event_term", None) or "Unknown event"
            event_type = getattr(ev, "event_type", None) or "N/A"
            affected = getattr(ev, "subjects_affected", None)
            n_trials = getattr(ev, "n_trials", None)
            lines.append(
                f"    - {term} | {event_type} | affected: {affected if affected is not None else 'N/A'} | trials: {n_trials if n_trials is not None else 'N/A'}"
            )

    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return "\n".join(lines)


def _format_outcomes_output(output: OutcomesSearchOutput) -> str:
    """Format clinical trial outcomes search results."""
    if output.status == "error":
        return f"❌ Outcomes search failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found" or not output.hits:
        return f"🔍 No outcome results found for: {output.query_summary}"

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    if output.mode == "outcomes_for_trial":
        for bundle in output.hits[:5]:
            nct_id = getattr(bundle, "nct_id", "N/A")
            outcomes = getattr(bundle, "outcomes", []) or []
            measurements = getattr(bundle, "measurements", []) or []
            analyses = getattr(bundle, "analyses", []) or []
            lines.append(f"Trial {nct_id}: outcomes={len(outcomes)}, measurements={len(measurements)}, analyses={len(analyses)}")

            for outcome in outcomes[:5]:
                title = getattr(outcome, "title", None) or "Unnamed outcome"
                outcome_type = getattr(outcome, "outcome_type", None) or "N/A"
                lines.append(f"    - [{outcome_type}] {title}")

            for analysis in analyses[:5]:
                title = getattr(analysis, "outcome_title", None) or "Outcome analysis"
                p_value = getattr(analysis, "p_value", None)
                p_text = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
                lines.append(f"    * Analysis: {title} | p={p_text}")
        return "\n".join(lines)

    for i, hit in enumerate(output.hits[:50], 1):
        if output.mode == "trials_with_outcome":
            nct_id = getattr(hit, "nct_id", "N/A")
            title = getattr(hit, "brief_title", None) or ""
            outcome_title = getattr(hit, "outcome_title", None) or "Outcome"
            outcome_type = getattr(hit, "outcome_type", None) or "N/A"
            lines.append(f"[{i}] {nct_id} | {outcome_type} | {outcome_title}")
            if title:
                lines.append(f"    {title}")
            continue

        # efficacy_comparison
        nct_id = getattr(hit, "nct_id", "N/A")
        outcome_title = getattr(hit, "outcome_title", None) or "Outcome"
        outcome_type = getattr(hit, "outcome_type", None) or "N/A"
        p_value = getattr(hit, "p_value", None)
        p_text = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
        lines.append(f"[{i}] {nct_id} | {outcome_type} | {outcome_title} | p={p_text}")

    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return "\n".join(lines)


def _format_orange_book_output(output: OrangeBookSearchOutput) -> str:
    """Format Orange Book search results."""
    if output.status == "error":
        return f"❌ Orange Book search failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found" or not output.hits:
        return f"🔍 No Orange Book results found for: {output.query_summary}"

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    for i, hit in enumerate(output.hits[:50], 1):
        trade = getattr(hit, "trade_name", None) or "Unknown"
        ingredient = getattr(hit, "ingredient", None) or "Unknown ingredient"
        te_code = getattr(hit, "te_code", None) or "N/A"
        appl_no = getattr(hit, "appl_no", None) or "N/A"
        lines.append(f"[{i}] {trade} | {ingredient} | TE: {te_code} | NDA: {appl_no}")

        if getattr(hit, "dosage_form", None) or getattr(hit, "route", None):
            lines.append(
                f"    Form: {getattr(hit, 'dosage_form', None) or 'N/A'} | Route: {getattr(hit, 'route', None) or 'N/A'}"
            )

        patents = getattr(hit, "patents", []) or []
        exclusivity = getattr(hit, "exclusivity", []) or []
        if patents:
            lines.append(f"    Patents: {len(patents)}")
            for patent in patents[:3]:
                lines.append(
                    f"      - {patent.patent_no or 'N/A'} | exp: {patent.patent_expiration_date or 'N/A'}"
                )
        if exclusivity:
            lines.append(f"    Exclusivity: {len(exclusivity)}")
            for exc in exclusivity[:3]:
                lines.append(f"      - {exc.exclusivity_code or 'N/A'} | {exc.exclusivity_date or 'N/A'}")

    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return "\n".join(lines)


def _format_cross_db_output(output: CrossDatabaseLookupOutput) -> str:
    """Format cross-database identifier lookup results."""
    if output.status == "error":
        return f"❌ Identifier lookup failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found":
        return f"🔍 No matches found for: {output.identifier}"

    lines = [
        f"✅ Identifier lookup: {output.identifier} ({output.identifier_type})",
        f"Molecules: {len(output.molecules)} | Labels: {len(output.labels)} | Trials: {len(output.trials)} | Targets: {len(output.targets)}",
        "=" * 40,
    ]

    for i, mol in enumerate(output.molecules[:10], 1):
        name = mol.concept_name or mol.pref_name or "Unknown"
        chembl = mol.chembl_id or "N/A"
        inchi = mol.inchi_key or "N/A"
        lines.append(f"[{i}] {name} | ChEMBL: {chembl} | InChIKey: {inchi}")

    if output.labels:
        lines.append("Labels:")
        for label in output.labels[:10]:
            lines.append(f"  - {label.title or 'Label'} ({label.set_id}) [{label.source}]")

    if output.trials:
        lines.append("Trials:")
        for trial in output.trials[:10]:
            lines.append(f"  - {trial.nct_id} | {trial.phase or 'N/A'} | {trial.status or 'N/A'}")

    if output.targets:
        lines.append("Targets:")
        for target in output.targets[:10]:
            pchembl = target.best_pchembl
            p_text = f"{pchembl:.2f}" if isinstance(pchembl, (int, float)) else "N/A"
            lines.append(f"  - {target.gene_symbol} | best pChEMBL: {p_text} | n={target.n_measurements or 0}")

    return "\n".join(lines)


def _format_biotherapeutic_output(output: BiotherapeuticSearchOutput) -> str:
    """Format biotherapeutic sequence search results."""
    if output.status == "error":
        return f"❌ Biotherapeutic search failed: {output.error}"
    if output.status == "invalid_input":
        return f"❌ Invalid input: {output.error}"
    if output.status == "not_found" or not output.hits:
        return f"🔍 No biotherapeutic results found for: {output.query_summary}"

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    for i, hit in enumerate(output.hits[:50], 1):
        name = getattr(hit, "pref_name", None) or "Unknown"
        chembl = getattr(hit, "chembl_id", None) or "N/A"
        bio_type = getattr(hit, "biotherapeutic_type", None) or "N/A"
        organism = getattr(hit, "organism", None) or "N/A"
        lines.append(f"[{i}] {name} | {bio_type} | ChEMBL: {chembl} | {organism}")

        components = getattr(hit, "components", []) or []
        for comp in components[:5]:
            comp_type = getattr(comp, "component_type", None) or "component"
            accession = getattr(comp, "uniprot_accession", None) or "N/A"
            seq_len = getattr(comp, "sequence_length", None)
            seq_text = f"{seq_len} aa" if seq_len else "N/A"
            lines.append(f"    - {comp_type} | UniProt: {accession} | {seq_text}")

    if output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return "\n".join(lines)


# =============================================================================
# CLINICAL TRIALS TOOLS
# =============================================================================

@tool("search_clinical_trials", return_direct=False)
@robust_unwrap_llm_inputs
async def search_clinical_trials(
    # === Search Queries ===
    query: str = "",
    condition: str | None = None,
    intervention: str | None = None,
    keyword: str | None = None,
    nct_ids: str | None = None,
    sponsor: str | None = None,
    
    # === Filters ===
    phase: str | None = None,
    status: str | None = None,
    study_type: str | None = None,
    intervention_type: str | None = None,
    outcome_type: str | None = None,
    eligibility_gender: str | None = None,
    eligibility_age_range: list[int] | tuple[int, int] | str | None = None,
    country: list[str] | str | None = None,
    
    # Date filters
    start_date_from: str | None = None,
    start_date_to: str | None = None,
    completion_date_from: str | None = None,
    completion_date_to: str | None = None,
    
    # Enrollment filters
    min_enrollment: int | None = None,
    max_enrollment: int | None = None,
    
    # Boolean filters
    has_results: bool | None = None,
    is_fda_regulated: bool | None = None,
    
    # === Search Options ===
    strategy: str = "combined",
    match_all: bool | None = None,
    similarity_threshold: float = 0.3,
    
    # === Output Options ===
    sort_by: str = "relevance",
    sort_order: str = "desc",
    limit: int = 10,
    offset: int = 0,
    
    # === Output Sections ===
    include_results: bool = False,
    include_adverse_events: bool = False,
    include_eligibility: bool = True,
    include_groups: bool = True,
    include_baseline: bool = False,
    include_sponsors: bool = True,
    include_countries: bool = True,
    
    # === Display Options ===
    brief_output: bool = False,
) -> str:
    """
    Search ClinicalTrials.gov for clinical studies.
    
    This tool searches over 500,000 clinical trials with flexible querying options.
    You can search by condition, intervention, keywords, sponsor, or NCT ID.
    
    ═══════════════════════════════════════════════════════════════════════════════
    SEARCH PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    query: General search term. Auto-detected as NCT ID if starts with "NCT",
           otherwise treated as a condition search. For more control, use the
           specific parameters below instead.
    
    condition: Disease or condition name
               Examples: "breast cancer", "type 2 diabetes", "EGFR-mutant NSCLC"
    
    intervention: Drug, therapy, or intervention name
                  Examples: "pembrolizumab", "radiation therapy", "CABG surgery"
    
    keyword: Free-text search across titles and descriptions
             Examples: "immunotherapy checkpoint", "first-line treatment"
    
    nct_ids: Comma-separated NCT IDs for direct lookup
             Example: "NCT04280705, NCT03456789"
    
    sponsor: Organization or company name
             Examples: "Pfizer", "NIH", "Memorial Sloan Kettering"
    
    ═══════════════════════════════════════════════════════════════════════════════
    FILTER PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    phase: Trial phase(s), comma-separated. Options:
           • "Phase 1", "Phase 2", "Phase 3", "Phase 4"
           • "Phase 1/Phase 2", "Phase 2/Phase 3"
           • "Early Phase 1", "N/A"
           Example: "Phase 2, Phase 3"
    
    status: Trial status(es), comma-separated. Options:
            • "Recruiting" - Currently enrolling participants
            • "Active, not recruiting" - Ongoing but not enrolling
            • "Completed" - Study finished
            • "Terminated" - Stopped early
            • "Withdrawn" - Withdrawn before enrollment
            • "Suspended" - Temporarily halted
            • "Not yet recruiting" - Approved but not started
            • "Enrolling by invitation" - By invitation only
            Example: "Recruiting, Active, not recruiting"
    
    study_type: Type of study. Options:
                • "Interventional" - Tests treatments/interventions
                • "Observational" - Observes outcomes without intervention
                • "Expanded Access" - Access to investigational drugs
    
    intervention_type: Type of intervention, comma-separated. Options:
                       • "Drug" - Pharmaceutical agents
                       • "Biological" - Vaccines, blood products, gene therapy
                       • "Device" - Medical devices
                       • "Procedure" - Surgical or other procedures
                       • "Radiation" - Radiation therapy
                       • "Behavioral" - Behavioral interventions
                       • "Dietary Supplement" - Vitamins, supplements
                       • "Genetic" - Gene therapy, genetic testing
                       • "Combination Product" - Drug-device combinations
                       • "Other" - Other intervention types
                       Example: "Drug, Biological"

    outcome_type: Filter by outcome type. Options:
                  • "primary", "secondary", "all"

    eligibility_gender: Filter by eligibility gender. Options:
                        • "male", "female", "all"

    eligibility_age_range: Age range filter for eligible participants.
                           Format: [min_age, max_age] or "min,max" (years).

    country: Country filter (single or comma-separated list).
             Examples: "United States", "Germany, France"
    
    ═══════════════════════════════════════════════════════════════════════════════
    DATE FILTERS (format: YYYY-MM-DD)
    ═══════════════════════════════════════════════════════════════════════════════
    
    start_date_from: Trials starting on or after this date
                     Example: "2020-01-01"
    
    start_date_to: Trials starting on or before this date
                   Example: "2024-12-31"
    
    completion_date_from: Trials completing on or after this date
    
    completion_date_to: Trials completing on or before this date
    
    ═══════════════════════════════════════════════════════════════════════════════
    ENROLLMENT FILTERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    min_enrollment: Minimum number of participants (e.g., 100)
    
    max_enrollment: Maximum number of participants (e.g., 5000)
    
    ═══════════════════════════════════════════════════════════════════════════════
    BOOLEAN FILTERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    has_results: True = only trials with posted results
                 False = only trials without results
                 None (default) = no filter
    
    is_fda_regulated: True = only FDA-regulated trials
                      False = exclude FDA-regulated trials
                      None (default) = no filter
    
    ═══════════════════════════════════════════════════════════════════════════════
    SEARCH OPTIONS
    ═══════════════════════════════════════════════════════════════════════════════
    
    strategy: Search matching strategy. Options:
              • "combined" (default) - Best of trigram + fulltext
              • "trigram" - Fuzzy matching (handles typos)
              • "fulltext" - Exact word matching
              • "exact" - Exact string matching
    
    match_all: If True, ALL search terms must match (AND logic)
               If False, ANY term can match (OR with scoring)
               If not specified, automatically uses AND logic when multiple search
               criteria are provided, OR logic for single criteria
    
    similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.3)
                          Lower = more results, higher = stricter matching
    
    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT OPTIONS
    ═══════════════════════════════════════════════════════════════════════════════
    
    sort_by: How to sort results. Options:
             • "relevance" (default) - By match score
             • "start_date" - By trial start date
             • "completion_date" - By completion date
             • "enrollment" - By number of participants
             • "nct_id" - By NCT ID
    
    sort_order: Sort direction
                • "desc" (default) - Descending
                • "asc" - Ascending
    
    limit: Maximum trials to return (default: 10, max: 100)
    
    offset: Number of results to skip for pagination (default: 0)
    
    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT SECTION CONTROLS
    ═══════════════════════════════════════════════════════════════════════════════
    
    include_results: Include outcome results and statistical analyses (default: False)
    
    include_adverse_events: Include adverse event data (default: False)
    
    include_eligibility: Include eligibility criteria (default: True)
    
    include_groups: Include arm/group descriptions (default: True)
    
    include_baseline: Include baseline measurements (default: False)
    
    include_sponsors: Include sponsor information (default: True)
    
    include_countries: Include country/location info (default: True)
    
    brief_output: Return brief summaries instead of full details (default: False)
    
    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLES
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. Find a specific trial by NCT ID:
       search_clinical_trials(nct_ids="NCT04280705")
    
    2. Search by condition:
       search_clinical_trials(condition="non-small cell lung cancer")
    
    3. Search by drug:
       search_clinical_trials(intervention="pembrolizumab")
    
    4. Combined search (trials for a specific drug in a specific condition):
       search_clinical_trials(
           condition="melanoma",
           intervention="nivolumab",
           match_all=True
       )
    
    5. Find recruiting Phase 3 trials:
       search_clinical_trials(
           condition="breast cancer",
           phase="Phase 3",
           status="Recruiting"
       )
    
    6. Find completed trials with results:
       search_clinical_trials(
           intervention="imatinib",
           status="Completed",
           has_results=True,
           include_results=True,
           include_adverse_events=True
       )
    
    7. Find large trials by a specific sponsor:
       search_clinical_trials(
           sponsor="Merck",
           min_enrollment=500,
           phase="Phase 3"
       )
    
    8. Find recent device trials:
       search_clinical_trials(
           intervention_type="Device",
           start_date_from="2023-01-01",
           status="Recruiting"
       )
    
    9. Find biological trials for autoimmune diseases:
       search_clinical_trials(
           condition="rheumatoid arthritis",
           intervention_type="Biological",
           phase="Phase 3",
           is_fda_regulated=True
       )
    
    10. Search with pagination:
        search_clinical_trials(
            condition="diabetes",
            limit=20,
            offset=20,  # Get results 21-40
            sort_by="enrollment",
            sort_order="desc"
        )
    
    Returns:
        Formatted clinical trial information including study design,
        enrollment, outcomes, and adverse events as requested.
    """
    from datetime import date as date_type
    
    try:
        # Handle legacy 'query' parameter for backwards compatibility
        if query and not any([condition, intervention, keyword, nct_ids]):
            query = query.strip()
            if query.upper().startswith("NCT"):
                nct_ids = query
            else:
                condition = query
        
        # Parse comma-separated NCT IDs
        nct_id_list = None
        if nct_ids:
            nct_id_list = [n.strip().upper() for n in nct_ids.split(",") if n.strip()]
        
        # Parse comma-separated phases
        phase_list = None
        if phase:
            phase_list = [p.strip() for p in phase.split(",") if p.strip()]
        
        # Parse comma-separated statuses
        status_list = None
        if status:
            status_list = [s.strip() for s in status.split(",") if s.strip()]
        
        # Parse comma-separated intervention types
        intervention_type_list = None
        if intervention_type:
            intervention_type_list = [t.strip() for t in intervention_type.split(",") if t.strip()]

        # Parse comma-separated countries
        country_list = None
        if country:
            if isinstance(country, str):
                country_list = [c.strip() for c in country.split(",") if c.strip()]
            elif isinstance(country, list):
                country_list = [c.strip() for c in country if isinstance(c, str) and c.strip()]

        # Parse eligibility age range
        age_range = None
        if eligibility_age_range:
            if isinstance(eligibility_age_range, (list, tuple)) and len(eligibility_age_range) == 2:
                try:
                    age_range = (int(eligibility_age_range[0]), int(eligibility_age_range[1]))
                except (TypeError, ValueError):
                    age_range = None
            elif isinstance(eligibility_age_range, str):
                separator = "," if "," in eligibility_age_range else "-"
                parts = [p.strip() for p in eligibility_age_range.split(separator) if p.strip()]
                if len(parts) == 2:
                    try:
                        age_range = (int(parts[0]), int(parts[1]))
                    except (TypeError, ValueError):
                        age_range = None

        # Auto-enable match_all=True when multiple search criteria are provided
        # Count non-None search criteria
        search_criteria_count = sum([
            condition is not None,
            intervention is not None,
            keyword is not None,
            nct_id_list is not None and len(nct_id_list) > 0,
            sponsor is not None
        ])

        # If user didn't explicitly set match_all, auto-detect based on criteria count
        if match_all is None:
            match_all = search_criteria_count > 1  # Use AND logic if multiple criteria

        # Parse dates
        def parse_date(date_str: str | None) -> date_type | None:
            if not date_str:
                return None
            try:
                parts = date_str.strip().split("-")
                return date_type(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                return None
        
        # Map string sort_by to enum
        sort_by_map = {
            "relevance": SortField.RELEVANCE,
            "start_date": SortField.START_DATE,
            "completion_date": SortField.COMPLETION_DATE,
            "enrollment": SortField.ENROLLMENT,
            "nct_id": SortField.NCT_ID,
        }
        sort_by_enum = sort_by_map.get(sort_by.lower(), SortField.RELEVANCE)
        
        # Map string sort_order to enum
        sort_order_enum = SortOrder.DESC if sort_order.lower() == "desc" else SortOrder.ASC
        
        # Map string strategy to enum
        strategy_map = {
            "combined": SearchStrategy.COMBINED,
            "trigram": SearchStrategy.TRIGRAM,
            "fulltext": SearchStrategy.FULLTEXT,
            "exact": SearchStrategy.EXACT,
        }
        strategy_enum = strategy_map.get(strategy.lower(), SearchStrategy.COMBINED)
        
        # Build search input
        search_input = ClinicalTrialsSearchInput(
            # Search queries
            condition=condition,
            intervention=intervention,
            keyword=keyword,
            nct_ids=nct_id_list,
            sponsor=sponsor,
            
            # Filters
            phase=phase_list,
            status=status_list,
            study_type=study_type,
            intervention_type=intervention_type_list,
            outcome_type=outcome_type,
            eligibility_gender=eligibility_gender,
            eligibility_age_range=age_range,
            country=country_list,
            
            # Date filters
            start_date_from=parse_date(start_date_from),
            start_date_to=parse_date(start_date_to),
            completion_date_from=parse_date(completion_date_from),
            completion_date_to=parse_date(completion_date_to),
            
            # Enrollment filters
            min_enrollment=min_enrollment,
            max_enrollment=max_enrollment,
            
            # Boolean filters
            has_results=has_results,
            is_fda_regulated=is_fda_regulated,
            
            # Search options
            strategy=strategy_enum,
            match_all=match_all,
            similarity_threshold=similarity_threshold,
            
            # Output options
            sort_by=sort_by_enum,
            sort_order=sort_order_enum,
            limit=min(limit, 100),
            offset=offset,
            include_study_json=True,
            
            # Output sections
            output_eligibility=include_eligibility,
            output_groups=include_groups,
            output_baseline_measurements=include_baseline,
            output_results=include_results,
            output_adverse_effects=include_adverse_events,
            output_sponsors=include_sponsors,
            output_countries=include_countries,
        )
        
        # Execute search
        result = await clinical_trials_search_async(DEFAULT_CONFIG, search_input)
        
        return _format_clinical_trials_output(result, brief=brief_output)
    
    except Exception as e:
        return f"❌ Error searching clinical trials: {type(e).__name__}: {e}"


# =============================================================================
# DRUG LABEL TOOLS (OpenFDA / DailyMed)
# =============================================================================

@tool("search_drug_labels", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_labels(
    drug_names: list[str] | str | None = None,
    section_queries: list[str] | str | None = None,
    keyword_query: list[str] | str | None = None,
    fetch_all_sections: bool = False,
    boxed_warnings_only: bool = False,
    drug_interactions_only: bool = False,
    adverse_reactions_only: bool = False,
    result_limit: int = 10,
) -> str:
    """
    Search FDA drug labels from OpenFDA and DailyMed databases.
    
    Use this tool to find drug information including:
    - Indications and usage
    - Warnings and precautions
    - Adverse reactions / side effects
    - Drug interactions
    - Dosage and administration
    - Contraindications
    - Clinical pharmacology
    
    Args:
        drug_names: Drug name(s) to search for. Can be brand or generic names.
            Examples: "Lipitor", "atorvastatin", ["Tylenol", "Advil"]
        section_queries: Label sections to retrieve. Use semantic names like:
            "warnings", "adverse reactions", "drug interactions", 
            "indications", "dosage", "contraindications", "pregnancy"
        keyword_query: Keywords to search for within the labels.
            Use this to find drugs associated with specific conditions or effects.
        fetch_all_sections: If True, returns ALL sections for the specified drugs.
            Use this when you need comprehensive drug information.
        boxed_warnings_only: If True, fetch only boxed warning sections.
        drug_interactions_only: If True, fetch only drug interaction sections.
        adverse_reactions_only: If True, fetch only adverse reaction sections.
        result_limit: Maximum number of results (default: 10, max: 50).
    
    Returns:
        Formatted drug label information organized by drug and section.
    
    Usage Patterns:
    
    1. **Get all info about a drug:**
       search_drug_labels(drug_names="Ozempic", fetch_all_sections=True)
    
    2. **Get specific sections:**
       search_drug_labels(drug_names="metformin", 
                         section_queries=["warnings", "drug interactions"])
    
    3. **Find drugs for a condition:**
       search_drug_labels(keyword_query="Stevens-Johnson syndrome")
    
    4. **Search within a drug's label:**
       search_drug_labels(drug_names="warfarin", keyword_query="vitamin K")
    
    5. **Targeted section + keyword:**
       search_drug_labels(drug_names="Humira",
                         section_queries=["adverse reactions"],
                         keyword_query="infection")

    6. **Quick safety lookups:**
       search_drug_labels(drug_names="warfarin", boxed_warnings_only=True)
       search_drug_labels(drug_names="atorvastatin", drug_interactions_only=True)
    """
    try:
        # Normalize inputs to lists
        if isinstance(drug_names, str):
            drug_names = [drug_names]
        if isinstance(section_queries, str):
            section_queries = [section_queries]
        if isinstance(keyword_query, str):
            keyword_query = [keyword_query]

        sections = list(section_queries or [])
        if boxed_warnings_only:
            sections.append("boxed warning")
        if drug_interactions_only:
            sections.append("drug interactions")
        if adverse_reactions_only:
            sections.append("adverse reactions")
        if sections:
            # Deduplicate while preserving order
            seen = set()
            section_queries = []
            for section in sections:
                if section and section not in seen:
                    section_queries.append(section)
                    seen.add(section)
        
        search_input = DailyMedAndOpenFDAInput(
            drug_names=drug_names,
            section_queries=section_queries,
            keyword_query=keyword_query,
            fetch_all_sections=fetch_all_sections,
            aggressive_deduplication=True,
            result_limit=min(result_limit, 50),
            top_n_drugs=5,
            sections_per_query=3,
        )
        
        result = await dailymed_and_openfda_search_async(DEFAULT_CONFIG, search_input)
        return _format_drug_labels_output(result)
    
    except ValueError as e:
        # Pydantic validation error - give helpful guidance
        return (
            f"Invalid search parameters: {e}\n\n"
            "Valid patterns:\n"
            "- drug_names only → get drug properties\n"
            "- drug_names + fetch_all_sections=True → get all label sections\n"
            "- drug_names + section_queries/keyword_query → targeted search\n"
            "- keyword_query only → discover drugs by keyword"
        )
    except Exception as e:
        return f"Error searching drug labels: {type(e).__name__}: {e}"


# =============================================================================
# MOLECULE-TRIAL CONNECTIVITY TOOLS
# =============================================================================

@tool("search_molecule_trials", return_direct=False)
@robust_unwrap_llm_inputs
async def search_molecule_trials(
    mode: str,
    molecule: str | None = None,
    inchi_key: str | None = None,
    condition: str | None = None,
    target_gene: str | None = None,
    min_pchembl: float = 6.0,
    phase: list[str] | str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Search clinical trials linked to molecules, conditions, or targets.

    Modes:
        - "trials_by_molecule": Find trials linked to a molecule (name or InChIKey).
        - "molecules_by_condition": Find molecules associated with a condition.
        - "trials_by_target": Find trials linked to a target gene.

    Args:
        mode: Search mode (see above).
        molecule: Molecule name (preferred name or synonym).
        inchi_key: InChIKey for precise molecule lookup.
        condition: Condition name for molecule discovery.
        target_gene: Target gene symbol for target-linked trials.
        min_pchembl: Potency threshold for target-linked mode (default: 6.0).
        phase: Optional list of trial phases (comma-separated or list).
        limit: Max results (default: 20, max: 500).
        offset: Offset for pagination (default: 0).
    """
    try:
        if isinstance(phase, str):
            phase_list = [p.strip() for p in phase.split(",") if p.strip()]
        elif isinstance(phase, list):
            phase_list = [p.strip() for p in phase if isinstance(p, str) and p.strip()]
        else:
            phase_list = None

        search_input = MoleculeTrialSearchInput(
            mode=mode,
            molecule_name=molecule,
            inchi_key=inchi_key,
            target_gene=target_gene,
            condition=condition,
            min_pchembl=min_pchembl,
            phase=phase_list,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await molecule_trial_search_async(DEFAULT_CONFIG, search_input)
        return _format_molecule_trials_output(result)
    except Exception as e:
        return f"Error searching molecule-trial links: {type(e).__name__}: {e}"


# =============================================================================
# ADVERSE EVENTS TOOLS
# =============================================================================

@tool("search_adverse_events", return_direct=False)
@robust_unwrap_llm_inputs
async def search_adverse_events(
    mode: str,
    drug_name: str | None = None,
    event_term: str | None = None,
    drug_names: list[str] | str | None = None,
    severity: str = "all",
    min_subjects_affected: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Search ClinicalTrials.gov adverse event data.

    Modes:
        - "events_for_drug": Top adverse events for a drug.
        - "drugs_with_event": Trials reporting a specific event.
        - "compare_safety": Compare adverse events across multiple drugs.
    """
    try:
        if isinstance(drug_names, str):
            drug_names = [drug_names]

        search_input = AdverseEventsSearchInput(
            mode=mode,
            drug_name=drug_name,
            event_term=event_term,
            drug_names=drug_names or [],
            severity=severity,
            min_subjects_affected=min_subjects_affected,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await adverse_events_search_async(DEFAULT_CONFIG, search_input)
        return _format_adverse_events_output(result)
    except Exception as e:
        return f"Error searching adverse events: {type(e).__name__}: {e}"


# =============================================================================
# OUTCOMES TOOLS
# =============================================================================

@tool("search_trial_outcomes", return_direct=False)
@robust_unwrap_llm_inputs
async def search_trial_outcomes(
    mode: str,
    nct_id: str | None = None,
    outcome_term: str | None = None,
    drug_name: str | None = None,
    outcome_type: str = "all",
    min_p_value: float | None = None,
    max_p_value: float | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Search trial outcomes, measurements, and statistical analyses.

    Modes:
        - "outcomes_for_trial": Outcomes and analyses for a specific NCT ID.
        - "trials_with_outcome": Trials matching an outcome keyword.
        - "efficacy_comparison": Analyses for trials involving a drug.
    """
    try:
        search_input = OutcomesSearchInput(
            mode=mode,
            nct_id=nct_id,
            outcome_keyword=outcome_term,
            drug_name=drug_name,
            outcome_type=outcome_type,
            min_p_value=min_p_value,
            max_p_value=max_p_value,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await outcomes_search_async(DEFAULT_CONFIG, search_input)
        return _format_outcomes_output(result)
    except Exception as e:
        return f"Error searching trial outcomes: {type(e).__name__}: {e}"


# =============================================================================
# ORANGE BOOK TOOLS
# =============================================================================

@tool("search_orange_book", return_direct=False)
@robust_unwrap_llm_inputs
async def search_orange_book(
    mode: str,
    drug_name: str | None = None,
    ingredient: str | None = None,
    nda_number: str | None = None,
    include_patents: bool = True,
    include_exclusivity: bool = True,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Search FDA Orange Book for products, TE codes, patents, and exclusivity.

    Modes:
        - "te_codes": TE codes and product summaries
        - "patents": Patents for matching products
        - "exclusivity": Exclusivity data for matching products
        - "generics": Products with patent/exclusivity details
    """
    try:
        search_input = OrangeBookSearchInput(
            mode=mode,
            drug_name=drug_name,
            ingredient=ingredient,
            nda_number=nda_number,
            include_patents=include_patents,
            include_exclusivity=include_exclusivity,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await orange_book_search_async(DEFAULT_CONFIG, search_input)
        return _format_orange_book_output(result)
    except Exception as e:
        return f"Error searching Orange Book: {type(e).__name__}: {e}"


# =============================================================================
# CROSS-DATABASE LOOKUP TOOLS
# =============================================================================

@tool("lookup_drug_identifiers", return_direct=False)
@robust_unwrap_llm_inputs
async def lookup_drug_identifiers(
    identifier: str,
    identifier_type: str = "auto",
    include_labels: bool = True,
    include_trials: bool = True,
    include_targets: bool = True,
    limit: int = 10,
) -> str:
    """
    Resolve a drug identifier across internal databases (ChEMBL, DrugCentral, labels, trials).
    """
    try:
        search_input = CrossDatabaseLookupInput(
            identifier=identifier,
            identifier_type=identifier_type,
            include_labels=include_labels,
            include_trials=include_trials,
            include_targets=include_targets,
            limit=min(limit, 200),
        )
        result = await cross_database_lookup_async(DEFAULT_CONFIG, search_input)
        return _format_cross_db_output(result)
    except Exception as e:
        return f"Error in cross-database lookup: {type(e).__name__}: {e}"


# =============================================================================
# BIOTHERAPEUTIC SEQUENCE TOOLS
# =============================================================================

@tool("search_biotherapeutics", return_direct=False)
@robust_unwrap_llm_inputs
async def search_biotherapeutics(
    mode: str,
    sequence: str | None = None,
    sequence_motif: str | None = None,
    target_gene: str | None = None,
    biotherapeutic_type: str = "all",
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Search biotherapeutics by sequence motif or target gene.

    Modes:
        - "by_sequence": Match by sequence motif (uses sequence/sequence_motif).
        - "similar_biologics": Find biologics sharing a motif (uses sequence/sequence_motif).
        - "by_target": Find biotherapeutics linked to a target gene.
    """
    try:
        sequence_value = sequence or sequence_motif
        search_input = BiotherapeuticSearchInput(
            mode=mode,
            sequence=sequence_value,
            target_gene=target_gene,
            biotherapeutic_type=biotherapeutic_type,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await biotherapeutic_sequence_search_async(DEFAULT_CONFIG, search_input)
        return _format_biotherapeutic_output(result)
    except Exception as e:
        return f"Error searching biotherapeutics: {type(e).__name__}: {e}"


# =============================================================================
# PHARMACOLOGY TOOLS (Drug-Target) - UPDATED
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
            Examples: "imatinib", "Gleevec", "aspirin", "ab-106"
        min_pchembl: Minimum pChEMBL value (potency threshold). 
            - 5.0 = 10 μM (weak, default)
            - 6.0 = 1 μM (moderate)
            - 7.0 = 100 nM (good)
            - 8.0 = 10 nM (potent)
            - 9.0 = 1 nM (very potent)
        data_source: Type of target data to return.
            - "both" (default): Activity data + curated mechanisms
            - "activity": Only quantitative assay data (IC50, Ki, etc.)
            - "mechanism": Only curated mechanism of action data
        include_all_organisms: If True, include targets from all species.
            Default (False) returns only human targets.
        limit: Maximum number of targets to return (default: 50).
    
    Returns:
        List of targets with gene symbols, activity values, mechanisms, 
        and data quality indicators.
    
    Examples:
        - search_drug_targets("imatinib")  # All targets of imatinib
        - search_drug_targets("aspirin", min_pchembl=6.0)  # Potent targets only
        - search_drug_targets("diazepam", data_source="mechanism")  # Mechanisms only
    """
    try:
        data_source_map = {
            "both": DataSource.BOTH,
            "activity": DataSource.ACTIVITY,
            "mechanism": DataSource.MECHANISM,
        }
        ds = data_source_map.get(data_source.lower(), DataSource.BOTH)
        
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
        return f"Error searching drug targets: {type(e).__name__}: {e}"


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
        gene_symbol: Gene symbol of the target protein.
            Examples: "EGFR", "ABL1", "JAK2", "BRAF", "CDK4", "ROS1", "NTRK1"
        min_pchembl: Minimum pChEMBL value (potency threshold).
            - 5.0 = 10 μM (weak, default)
            - 6.0 = 1 μM (moderate)  
            - 7.0 = 100 nM (good)
            - 8.0 = 10 nM (potent)
        data_source: Type of data to return.
            - "both" (default): Activity + mechanism data
            - "activity": Only quantitative assay data
            - "mechanism": Only curated mechanisms
        limit: Maximum results (default: 50).
    
    Returns:
        List of drugs/compounds with activity data and/or mechanisms.
    
    Examples:
        - search_target_drugs("EGFR")  # All EGFR modulators
        - search_target_drugs("JAK2", min_pchembl=7.0)  # Potent JAK2 inhibitors
        - search_target_drugs("ROS1", data_source="mechanism")  # Approved ROS1 drugs
    """
    try:
        data_source_map = {
            "both": DataSource.BOTH,
            "activity": DataSource.ACTIVITY,
            "mechanism": DataSource.MECHANISM,
        }
        ds = data_source_map.get(data_source.lower(), DataSource.BOTH)
        
        search_input = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_TARGET,
            query=gene_symbol.upper(),
            min_pchembl=min_pchembl,
            data_source=ds,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return f"Error searching target drugs: {type(e).__name__}: {e}"


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
    Useful for finding related compounds, potential drug candidates,
    or understanding structure-activity relationships.
    
    Args:
        smiles: SMILES string of the query molecule.
            Example: "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C" (imatinib)
        similarity_threshold: Minimum Tanimoto similarity (0.0 to 1.0).
            - 0.5 = distant analogs
            - 0.7 = similar scaffold (default)
            - 0.85 = close analogs
            - 0.95 = very similar (stereoisomers, salts)
        min_pchembl: Minimum potency for activity data (default: 5.0).
        limit: Maximum results (default: 50).
    
    Returns:
        List of similar molecules with similarity scores and activity data.
    
    Examples:
        - search_similar_molecules("CCO", similarity_threshold=0.5)  # Ethanol analogs
        - search_similar_molecules(imatinib_smiles, similarity_threshold=0.85)
    """
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
        return f"Error searching similar molecules: {type(e).__name__}: {e}"


@tool("search_exact_structure", return_direct=False)
@robust_unwrap_llm_inputs
async def search_exact_structure(
    smiles: str,
    min_pchembl: float = 5.0,
) -> str:
    """
    Find an exact structure match for a molecule.
    
    Use this when you have a SMILES and want to identify the compound
    and retrieve its target activity and mechanism data.
    
    Args:
        smiles: SMILES string of the molecule to identify.
        min_pchembl: Minimum potency for activity data (default: 5.0).
    
    Returns:
        Compound identification with activity and mechanism data if found.
    
    Examples:
        - search_exact_structure("CC(=O)Oc1ccccc1C(=O)O")  # Identify aspirin
        - search_exact_structure("Cc1nc(CNC(=O)NC2CCN(c3ncccc3Cl)C2)oc1C")
    """
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
        return f"Error in exact structure search: {type(e).__name__}: {e}"


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
    
    Note: This search can be slow for very common substructures.
    
    Args:
        pattern: SMILES or SMARTS pattern to search for.
            Examples:
            - "c1ccccc1" (benzene ring)
            - "C(=O)N" (amide)
            - "c1ccc2[nH]ccc2c1" (indole)
            - "[#7]1~[#6]~[#6]~[#7]~[#6]~[#6]1" (pyrimidine SMARTS)
        limit: Maximum results (default: 50).
    
    Returns:
        List of molecules containing the substructure.
    
    Examples:
        - search_substructure("c1ccc2[nH]ccc2c1")  # Indole-containing compounds
        - search_substructure("C(F)(F)F")  # Trifluoromethyl compounds
    """
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
        return f"Error in substructure search: {type(e).__name__}: {e}"


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
        - get_drug_profile("imatinib")
        - get_drug_profile("aspirin", include_trials=False)
    """
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
        return f"Error getting drug profile: {type(e).__name__}: {e}"


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
        - get_drug_forms("imatinib")  # Shows imatinib mesylate, etc.
        - get_drug_forms("metformin")  # Shows metformin HCl, etc.
    """
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
        return f"Error getting drug forms: {type(e).__name__}: {e}"


@tool("search_drug_trials", return_direct=False)
@robust_unwrap_llm_inputs
async def search_drug_trials(
    drug_name: str,
    limit: int = 50,
) -> str:
    """
    Find clinical trials for a drug.
    
    Searches ClinicalTrials.gov for trials involving the specified drug.
    Works for both small molecules and biologics.
    
    Args:
        drug_name: Name of the drug (brand, generic, or synonym).
            Examples: "imatinib", "pembrolizumab", "Keytruda"
        limit: Maximum results (default: 50).
    
    Returns:
        List of clinical trials with NCT IDs, titles, phases, and status.
    
    Examples:
        - search_drug_trials("imatinib")
        - search_drug_trials("pembrolizumab")
    """
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
        return f"Error searching drug trials: {type(e).__name__}: {e}"


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
        - compare_drugs_on_target("ABL1", ["imatinib", "dasatinib", "nilotinib"])
        - compare_drugs_on_target("EGFR", ["erlotinib", "gefitinib", "osimertinib"])
    """
    try:
        if isinstance(drug_names, str):
            drug_names = [drug_names]
        
        search_input = TargetSearchInput(
            mode=SearchMode.COMPARE_DRUGS,
            target=target.upper(),
            drug_names=drug_names,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return f"Error comparing drugs: {type(e).__name__}: {e}"


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
        off_targets: Gene symbols of targets to avoid.
            Examples: ["JAK1", "JAK3"] to find JAK2-selective compounds
        min_selectivity_fold: Minimum fold-selectivity required (default: 10x).
            A value of 10 means the drug must be 10x more potent on 
            the primary target vs off-targets.
        min_pchembl: Minimum potency on primary target (default: 6.0 = 1 μM).
        limit: Maximum results (default: 50).
    
    Returns:
        List of selective compounds with selectivity ratios.
    
    Examples:
        - search_selective_drugs("JAK2", ["JAK1", "JAK3"])  # JAK2-selective
        - search_selective_drugs("CDK4", ["CDK6"], min_selectivity_fold=100)
        - search_selective_drugs("EGFR", ["ERBB2", "ERBB4"])  # EGFR-selective
    """
    try:
        if isinstance(off_targets, str):
            off_targets = [off_targets]
        
        search_input = TargetSearchInput(
            mode=SearchMode.SELECTIVE_DRUGS,
            target=target.upper(),
            off_targets=[t.upper() for t in off_targets],
            min_selectivity_fold=min_selectivity_fold,
            min_pchembl=min_pchembl,
            limit=limit,
        )
        
        searcher = PharmacologySearch(DEFAULT_CONFIG)
        result = await searcher.search(search_input)
        return _format_pharmacology_output(result)
    
    except Exception as e:
        return f"Error searching selective drugs: {type(e).__name__}: {e}"


# =============================================================================
# TOOL COLLECTION
# =============================================================================

# Export all tools for easy registration
DBSEARCH_TOOLS = [
    # Clinical Trials
    search_clinical_trials,
    # Drug Labels
    search_drug_labels,
    # Molecule/Trial Connectivity
    search_molecule_trials,
    # Adverse Events
    search_adverse_events,
    # Outcomes
    search_trial_outcomes,
    # Orange Book
    search_orange_book,
    # Cross-Database Lookup
    lookup_drug_identifiers,
    # Biotherapeutics
    search_biotherapeutics,
    # Pharmacology
    search_drug_targets,
    search_target_drugs,
    search_similar_molecules,
    search_substructure,
    get_drug_profile,
    compare_drugs_on_target,
    search_selective_drugs,
]