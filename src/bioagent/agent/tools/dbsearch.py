# dbsearch.py
"""
Biomedical research tools for LangChain/LangGraph agents.

Exposes clinical trials, drug labels, and pharmacology data through
LLM-friendly interfaces with flat parameters and formatted outputs.
"""

from __future__ import annotations

import json
import inspect
from datetime import date
from functools import wraps

from langchain_core.tools import tool

from .tool_utils import build_handoff_signals

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
    SearchMode,
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
# INPUT VALIDATION HELPERS
# =============================================================================

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
# OUTPUT FORMATTERS
# =============================================================================

def _wrap_results(
    results_text: str,
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    signals = build_handoff_signals(source_tool or "", result_context or {})
    if not signals:
        signals = "[AGENT_SIGNALS]\n---\nRelated searches:\n  -> None"
    return f"[RESULTS]\n{results_text}\n\n{signals}"

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
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """
    Format clinical trials search results for LLM consumption.
    
    Args:
        output: The search output
        brief: If True, show brief summaries; if False, show full details
    """
    if output.status == "error":
        return _wrap_results(f"❌ Search failed: {output.error}", source_tool, result_context)
    
    if output.status == "not_found" or not output.hits:
        return _wrap_results((
            "🔍 No clinical trials found matching your criteria.\n\n"
            "Suggestions:\n"
            "- Try broader search terms\n"
            "- Check spelling of condition/drug names\n"
            "- Remove some filters to expand results"
        ), source_tool, result_context)
    
    lines = [
        f"✅ Found {output.total_hits} clinical trial(s)",
        f"Query: {output.query_summary}",
    ]
    
    if output.filters_applied:
        lines.append(f"Filters: {', '.join(output.filters_applied)}")
    
    if output.has_more:
        lines.append(f"(Showing {len(output.hits)} of {output.total_hits} - more available)")
    
    lines.append("=" * 60)
    
    brief = detail_level == "brief"
    for i, hit in enumerate(output.hits, 1):
        lines.append("")
        if brief:
            lines.append(_format_hit_brief(hit, i))
        else:
            lines.append(_format_hit_full(hit, i))
    
    return _wrap_results("\n".join(lines), source_tool, result_context)

def _format_drug_labels_output(
    output: DailyMedAndOpenFDASearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format drug label search results for LLM consumption."""
    if output.status == "error":
        return _wrap_results(f"Search failed: {output.error}", source_tool, result_context)
    
    if output.status == "not_found" or not output.results:
        return _wrap_results(
            "No drug information found. Try different drug names or search terms.",
            source_tool,
            result_context,
        )
    
    lines = [
        f"Found information for {len(output.results)} drug(s):",
        "=" * 40,
    ]
    
    if detail_level == "brief":
        max_section_len = 200
    elif detail_level == "comprehensive":
        max_section_len = None
    else:
        max_section_len = 1000

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
            if max_section_len is not None and len(text) > max_section_len:
                text = text[:max_section_len] + "... [truncated]"
            lines.append(f"\n  [{section.section_name}] (source: {section.source})")
            lines.append(f"    {text}")
    
    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_molecule_trials_output(
    output: MoleculeTrialSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format molecule-trial connectivity results."""
    if output.status == "error":
        return _wrap_results(f"❌ Molecule-trial search failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found" or not output.hits:
        return _wrap_results(
            f"🔍 No molecule-trial results found for: {output.query_summary}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    max_hits = len(output.hits) if detail_level == "comprehensive" else 50
    for i, hit in enumerate(output.hits[:max_hits], 1):
        if output.mode == "molecules_by_condition":
            name = getattr(hit, "concept_name", None) or "Unknown"
            lines.append(f"[{i}] {name} | Trials: {getattr(hit, 'n_trials', 0)}")
            continue

        if hasattr(hit, "group_label") and hasattr(hit, "trials"):
            group_label = getattr(hit, "group_label", "Unknown group")
            n_trials = getattr(hit, "n_trials", 0)
            lines.append(f"[{i}] Group: {group_label} | Trials: {n_trials}")
            if detail_level != "brief":
                group_trials = getattr(hit, "trials", []) or []
                max_group_trials = len(group_trials) if detail_level == "comprehensive" else min(10, len(group_trials))
                for trial in group_trials[:max_group_trials]:
                    trial_nct = getattr(trial, "nct_id", "N/A")
                    trial_phase = getattr(trial, "phase", None) or "N/A"
                    trial_status = getattr(trial, "status", None) or "N/A"
                    trial_title = getattr(trial, "brief_title", None) or ""
                    lines.append(f"    - {trial_nct} | {trial_phase} | {trial_status}")
                    if trial_title:
                        lines.append(f"      {trial_title}")
                if detail_level != "comprehensive" and len(group_trials) > max_group_trials:
                    lines.append(f"    ... and {len(group_trials) - max_group_trials} more trials in group")
            continue

        nct_id = getattr(hit, "nct_id", "N/A")
        phase = getattr(hit, "phase", None) or "N/A"
        status = getattr(hit, "status", None) or "N/A"
        enrollment = getattr(hit, "enrollment", None)
        start_date = getattr(hit, "start_date", None)
        completion_date = getattr(hit, "completion_date", None)
        lead_sponsor = getattr(hit, "lead_sponsor", None)
        head = f"[{i}] {nct_id} | {phase} | {status}"
        if enrollment is not None:
            head += f" | n={enrollment:,}"
        lines.append(head)

        if detail_level != "brief":
            title = getattr(hit, "brief_title", None)
            if title:
                lines.append(f"    {title}")

            concept_name = getattr(hit, "concept_name", None) or getattr(hit, "molecule_name", None)
            if concept_name:
                lines.append(f"    Molecule: {concept_name}")

            if start_date or completion_date or lead_sponsor:
                meta_parts = []
                if start_date:
                    meta_parts.append(f"Start: {start_date}")
                if completion_date:
                    meta_parts.append(f"Completion: {completion_date}")
                if lead_sponsor:
                    meta_parts.append(f"Sponsor: {_truncate(lead_sponsor, 50)}")
                lines.append(f"    {' | '.join(meta_parts)}")

            conditions = getattr(hit, "conditions", None)
            if conditions and isinstance(conditions, list) and len(conditions) > 0:
                cond_preview = ", ".join(conditions[:3])
                if len(conditions) > 3:
                    cond_preview += f" (+{len(conditions) - 3} more)"
                lines.append(f"    Conditions: {cond_preview}")

            match_type = getattr(hit, "match_type", None)
            confidence = getattr(hit, "confidence", None)
            if match_type or confidence is not None:
                conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
                lines.append(f"    Match: {match_type or 'N/A'} | Confidence: {conf_text}")
            target_evidence = getattr(hit, "target_evidence", None)
            if target_evidence and isinstance(target_evidence, list):
                lines.append(f"    Target evidence: {', '.join(target_evidence)}")

            inchi_key = getattr(hit, "inchi_key", None)
            if inchi_key:
                lines.append(f"    InChIKey: {inchi_key}")

            rendered_summary = getattr(hit, "rendered_summary", None)
            if detail_level == "comprehensive" and rendered_summary:
                summary = rendered_summary[:2000] + ("..." if len(rendered_summary) > 2000 else "")
                lines.append(f"    [Study summary]\n    " + summary.replace("\n", "\n    "))

    if detail_level != "comprehensive" and output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_adverse_events_output(
    output: AdverseEventsSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format adverse events search results."""
    if output.status == "error":
        return _wrap_results(f"❌ Adverse event search failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found" or not output.hits:
        return _wrap_results(
            f"🔍 No adverse event results found for: {output.query_summary}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    max_hits = len(output.hits) if detail_level == "comprehensive" else 50
    for i, hit in enumerate(output.hits[:max_hits], 1):
        if output.mode == "events_for_drug":
            term = getattr(hit, "adverse_event_term", None) or "Unknown event"
            event_type = getattr(hit, "event_type", None) or "N/A"
            affected = getattr(hit, "subjects_affected", None)
            at_risk = getattr(hit, "subjects_at_risk", None)
            n_trials = getattr(hit, "n_trials", None)
            lines.append(f"[{i}] {term} | {event_type}")
            if detail_level != "brief":
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
            if detail_level != "brief":
                if title:
                    lines.append(f"    {title}")
                if affected is not None:
                    lines.append(f"    Subjects affected: {affected}")
            continue

        # compare_safety
        drug_name = getattr(hit, "drug_name", None) or "Unknown drug"
        lines.append(f"[{i}] {drug_name}")
        if detail_level != "brief":
            top_events = getattr(hit, "top_events", []) or []
            for ev in top_events[:5]:
                term = getattr(ev, "adverse_event_term", None) or "Unknown event"
                event_type = getattr(ev, "event_type", None) or "N/A"
                affected = getattr(ev, "subjects_affected", None)
                n_trials = getattr(ev, "n_trials", None)
                lines.append(
                    f"    - {term} | {event_type} | affected: {affected if affected is not None else 'N/A'} | trials: {n_trials if n_trials is not None else 'N/A'}"
                )

    if detail_level != "comprehensive" and output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_outcomes_output(
    output: OutcomesSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format clinical trial outcomes search results."""
    if output.status == "error":
        return _wrap_results(f"❌ Outcomes search failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found" or not output.hits:
        return _wrap_results(
            f"🔍 No outcome results found for: {output.query_summary}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    if output.mode == "outcomes_for_trial":
        max_bundles = len(output.hits) if detail_level == "comprehensive" else 5
        for bundle in output.hits[:max_bundles]:
            nct_id = getattr(bundle, "nct_id", "N/A")
            outcomes = getattr(bundle, "outcomes", []) or []
            measurements = getattr(bundle, "measurements", []) or []
            analyses = getattr(bundle, "analyses", []) or []
            lines.append(f"Trial {nct_id}: outcomes={len(outcomes)}, measurements={len(measurements)}, analyses={len(analyses)}")
            if detail_level != "brief":
                for outcome in outcomes[:5]:
                    title = getattr(outcome, "title", None) or "Unnamed outcome"
                    outcome_type = getattr(outcome, "outcome_type", None) or "N/A"
                    lines.append(f"    - [{outcome_type}] {title}")

                for analysis in analyses[:5]:
                    title = getattr(analysis, "outcome_title", None) or "Outcome analysis"
                    p_value = getattr(analysis, "p_value", None)
                    p_text = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
                    lines.append(f"    * Analysis: {title} | p={p_text}")
        return _wrap_results("\n".join(lines), source_tool, result_context)

    max_hits = len(output.hits) if detail_level == "comprehensive" else 50
    for i, hit in enumerate(output.hits[:max_hits], 1):
        if output.mode == "trials_with_outcome":
            nct_id = getattr(hit, "nct_id", "N/A")
            title = getattr(hit, "brief_title", None) or ""
            outcome_title = getattr(hit, "outcome_title", None) or "Outcome"
            outcome_type = getattr(hit, "outcome_type", None) or "N/A"
            lines.append(f"[{i}] {nct_id} | {outcome_type} | {outcome_title}")
            if detail_level != "brief" and title:
                lines.append(f"    {title}")
            continue

        # efficacy_comparison
        nct_id = getattr(hit, "nct_id", "N/A")
        outcome_title = getattr(hit, "outcome_title", None) or "Outcome"
        outcome_type = getattr(hit, "outcome_type", None) or "N/A"
        p_value = getattr(hit, "p_value", None)
        p_text = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
        lines.append(f"[{i}] {nct_id} | {outcome_type} | {outcome_title} | p={p_text}")

    if detail_level != "comprehensive" and output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_orange_book_output(
    output: OrangeBookSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format Orange Book search results."""
    if output.status == "error":
        return _wrap_results(f"❌ Orange Book search failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found" or not output.hits:
        return _wrap_results(
            f"🔍 No Orange Book results found for: {output.query_summary}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    max_hits = len(output.hits) if detail_level == "comprehensive" else 50
    for i, hit in enumerate(output.hits[:max_hits], 1):
        trade = getattr(hit, "trade_name", None) or "Unknown"
        ingredient = getattr(hit, "ingredient", None) or "Unknown ingredient"
        te_code = getattr(hit, "te_code", None) or "N/A"
        appl_no = getattr(hit, "appl_no", None) or "N/A"
        lines.append(f"[{i}] {trade} | {ingredient} | TE: {te_code} | NDA: {appl_no}")
        if detail_level != "brief":
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

    if detail_level != "comprehensive" and output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_cross_db_output(
    output: CrossDatabaseLookupOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format cross-database identifier lookup results."""
    if output.status == "error":
        return _wrap_results(f"❌ Identifier lookup failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found":
        return _wrap_results(
            f"🔍 No matches found for: {output.identifier}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Identifier lookup: {output.identifier} ({output.identifier_type})",
        f"Molecules: {len(output.molecules)} | Labels: {len(output.labels)} | Trials: {len(output.trials)} | Targets: {len(output.targets)}",
        "=" * 40,
    ]

    max_molecules = len(output.molecules) if detail_level == "comprehensive" else 10
    for i, mol in enumerate(output.molecules[:max_molecules], 1):
        name = mol.concept_name or mol.pref_name or "Unknown"
        chembl = mol.chembl_id or "N/A"
        inchi = mol.inchi_key or "N/A"
        lines.append(f"[{i}] {name} | ChEMBL: {chembl} | InChIKey: {inchi}")

    if detail_level != "brief" and output.labels:
        lines.append("Labels:")
        max_labels = len(output.labels) if detail_level == "comprehensive" else 10
        for label in output.labels[:max_labels]:
            lines.append(f"  - {label.title or 'Label'} ({label.set_id}) [{label.source}]")

    if detail_level != "brief" and output.trials:
        lines.append("Trials:")
        max_trials = len(output.trials) if detail_level == "comprehensive" else 10
        for trial in output.trials[:max_trials]:
            lines.append(f"  - {trial.nct_id} | {trial.phase or 'N/A'} | {trial.status or 'N/A'}")

    if detail_level != "brief" and output.targets:
        lines.append("Targets:")
        max_targets = len(output.targets) if detail_level == "comprehensive" else 10
        for target in output.targets[:max_targets]:
            pchembl = target.best_pchembl
            p_text = f"{pchembl:.2f}" if isinstance(pchembl, (int, float)) else "N/A"
            lines.append(f"  - {target.gene_symbol} | best pChEMBL: {p_text} | n={target.n_measurements or 0}")

    return _wrap_results("\n".join(lines), source_tool, result_context)


def _format_biotherapeutic_output(
    output: BiotherapeuticSearchOutput,
    detail_level: str = "standard",
    source_tool: str | None = None,
    result_context: dict | None = None,
) -> str:
    """Format biotherapeutic sequence search results."""
    if output.status == "error":
        return _wrap_results(f"❌ Biotherapeutic search failed: {output.error}", source_tool, result_context)
    if output.status == "invalid_input":
        return _wrap_results(f"❌ Invalid input: {output.error}", source_tool, result_context)
    if output.status == "not_found" or not output.hits:
        return _wrap_results(
            f"🔍 No biotherapeutic results found for: {output.query_summary}",
            source_tool,
            result_context,
        )

    lines = [
        f"✅ Found {output.total_hits} result(s)",
        f"Mode: {output.mode}",
        f"Query: {output.query_summary}",
        "=" * 40,
    ]

    max_hits = len(output.hits) if detail_level == "comprehensive" else 50
    for i, hit in enumerate(output.hits[:max_hits], 1):
        name = getattr(hit, "pref_name", None) or "Unknown"
        chembl = getattr(hit, "chembl_id", None) or "N/A"
        bio_type = getattr(hit, "biotherapeutic_type", None) or "N/A"
        organism = getattr(hit, "organism", None) or "N/A"
        lines.append(f"[{i}] {name} | {bio_type} | ChEMBL: {chembl} | {organism}")
        if detail_level != "brief":
            components = getattr(hit, "components", []) or []
            for comp in components[:5]:
                comp_type = getattr(comp, "component_type", None) or "component"
                accession = getattr(comp, "uniprot_accession", None) or "N/A"
                seq_len = getattr(comp, "sequence_length", None)
                seq_text = f"{seq_len} aa" if seq_len else "N/A"
                lines.append(f"    - {comp_type} | UniProt: {accession} | {seq_text}")

    if detail_level != "comprehensive" and output.total_hits > 50:
        lines.append(f"\n... and {output.total_hits - 50} more results (truncated)")

    return _wrap_results("\n".join(lines), source_tool, result_context)


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
    eligibility_age_range: list[int] | str | None = None,
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

    # === Detail Level ===
    detail_level: str = "standard",
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
        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
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
            if isinstance(eligibility_age_range, list) and len(eligibility_age_range) == 2:
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
        
        if detail_level == "brief":
            brief_output = True
            include_results = False
            include_adverse_events = False
            include_eligibility = False
            include_groups = False
            include_baseline = False
            include_sponsors = True
            include_countries = False
        elif detail_level == "comprehensive":
            brief_output = False
            include_results = True
            include_adverse_events = True
            include_baseline = True

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
        
        top_nct_id = result.hits[0].nct_id if result.hits else None
        return _format_clinical_trials_output(
            result,
            detail_level=detail_level,
            source_tool="search_clinical_trials",
            result_context={"top_nct_id": top_nct_id},
        )
    
    except Exception as e:
        return f"❌ Error searching clinical trials: {type(e).__name__}: {e}"


# =============================================================================
# CLINICAL TRIAL DETAILS TOOL
# =============================================================================

@tool("get_clinical_trial_details", return_direct=False)
@robust_unwrap_llm_inputs
async def get_clinical_trial_details(
    nct_ids: str,
    detail_level: str = "comprehensive",
) -> str:
    """
    Retrieve detailed information for specific ClinicalTrials.gov NCT IDs.

    Use this when you already know one or more NCT IDs and want full trial details.

    Args:
        nct_ids: Comma-separated NCT IDs (e.g., "NCT01295827, NCT04280705").
        detail_level: "brief", "standard", or "comprehensive" (default: comprehensive).

    Returns:
        Detailed trial information for each NCT ID.
    """
    try:
        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"

        if not nct_ids or not isinstance(nct_ids, str):
            return "❌ Invalid input: nct_ids is required (comma-separated NCT IDs)."

        nct_list = [n.strip().upper() for n in nct_ids.split(",") if n.strip()]
        if not nct_list:
            return "❌ Invalid input: Provide at least one valid NCT ID."
        if len(nct_list) > 10:
            return "❌ Invalid input: Maximum of 10 NCT IDs per request."

        invalid = [n for n in nct_list if not (n.startswith("NCT") and len(n) == 11 and n[3:].isdigit())]
        if invalid:
            return f"❌ Invalid input: Invalid NCT ID(s): {', '.join(invalid)}"

        include_results = detail_level == "comprehensive"
        include_adverse_events = detail_level == "comprehensive"
        include_baseline = detail_level == "comprehensive"
        include_eligibility = detail_level != "brief"
        include_groups = detail_level != "brief"
        include_sponsors = True
        include_countries = detail_level != "brief"

        search_input = ClinicalTrialsSearchInput(
            nct_ids=nct_list,
            sort_by=SortField.NCT_ID,
            sort_order=SortOrder.ASC,
            limit=min(len(nct_list), 10),
            offset=0,
            include_study_json=True,
            output_eligibility=include_eligibility,
            output_groups=include_groups,
            output_baseline_measurements=include_baseline,
            output_results=include_results,
            output_adverse_effects=include_adverse_events,
            output_sponsors=include_sponsors,
            output_countries=include_countries,
        )

        result = await clinical_trials_search_async(DEFAULT_CONFIG, search_input)
        return _format_clinical_trials_output(
            result,
            detail_level=detail_level,
            source_tool="get_clinical_trial_details",
            result_context={"top_nct_id": nct_list[0] if nct_list else None},
        )
    except Exception as e:
        return f"❌ Error retrieving trial details: {type(e).__name__}: {e}"


# =============================================================================
# LOOKUP / ROUTER TOOL
# =============================================================================

@tool("check_data_availability", return_direct=False)
@robust_unwrap_llm_inputs
async def check_data_availability(
    entity: str,
    entity_type: str = "auto",
) -> str:
    """
    Check what data sources contain information for an entity.

    Args:
        entity: Drug name, gene symbol, NCT ID, SMILES, or condition.
        entity_type: auto | drug | gene | nct_id | smiles | condition

    Returns:
        A summary of available data sources and suggested next steps.
    """
    if not entity or not isinstance(entity, str) or not entity.strip():
        return "❌ Invalid input: entity is required."

    entity = entity.strip()
    allowed_types = {"auto", "drug", "gene", "nct_id", "smiles", "condition"}
    if entity_type not in allowed_types:
        return "❌ Invalid input: entity_type must be one of auto, drug, gene, nct_id, smiles, condition"

    detected = entity_type
    if entity_type == "auto":
        upper = entity.upper()
        if upper.startswith("NCT") and len(upper) == 11 and upper[3:].isdigit():
            detected = "nct_id"
        elif any(ch in entity for ch in "[]()=#@") and len(entity) > 6:
            detected = "smiles"
        elif entity.isupper() and 2 <= len(entity) <= 10 and entity.replace("-", "").isalnum():
            detected = "gene"
        else:
            detected = "drug"

    lines = [
        f"Entity: {entity}",
        f"Detected type: {detected}",
        "",
        "Available data:",
    ]

    has_targets = False
    has_trials = False
    has_labels = False
    has_orange_book = False

    try:
        if detected in {"drug", "smiles"}:
            mode = SearchMode.TARGETS_FOR_DRUG if detected == "drug" else SearchMode.EXACT_STRUCTURE
            search_input = TargetSearchInput(
                mode=mode,
                query=entity if detected == "drug" else None,
                smiles=entity if detected == "smiles" else None,
                limit=1,
            )
            searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=False)
            result = await searcher.search(search_input)
            has_targets = bool(result.hits)
            lines.append(f"  Pharmacology targets: {'Yes' if has_targets else 'No'}")

        if detected in {"drug", "condition", "nct_id"}:
            trial_input = ClinicalTrialsSearchInput(
                intervention=entity if detected == "drug" else None,
                condition=entity if detected == "condition" else None,
                nct_ids=[entity] if detected == "nct_id" else None,
                limit=1,
                include_study_json=False,
                output_results=False,
                output_adverse_effects=False,
                output_baseline_measurements=False,
            )
            trial_output = await clinical_trials_search_async(DEFAULT_CONFIG, trial_input)
            has_trials = bool(trial_output.hits)
            lines.append(f"  Clinical trials: {'Yes' if has_trials else 'No'}")

        if detected == "drug":
            label_input = DailyMedAndOpenFDAInput(
                drug_names=[entity],
                section_queries=None,
                keyword_query=None,
                fetch_all_sections=False,
                aggressive_deduplication=True,
                result_limit=1,
                top_n_drugs=1,
                sections_per_query=1,
            )
            label_output = await dailymed_and_openfda_search_async(DEFAULT_CONFIG, label_input)
            has_labels = bool(label_output.results)
            lines.append(f"  FDA drug labels: {'Yes' if has_labels else 'No'}")

            orange_input = OrangeBookSearchInput(
                mode="te_codes",
                drug_name=entity,
                limit=1,
                offset=0,
            )
            orange_output = await orange_book_search_async(DEFAULT_CONFIG, orange_input)
            has_orange_book = bool(orange_output.hits)
            lines.append(f"  Orange Book: {'Yes' if has_orange_book else 'No'}")

        if detected == "gene":
            gene_input = TargetSearchInput(
                mode=SearchMode.DRUGS_FOR_TARGET,
                query=entity,
                limit=1,
            )
            searcher = PharmacologySearch(DEFAULT_CONFIG, verbose=False)
            gene_result = await searcher.search(gene_input)
            has_targets = bool(gene_result.hits)
            lines.append(f"  Target-linked drugs: {'Yes' if has_targets else 'No'}")

    except Exception as exc:
        lines.append(f"  ⚠ Availability check error: {type(exc).__name__}: {exc}")

    result_context = {
        "drug_name": entity if detected == "drug" else None,
        "has_targets": has_targets,
        "has_trials": has_trials,
        "has_labels": has_labels,
    }
    return _wrap_results("\n".join(lines), source_tool="check_data_availability", result_context=result_context)


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
    detail_level: str = "standard",
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
        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"Invalid search parameters: {error}"
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
        primary_drug = None
        if isinstance(drug_names, list) and drug_names:
            primary_drug = drug_names[0]
        elif isinstance(drug_names, str):
            primary_drug = drug_names
        return _format_drug_labels_output(
            result,
            detail_level=detail_level,
            source_tool="search_drug_labels",
            result_context={"drug_name": primary_drug},
        )
    
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
    sequence: str | None = None,
    smiles: str | None = None,
    smarts: str | None = None,
    similarity_threshold: float = 0.7,
    min_pchembl: float = 6.0,
    phase: list[str] | str | None = None,
    status: list[str] | str | None = None,
    molecule_type: str = "all",
    limit: int = 20,
    offset: int = 0,
    detail_level: str = "standard",
    group_by: str = "none",
    include_activity: bool = False,
    include_study_details: bool = False,
    start_date_from: str | None = None,
    start_date_to: str | None = None,
    completion_date_from: str | None = None,
    completion_date_to: str | None = None,
    min_enrollment: int | None = None,
    max_enrollment: int | None = None,
    lead_sponsor: str | None = None,
    country: list[str] | str | None = None,
    has_results: bool | None = None,
    is_fda_regulated: bool | None = None,
) -> str:
    """
    Search clinical trials linked to molecules, conditions, targets, or structures.
    
    Links small molecules to clinical trials using mapping tables and
    concept identifiers. Supports molecule-centric, condition-centric,
    target-centric, and structure-centric queries.
    
    ═══════════════════════════════════════════════════════════════════════════════
    SEARCH MODES
    ═══════════════════════════════════════════════════════════════════════════════
    
    mode: Required. One of:
        - "trials_by_molecule": Find trials linked to a molecule name or InChIKey.
            Requires: molecule (name) or inchi_key
            Example: search_molecule_trials(mode="trials_by_molecule", molecule="imatinib")
        
        - "molecules_by_condition": Find molecules being studied for a condition.
            Requires: condition (disease/condition name)
            Example: search_molecule_trials(mode="molecules_by_condition", condition="breast cancer")
        
        - "trials_by_target": Find trials for drugs that modulate a specific target gene.
            Requires: target_gene (gene symbol like "EGFR", "ABL1")
            Example: search_molecule_trials(mode="trials_by_target", target_gene="EGFR")
            Use this to go from a target/gene → clinical trials

        - "trials_by_sequence": Find trials for biologics matching a sequence motif.
            Requires: sequence (amino acid motif)
            Example: search_molecule_trials(mode="trials_by_sequence", sequence="EVQLVESGG")
        
        - "trials_by_structure": Find trials for molecules similar to a query SMILES.
            Requires: smiles (SMILES string)
            Example: search_molecule_trials(mode="trials_by_structure", smiles="CC(=O)Oc1ccccc1C(=O)O")
        
        - "trials_by_substructure": Find trials for molecules containing a substructure.
            Requires: smiles or smarts (substructure pattern)
            Example: search_molecule_trials(mode="trials_by_substructure", smiles="c1ccccc1")
    
    ═══════════════════════════════════════════════════════════════════════════════
    PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    molecule: Molecule name (preferred name, synonym, or brand name).
        Examples: "imatinib", "Gleevec", "pembrolizumab"
    
    inchi_key: InChIKey for precise molecule lookup (27-character identifier).
        Example: "KTUFNOKKBVMGRW-UHFFFAOYSA-N" (imatinib)
    
    condition: Condition/disease name for discovering molecules.
        Examples: "breast cancer", "rheumatoid arthritis", "type 2 diabetes"
    
    target_gene: Gene symbol of the target protein (HGNC official symbol).
        Examples: "EGFR", "ABL1", "JAK2", "BRAF", "CDK4", "ROS1"
        Use this to find trials for drugs targeting a specific protein.

    sequence: Sequence motif for biologic searches (e.g., antibody framework motif).
        Required for: trials_by_sequence
    
    smiles: SMILES string for structure-based searches.
        Required for: trials_by_structure, trials_by_substructure
    
    smarts: SMARTS pattern for substructure search (alternative to smiles).
        Example: "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1" (benzene ring)
    
    similarity_threshold: Minimum Tanimoto similarity for structure search (0.0-1.0).
        Default: 0.7. Lower = more results, higher = stricter matching.
    
    min_pchembl: Potency threshold for target-linked mode (default: 6.0 = 1 μM).
        - 5.0 = 10 μM (weak, more results)
        - 6.0 = 1 μM (moderate, default)
        - 7.0 = 100 nM (good)
        - 8.0 = 10 nM (potent, fewer results)
    
    phase: Trial phase filter (single or comma-separated list).
        Examples: "Phase 3", "Phase 2, Phase 3"
    
    status: Trial status filter (single or comma-separated list).
        Examples: "Recruiting", "Completed"

    molecule_type: Filter by molecule type.
        - "all" (default): Small molecules + biotherapeutics
        - "small_molecule": Only small molecules
        - "biotherapeutic": Only biologics
    
    limit: Maximum results (default: 20, max: 500).
    
    offset: Offset for pagination (default: 0).
    
    include_activity: For trials_by_target, also include compound-target activity (default: False).

    group_by: Optional grouping for trial result modes.
        For mode="trials_by_molecule":
        - "none" (default): Trial-by-trial output
        - "condition": Group trials by condition text
        - "molecule_concept": Group trials by molecule concept
        - "intervention": Group trials by intervention text
        For mode="trials_by_target":
        - "none" (default): Trial-by-trial output
        - "condition": Group trials by condition text
        - "molecule_concept": Group by matched molecule concept (from map_ctgov_molecules.concept_id)
        - "intervention": Group trials by intervention text
    
    include_study_details: Include full study summary text (default: False).
    
    start_date_from, start_date_to: Filter by trial start date (YYYY-MM-DD).
    
    completion_date_from, completion_date_to: Filter by completion date (YYYY-MM-DD).
    
    min_enrollment, max_enrollment: Filter by enrollment count.
    
    lead_sponsor: Filter by sponsor name (substring match).
    
    country: Filter by country/countries (list or comma-separated).
    
    has_results: Filter by results submitted (True/False).
    
    is_fda_regulated: Filter by FDA-regulated drug (True/False).
    
    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLES
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. Find trials for a specific drug:
       search_molecule_trials(mode="trials_by_molecule", molecule="imatinib")

    1b. Find trials for a drug grouped by condition:
       search_molecule_trials(mode="trials_by_molecule", molecule="crizotinib", group_by="condition")
    
    2. Find drugs being studied for a condition:
       search_molecule_trials(mode="molecules_by_condition", condition="lung cancer")
    
    3. Find trials for drugs targeting EGFR:
       search_molecule_trials(mode="trials_by_target", target_gene="EGFR", phase="Phase 3")

    3b. Find EGFR-targeted trials grouped by intervention:
       search_molecule_trials(mode="trials_by_target", target_gene="EGFR", group_by="intervention")
    
    4. Find trials for structurally similar molecules:
       search_molecule_trials(mode="trials_by_structure", smiles="Cc1ccc(cc1)NC(=O)...", similarity_threshold=0.7)
    
    5. Find trials for molecules with a specific substructure:
       search_molecule_trials(mode="trials_by_substructure", smiles="c1ccc2[nH]ccc2c1")

    6. Find trials for antibody sequence motifs:
       search_molecule_trials(mode="trials_by_sequence", sequence="EVQLVESGG", molecule_type="biotherapeutic")
    
    Returns:
        Trials linked to molecules with NCT IDs, phases, status, and match confidence.
    """
    try:
        valid_modes = {
            "trials_by_molecule",
            "molecules_by_condition",
            "trials_by_target",
            "trials_by_sequence",
            "trials_by_structure",
            "trials_by_substructure",
        }
        if not mode or mode not in valid_modes:
            return (
                "❌ Invalid input: mode is required.\n"
                "Valid options: trials_by_molecule, molecules_by_condition, trials_by_target, "
                "trials_by_sequence, trials_by_structure, trials_by_substructure"
            )

        if mode == "trials_by_molecule" and not (molecule or inchi_key):
            return (
                "❌ Invalid input: molecule or inchi_key is required for mode='trials_by_molecule'.\n"
                "Example: search_molecule_trials(mode='trials_by_molecule', molecule='imatinib')"
            )
        allowed_group_by = {"none", "condition", "molecule_concept", "intervention"}
        group_by = (group_by or "none").strip().lower()
        if group_by not in allowed_group_by:
            return (
                "❌ Invalid input: group_by must be one of: none, condition, molecule_concept, intervention.\n"
                "Example: search_molecule_trials(mode='trials_by_molecule', molecule='crizotinib', group_by='condition')"
            )
        if mode == "molecules_by_condition" and not condition:
            return (
                "❌ Invalid input: condition is required for mode='molecules_by_condition'.\n"
                "Example: search_molecule_trials(mode='molecules_by_condition', condition='breast cancer')"
            )
        if mode == "trials_by_target" and not target_gene:
            return (
                "❌ Invalid input: target_gene is required for mode='trials_by_target'.\n"
                "Example: search_molecule_trials(mode='trials_by_target', target_gene='EGFR')"
            )
        if mode == "trials_by_sequence" and not sequence:
            return (
                "❌ Invalid input: sequence is required for mode='trials_by_sequence'.\n"
                "Example: search_molecule_trials(mode='trials_by_sequence', sequence='EVQLVESGG')"
            )
        if mode == "trials_by_structure" and not smiles:
            return (
                "❌ Invalid input: smiles is required for mode='trials_by_structure'.\n"
                "Example: search_molecule_trials(mode='trials_by_structure', smiles='CC(=O)Oc1ccccc1C(=O)O')"
            )
        if mode == "trials_by_substructure" and not (smiles or smarts):
            return (
                "❌ Invalid input: smiles or smarts is required for mode='trials_by_substructure'.\n"
                "Example: search_molecule_trials(mode='trials_by_substructure', smarts='c1ccccc1')"
            )

        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"Invalid input: {error}"
        if isinstance(phase, str):
            phase_list = [p.strip() for p in phase.split(",") if p.strip()]
        elif isinstance(phase, list):
            phase_list = [p.strip() for p in phase if isinstance(p, str) and p.strip()]
        else:
            phase_list = None

        if isinstance(status, str):
            status_list = [s.strip() for s in status.split(",") if s.strip()]
        elif isinstance(status, list):
            status_list = [s.strip() for s in status if isinstance(s, str) and s.strip()]
        else:
            status_list = None

        def _parse_date(s: str | None) -> date | None:
            if not s or not isinstance(s, str):
                return None
            try:
                return date.fromisoformat(s.strip())
            except (ValueError, TypeError):
                return None

        country_list: list[str] | None = None
        if country is not None:
            if isinstance(country, str):
                country_list = [c.strip() for c in country.split(",") if c.strip()]
            elif isinstance(country, list):
                country_list = [c.strip() for c in country if isinstance(c, str) and c.strip()]
            if not country_list:
                country_list = None

        search_input = MoleculeTrialSearchInput(
            mode=mode,
            molecule_name=molecule,
            inchi_key=inchi_key,
            target_gene=target_gene,
            condition=condition,
            sequence=sequence,
            smiles=smiles,
            smarts=smarts,
            similarity_threshold=similarity_threshold,
            min_pchembl=min_pchembl,
            phase=phase_list,
            status=status_list,
            molecule_type=molecule_type,
            limit=min(limit, 500),
            offset=offset,
            group_by=group_by,
            include_activity_data=include_activity,
            include_study_details=include_study_details,
            start_date_from=_parse_date(start_date_from),
            start_date_to=_parse_date(start_date_to),
            completion_date_from=_parse_date(completion_date_from),
            completion_date_to=_parse_date(completion_date_to),
            min_enrollment=min_enrollment,
            max_enrollment=max_enrollment,
            lead_sponsor=lead_sponsor.strip() if isinstance(lead_sponsor, str) and lead_sponsor else None,
            country=country_list,
            has_results=has_results,
            is_fda_regulated=is_fda_regulated,
        )
        result = await molecule_trial_search_async(DEFAULT_CONFIG, search_input)
        return _format_molecule_trials_output(
            result,
            detail_level=detail_level,
            source_tool="search_molecule_trials",
            result_context={"molecule_name": molecule},
        )
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
    detail_level: str = "standard",
) -> str:
    """
    Search ClinicalTrials.gov adverse event data and safety signals.
    
    Provides safety signal summaries and trial-level adverse event data for
    drug- and event-centric queries. Joins reported event tables to the 
    trial search index.
    
    ═══════════════════════════════════════════════════════════════════════════════
    SEARCH MODES
    ═══════════════════════════════════════════════════════════════════════════════
    
    mode: Required. One of:
        - "events_for_drug": Summarize the most frequent adverse events for a drug.
            Requires: drug_name
            Returns: Adverse event terms ranked by subject count
            Example: search_adverse_events(mode="events_for_drug", drug_name="imatinib")
        
        - "drugs_with_event": Find trials reporting a specific adverse event term.
            Requires: event_term
            Returns: Trials that reported the specified event
            Example: search_adverse_events(mode="drugs_with_event", event_term="neutropenia")
        
        - "compare_safety": Compare adverse event profiles across multiple drugs.
            Requires: drug_names (list of at least 2 drugs)
            Returns: Top events for each drug side-by-side
            Example: search_adverse_events(mode="compare_safety", drug_names=["imatinib", "dasatinib"])
    
    ═══════════════════════════════════════════════════════════════════════════════
    PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    drug_name: Drug/intervention name for events_for_drug mode.
        Examples: "imatinib", "pembrolizumab", "aspirin"
    
    event_term: Adverse event term to search for in drugs_with_event mode.
        Examples: "neutropenia", "nausea", "headache", "cardiac arrest"
    
    drug_names: List of drug names for compare_safety mode (at least 2 required).
        Examples: ["imatinib", "dasatinib"], ["aspirin", "ibuprofen", "naproxen"]
    
    severity: Filter by event type/severity.
        - "all" (default): All event types
        - "serious": Only serious adverse events
        - "other": Only non-serious events
    
    min_subjects_affected: Minimum number of subjects affected to include event.
        Use this to filter out rare events and focus on common ones.
    
    limit: Maximum results per query (default: 20, max: 500).
    
    offset: Offset for pagination (default: 0).
    
    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLES
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. Get top adverse events for imatinib:
       search_adverse_events(mode="events_for_drug", drug_name="imatinib")
    
    2. Find trials reporting neutropenia:
       search_adverse_events(mode="drugs_with_event", event_term="neutropenia")
    
    3. Compare safety profiles of BCR-ABL inhibitors:
       search_adverse_events(
           mode="compare_safety", 
           drug_names=["imatinib", "dasatinib", "nilotinib"]
       )
    
    4. Get serious adverse events for a drug:
       search_adverse_events(mode="events_for_drug", drug_name="warfarin", severity="serious")
    
    Returns:
        Adverse event summaries with subject counts, event types, and trial associations.
    """
    try:
        valid_modes = {"events_for_drug", "drugs_with_event", "compare_safety"}
        if not mode or mode not in valid_modes:
            return (
                "❌ Invalid input: mode is required.\n"
                "Valid options: events_for_drug, drugs_with_event, compare_safety"
            )
        if mode == "events_for_drug" and not drug_name:
            return (
                "❌ Invalid input: drug_name is required for mode='events_for_drug'.\n"
                "Example: search_adverse_events(mode='events_for_drug', drug_name='imatinib')"
            )
        if mode == "drugs_with_event" and not event_term:
            return (
                "❌ Invalid input: event_term is required for mode='drugs_with_event'.\n"
                "Example: search_adverse_events(mode='drugs_with_event', event_term='neutropenia')"
            )
        if mode == "compare_safety":
            if isinstance(drug_names, str):
                drug_names = [drug_names]
            if not drug_names or len(drug_names) < 2:
                return (
                    "❌ Invalid input: drug_names (at least 2) is required for mode='compare_safety'.\n"
                    "Example: search_adverse_events(mode='compare_safety', drug_names=['imatinib','dasatinib'])"
                )

        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
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
        return _format_adverse_events_output(
            result,
            detail_level=detail_level,
            source_tool="search_adverse_events",
            result_context={"drug_name": drug_name},
        )
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
    detail_level: str = "standard",
) -> str:
    """
    Search clinical trial outcomes, measurements, and statistical analyses.
    
    Queries outcome titles, measurements, and analyses from ClinicalTrials.gov tables.
    Supports trial-centric outcome retrieval, outcome keyword searches across trials,
    and drug-centric efficacy comparisons based on p-values.
    
    ═══════════════════════════════════════════════════════════════════════════════
    SEARCH MODES
    ═══════════════════════════════════════════════════════════════════════════════
    
    mode: Required. One of:
        - "outcomes_for_trial": Get all outcomes and analyses for a specific trial.
            Requires: nct_id
            Returns: Outcomes, measurements, and statistical analyses for the trial
            Example: search_trial_outcomes(mode="outcomes_for_trial", nct_id="NCT01295827")
        
        - "trials_with_outcome": Find trials with outcomes matching a keyword.
            Requires: outcome_term
            Returns: Trials that have outcomes matching the search term
            Example: search_trial_outcomes(mode="trials_with_outcome", outcome_term="overall survival")
        
        - "efficacy_comparison": Find statistical analyses for trials involving a drug.
            Requires: drug_name
            Returns: Outcome analyses with p-values for trials involving the drug
            Example: search_trial_outcomes(mode="efficacy_comparison", drug_name="pembrolizumab")
    
    ═══════════════════════════════════════════════════════════════════════════════
    PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════════
    
    nct_id: NCT identifier for a specific trial (outcomes_for_trial mode).
        Example: "NCT01295827"
    
    outcome_term: Keyword to search for in outcome titles/descriptions.
        Examples: "overall survival", "progression-free survival", "response rate",
                  "complete remission", "hemoglobin A1c", "blood pressure"
    
    drug_name: Drug/intervention name for efficacy_comparison mode.
        Examples: "pembrolizumab", "imatinib", "metformin"
    
    outcome_type: Filter by outcome type.
        - "all" (default): Both primary and secondary outcomes
        - "primary": Only primary outcomes
        - "secondary": Only secondary outcomes
    
    min_p_value: Minimum p-value filter (0.0 to 1.0).
        Use with max_p_value to find statistically significant results.
    
    max_p_value: Maximum p-value filter (0.0 to 1.0).
        Example: max_p_value=0.05 for significant results only.
    
    limit: Maximum results (default: 20, max: 500).
    
    offset: Offset for pagination (default: 0).
    
    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLES
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. Get all outcomes for a specific trial:
       search_trial_outcomes(mode="outcomes_for_trial", nct_id="NCT01295827")
    
    2. Find trials with overall survival outcomes:
       search_trial_outcomes(mode="trials_with_outcome", outcome_term="overall survival")
    
    3. Find primary outcomes only:
       search_trial_outcomes(
           mode="trials_with_outcome", 
           outcome_term="progression-free survival",
           outcome_type="primary"
       )
    
    4. Get efficacy analyses for a drug with significant p-values:
       search_trial_outcomes(
           mode="efficacy_comparison", 
           drug_name="pembrolizumab",
           max_p_value=0.05
       )
    
    Returns:
        Outcome data including titles, measurements, p-values, confidence intervals,
        and statistical methods used.
    """
    try:
        valid_modes = {"outcomes_for_trial", "trials_with_outcome", "efficacy_comparison"}
        if not mode or mode not in valid_modes:
            return (
                "❌ Invalid input: mode is required.\n"
                "Valid options: outcomes_for_trial, trials_with_outcome, efficacy_comparison"
            )
        if mode == "outcomes_for_trial" and not nct_id:
            return (
                "❌ Invalid input: nct_id is required for mode='outcomes_for_trial'.\n"
                "Example: search_trial_outcomes(mode='outcomes_for_trial', nct_id='NCT01295827')"
            )
        if mode == "trials_with_outcome" and not outcome_term:
            return (
                "❌ Invalid input: outcome_term is required for mode='trials_with_outcome'.\n"
                "Example: search_trial_outcomes(mode='trials_with_outcome', outcome_term='overall survival')"
            )
        if mode == "efficacy_comparison" and not drug_name:
            return (
                "❌ Invalid input: drug_name is required for mode='efficacy_comparison'.\n"
                "Example: search_trial_outcomes(mode='efficacy_comparison', drug_name='pembrolizumab')"
            )

        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
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
        top_nct_id = result.hits[0].nct_id if result.hits else None
        return _format_outcomes_output(
            result,
            detail_level=detail_level,
            source_tool="search_trial_outcomes",
            result_context={"top_nct_id": top_nct_id},
        )
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
    detail_level: str = "standard",
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
        valid_modes = {"te_codes", "patents", "exclusivity", "generics"}
        if not mode or mode not in valid_modes:
            return (
                "❌ Invalid input: mode is required.\n"
                "Valid options: te_codes, patents, exclusivity, generics"
            )
        if not (drug_name or ingredient or nda_number):
            return (
                "❌ Invalid input: Provide at least one of drug_name, ingredient, or nda_number.\n"
                "Example: search_orange_book(mode='te_codes', drug_name='metformin')"
            )

        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
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
        return _format_orange_book_output(
            result,
            detail_level=detail_level,
            source_tool="search_orange_book",
            result_context={"drug_name": drug_name},
        )
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
    detail_level: str = "standard",
) -> str:
    """
    Resolve a drug identifier across internal databases (ChEMBL, DrugCentral, labels, trials).
    """
    try:
        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
        search_input = CrossDatabaseLookupInput(
            identifier=identifier,
            identifier_type=identifier_type,
            include_labels=include_labels,
            include_trials=include_trials,
            include_targets=include_targets,
            limit=min(limit, 200),
        )
        result = await cross_database_lookup_async(DEFAULT_CONFIG, search_input)
        return _format_cross_db_output(
            result,
            detail_level=detail_level,
            source_tool="lookup_drug_identifiers",
        )
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
    biotherapeutic_type: str = "all",
    limit: int = 20,
    offset: int = 0,
    detail_level: str = "standard",
) -> str:
    """
    Search biotherapeutics by sequence motif.

    Modes:
        - "by_sequence": Match by sequence motif (uses sequence/sequence_motif).
        - "similar_biologics": Find biologics sharing a motif (uses sequence/sequence_motif).

    For target-based biologic queries, use search_target_drugs with
    molecule_type="biotherapeutic".
    """
    try:
        valid_modes = {"by_sequence", "similar_biologics"}
        if not mode or mode not in valid_modes:
            return (
                "❌ Invalid input: mode is required.\n"
                "Valid options: by_sequence, similar_biologics"
            )
        if not (sequence or sequence_motif):
            return (
                "❌ Invalid input: sequence or sequence_motif is required.\n"
                "Example: search_biotherapeutics(mode='by_sequence', sequence='EVQLVESGG')"
            )

        detail_level, error = _validate_detail_level(detail_level)
        if error:
            return f"❌ Invalid input: {error}"
        sequence_value = sequence or sequence_motif
        search_input = BiotherapeuticSearchInput(
            mode=mode,
            sequence=sequence_value,
            biotherapeutic_type=biotherapeutic_type,
            limit=min(limit, 500),
            offset=offset,
        )
        result = await biotherapeutic_sequence_search_async(DEFAULT_CONFIG, search_input)
        return _format_biotherapeutic_output(
            result,
            detail_level=detail_level,
            source_tool="search_biotherapeutics",
        )
    except Exception as e:
        return f"Error searching biotherapeutics: {type(e).__name__}: {e}"


# =============================================================================
# TOOL COLLECTION
# =============================================================================

# Export all tools for easy registration
# NOTE: Pharmacology tools (search_drug_targets, search_target_drugs, etc.) 
# are in target_search.py and exported via TARGET_SEARCH_TOOLS
DBSEARCH_TOOLS = [
    # Clinical Trials
    search_clinical_trials,
    get_clinical_trial_details,
    # Lookup / Router
    check_data_availability,
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
]