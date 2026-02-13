import json
import inspect
from functools import wraps

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


def build_handoff_signals(source_tool: str, result_context: dict) -> str:
    """Build contextual next-step suggestions based on tool outputs."""
    suggestions: list[str] = []
    tool = (source_tool or "").strip()
    search_type = result_context.get("search_type")
    drug_name = result_context.get("drug_name")
    gene_symbol = result_context.get("gene_symbol")
    nct_id = result_context.get("top_nct_id")
    molecule_name = result_context.get("molecule_name")

    def add(line: str) -> None:
        if line and line not in suggestions:
            suggestions.append(line)

    if tool in {"pharmacology_search", "search_drug_targets", "get_drug_profile", "search_drug_trials"}:
        if drug_name:
            add(f'To see clinical trials: search_clinical_trials(intervention="{drug_name}")')
            add(f'To see FDA label: search_drug_labels(drug_names="{drug_name}")')
        if search_type == "drug_profile" and drug_name:
            add(f'To see trial outcomes: search_clinical_trials(intervention="{drug_name}", has_results=True)')
    if tool in {"pharmacology_search", "search_target_drugs"}:
        if gene_symbol:
            add(
                f'To see target-linked trials: search_molecule_trials(mode="trials_by_target", target_gene="{gene_symbol}")'
            )
    if tool == "search_clinical_trials":
        if nct_id:
            add(f'To see outcomes for a top trial: search_trial_outcomes(mode="outcomes_for_trial", nct_id="{nct_id}")')
    if tool == "get_clinical_trial_details":
        if nct_id:
            add(
                f'To inspect reported outcomes: search_trial_outcomes(mode="outcomes_for_trial", nct_id="{nct_id}")'
            )
    if tool == "search_drug_labels":
        if drug_name:
            add(
                f'To see pharmacology targets: pharmacology_search(search_type="drug_targets", drug_name="{drug_name}")'
            )
    if tool == "search_molecule_trials":
        if molecule_name:
            add(
                f'To see targets: pharmacology_search(search_type="drug_targets", drug_name="{molecule_name}")'
            )
    if tool == "search_adverse_events":
        if drug_name:
            add(
                f'To review label warnings: search_drug_labels(drug_names="{drug_name}", adverse_reactions_only=True)'
            )
    if tool == "check_data_availability":
        if result_context.get("has_targets") and drug_name:
            add(f'To see targets: pharmacology_search(search_type="drug_targets", drug_name="{drug_name}")')
        if result_context.get("has_trials") and drug_name:
            add(f'To see clinical trials: search_clinical_trials(intervention="{drug_name}")')
        if result_context.get("has_labels") and drug_name:
            add(f'To see FDA label: search_drug_labels(drug_names="{drug_name}", fetch_all_sections=True)')

    if not suggestions:
        return ""

    lines = ["[AGENT_SIGNALS]", "---", "Related searches:"]
    lines.extend([f"  -> {item}" for item in suggestions])
    return "\n".join(lines)