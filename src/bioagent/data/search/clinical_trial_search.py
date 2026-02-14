# clinical_trials_search.py
"""
Flexible Clinical Trials Search.

Builds dynamic SQL queries across ClinicalTrials.gov-derived tables with
multiple search strategies (fulltext, trigram, exact) and a rich set of
filters (status, phase, dates, enrollment, eligibility, geography).

Data Sources:
    - rag_study_search (trial metadata and search index)
    - rag_study_corpus (full text)
    - ctgov_eligibilities / ctgov_outcomes (eligibility/outcome details)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Iterable
from enum import Enum
import json

from pydantic import BaseModel, Field, field_validator

# Rich imports for pretty printing
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich import box

# Import config and async connection
from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig
from bioagent.data.semantic_utils import encode_query_vector, normalize_semantic_text


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _nz(x: Any, default: str = "—") -> str:
    """Safe string for None/empty."""
    return default if x in (None, "", [], {}) else str(x)


def _as_list(obj: Any) -> list[Any]:
    return obj if isinstance(obj, list) else []


def _fmt_bool(b: bool | None) -> str:
    if b is True:
        return "True"
    if b is False:
        return "False"
    return "Unknown"


def _fmt_ci(lower: Any, upper: Any, pct: Any) -> str | None:
    if lower is not None and upper is not None:
        return f"{_nz(pct, '')}% CI [{lower}, {upper}]".strip()
    return None


def _as_json_dict(v) -> dict:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return json.loads(bytes(v).decode("utf-8"))
        except Exception:
            return {}
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    try:
        return dict(v)
    except Exception:
        return {}


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return text[:max_len] + "..." if len(text) > max_len else text


# =============================================================================
# OUTCOME/AE ITERATORS
# =============================================================================

def _iter_outcome_measures(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for m in _as_list(o.get("measures")):
            yield {
                "outcome_type": o.get("type"),
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "group_title": m.get("group_title") or f"Group {m.get('group_idx')}",
                "classification": m.get("classification"),
                "param_type": m.get("param_type"),
                "value": m.get("value"),
                "value_num": m.get("value_num"),
                "units": m.get("units"),
                "dispersion_type": m.get("dispersion_type"),
                "dispersion": m.get("dispersion"),
                "lower": m.get("lower"),
                "upper": m.get("upper"),
            }


def _iter_outcome_counts(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for c in _as_list(o.get("counts")):
            yield {
                "outcome_type": o.get("type"),
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "group_title": c.get("group_title") or f"Group {c.get('group_idx')}",
                "scope": c.get("scope"),
                "units": c.get("units"),
                "count": c.get("count"),
            }


def _iter_outcome_analyses(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for o in _as_list(study_json.get("outcomes")):
        for a in _as_list(o.get("analyses")):
            comps = " vs ".join(
                g.get("group_title") for g in _as_list(a.get("comparison_groups")) if g.get("group_title")
            )
            yield {
                "outcome_title": o.get("title"),
                "time_frame": o.get("time_frame"),
                "method": a.get("method"),
                "param_type": a.get("param_type"),
                "param_value": a.get("param_value"),
                "p_value": a.get("p_value"),
                "ci_percent": a.get("ci_percent"),
                "ci_lower": a.get("ci_lower"),
                "ci_upper": a.get("ci_upper"),
                "comparison_groups": comps,
                "description": a.get("description"),
            }


def _iter_adverse_events(study_json: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for ae in _as_list(study_json.get("adverse_events")):
        yield {
            "term": ae.get("term"),
            "event_type": ae.get("event_type"),
            "group_title": ae.get("group_title") or f"Group {ae.get('group_idx')}",
            "affected": ae.get("subjects_affected"),
            "at_risk": ae.get("subjects_at_risk"),
            "rate_pct": ae.get("rate"),
        }


# =============================================================================
# RENDER STUDY TEXT
# =============================================================================

def render_study_text_full(
    study_json: dict[str, Any],
    nct_id: str | None = None,
    output_eligibility: bool = True,
    output_groups: bool = True,
    output_baseline_measurements: bool = True,
    output_results: bool = True,
    output_adverse_effects: bool = True,
    output_sponsors: bool = True,
    output_countries: bool = True,
) -> str:
    """Render a comprehensive, ordered narrative for a single trial."""
    md = study_json.get("metadata") or {}
    dsn = study_json.get("design") or {}
    elig = study_json.get("eligibility") or {}
    labels = study_json.get("labels") or {}
    reg_groups = _as_list(study_json.get("registry_groups"))
    res_groups = _as_list(study_json.get("results_groups"))
    outcomes = _as_list(study_json.get("outcomes"))
    aes = _as_list(study_json.get("adverse_events"))

    lines: list[str] = []

    # 1) Header & metadata
    header = _nz(md.get("brief_title"), "[No brief title]")
    if nct_id:
        header = f"{header} [{nct_id}]"
    lines.append(header)

    lines.append(
        "Type/Phase/Status: {stype}, {phase}, {status}.".format(
            stype=_nz(md.get("study_type")),
            phase=_nz(md.get("phase")),
            status=_nz(md.get("overall_status")),
        )
    )
    lines.append(
        "Enrollment: {enr} ({enr_t}). Dates — start: {start}; primary completion: {pc}; completion: {comp}.".format(
            enr=_nz(md.get("enrollment")),
            enr_t=_nz(md.get("enrollment_type")),
            start=_nz(md.get("start_date")),
            pc=_nz(md.get("primary_completion_date")),
            comp=_nz(md.get("completion_date")),
        )
    )
    if md.get("why_stopped"):
        lines.append(f"Why stopped: {_nz(md.get('why_stopped'))}.")
    if "has_results" in md:
        lines.append(f"Results posted: {_fmt_bool(md.get('has_results'))}.")

    # 2) Design
    lines.append(
        "Design — allocation: {alloc}; model: {model}; masking: {mask}; "
        "purpose: {purpose}; arms: {arms}; groups: {groups}.".format(
            alloc=_nz(dsn.get("allocation")),
            model=_nz(dsn.get("intervention_model")),
            mask=_nz(dsn.get("masking")),
            purpose=_nz(dsn.get("primary_purpose")),
            arms=_nz(md.get("number_of_arms")),
            groups=_nz(md.get("number_of_groups")),
        )
    )
    if dsn.get("masking_description"):
        lines.append("Masking description: " + _nz(dsn.get("masking_description")) + ".")

    # 3) Eligibility
    if output_eligibility:
        lines.append(
            "Eligibility — gender: {gender}; age: {min_age}–{max_age}; healthy volunteers: {hv}.".format(
                gender=_nz(elig.get("gender")),
                min_age=_nz(elig.get("minimum_age")),
                max_age=_nz(elig.get("maximum_age")),
                hv=_fmt_bool(elig.get("healthy_volunteers")),
            )
        )
        crit = elig.get("criteria") or ""
        if isinstance(crit, str) and crit.strip():
            first_line = crit.strip().splitlines()[0]
            lines.append("Eligibility criteria (first line): " + first_line)

    # 4) Labels
    conditions = labels.get("conditions") or []
    interventions = labels.get("interventions") or []
    sponsors = labels.get("sponsors") or []
    countries = labels.get("countries") or []
    lines.append("Conditions: " + (", ".join(map(str, conditions)) if conditions else "—") + ".")
    lines.append("Interventions (registry): " + (", ".join(map(str, interventions)) if interventions else "—") + ".")
    if output_sponsors and sponsors:
        lines.append("Lead sponsors: " + ", ".join(map(str, sponsors)) + ".")
    if output_countries and countries:
        lines.append("Countries: " + ", ".join(map(str, countries)) + ".")

    # 5) Registry-defined groups
    if output_groups and reg_groups:
        lines.append("Registry-defined groups & assigned interventions:")
        for g in reg_groups:
            title = _nz(g.get("title"))
            desc = g.get("description")
            ivs = [i for i in _as_list(g.get("interventions")) if isinstance(i, dict)]
            if ivs:
                iv_strs = []
                for i in ivs:
                    bit = _nz(i.get("name"))
                    itype = i.get("type")
                    idesc = i.get("description")
                    if itype:
                        bit += f" [{itype}]"
                    if idesc:
                        bit += f" — {idesc}"
                    iv_strs.append(bit)
                lines.append(f"  • {title}: " + "; ".join(iv_strs))
            else:
                lines.append(f"  • {title}")
            if desc and not ivs:
                lines.append(f"    {desc}")

    # 6) Results groups & baseline measurements
    if output_baseline_measurements and res_groups:
        lines.append("Results groups, baseline counts & baseline measurements:")
        for g in res_groups:
            title = _nz(g.get("title"))
            lines.append(f"  • Group: {title}")
            for bc in _as_list(g.get("baselines")):
                units = bc.get("units")
                scope = bc.get("scope")
                count = bc.get("count")
                lines.append(f"    - Baseline count: {_nz(count)} {units or ''} (scope: {_nz(scope)})".rstrip())
            for bm in _as_list(g.get("baseline_measurements")):
                parts = [
                    _nz(bm.get("title")),
                    (_nz(bm.get("param_type")).title() if bm.get("param_type") else ""),
                    _nz(bm.get("value")),
                    _nz(bm.get("units"), ""),
                ]
                disp = None
                if bm.get("dispersion_type") and bm.get("dispersion") is not None:
                    disp = f"{str(bm['dispersion_type']).replace('_', ' ').title()} {_nz(bm['dispersion'])}"
                if disp:
                    parts.append(f"({disp})")
                lines.append("    - Baseline measure: " + " ".join([p for p in parts if p]).strip())

    # 7) Outcomes
    if output_results and outcomes:
        lines.append("Outcomes:")
        measures_by_header = {}
        for row in _iter_outcome_measures(study_json):
            header_key = (row['outcome_type'], row['outcome_title'], row['time_frame'])
            if header_key not in measures_by_header:
                measures_by_header[header_key] = []
            measures_by_header[header_key].append(row)

        for header_key, measures in measures_by_header.items():
            outcome_type, outcome_title, time_frame = header_key
            header = f"  • [{_nz(outcome_type).title()}] {_nz(outcome_title)} [{_nz(time_frame)}]"
            lines.append(header)
            for row in measures:
                parts = [
                    f"    - {row['group_title']}:",
                    (row.get("param_type") or ""),
                    _nz(row.get("value")),
                    _nz(row.get("units"), ""),
                ]
                if row.get("dispersion_type") and row.get("dispersion") is not None:
                    parts.append(f"({str(row['dispersion_type']).replace('_', ' ').title()} {row['dispersion']})")
                ci = _fmt_ci(row.get("lower"), row.get("upper"), None)
                if ci:
                    parts.append(ci)
                if row.get("classification"):
                    parts.append(f"[{row['classification']}]")
                lines.append(" ".join([p for p in parts if p]).strip())

        counts_by_header = {}
        for row in _iter_outcome_counts(study_json):
            header_key = (row['outcome_type'], row['outcome_title'], row['time_frame'])
            if header_key not in counts_by_header:
                counts_by_header[header_key] = []
            counts_by_header[header_key].append(row)

        for header_key, counts in counts_by_header.items():
            outcome_type, outcome_title, time_frame = header_key
            header = f"  • [{_nz(outcome_type).title()}] {_nz(outcome_title)} [{_nz(time_frame)}]"
            lines.append(header)
            for row in counts:
                lines.append(f"    - {row['group_title']}: {_nz(row['count'])} {_nz(row['units'])} (scope: {_nz(row['scope'])})")

        analyses_by_outcome = {}
        for a in _iter_outcome_analyses(study_json):
            outcome_key = (a['outcome_title'], a['time_frame'])
            if outcome_key not in analyses_by_outcome:
                analyses_by_outcome[outcome_key] = []
            analyses_by_outcome[outcome_key].append(a)

        for outcome_key, analyses in analyses_by_outcome.items():
            for a in analyses:
                bits = [a.get("method") or "", a.get("comparison_groups") or ""]
                if a.get("param_type") and a.get("param_value") is not None:
                    bits.append(f"{a['param_type']}={a['param_value']}")
                if a.get("p_value") is not None:
                    bits.append(f"p={a['p_value']}")
                ci = _fmt_ci(a.get("ci_lower"), a.get("ci_upper"), a.get("ci_percent"))
                if ci:
                    bits.append(ci)
                desc = a.get("description")
                line = "    - Analysis: " + ", ".join([b for b in bits if b]).strip()
                lines.append(line)
                if desc:
                    lines.append(f"      {desc}")

    # 8) Adverse events
    if output_adverse_effects and aes:
        lines.append("Adverse events (per group):")
        aes_by_term = {}
        for ae in _iter_adverse_events(study_json):
            ae_key = (ae['term'], ae['event_type'])
            if ae_key not in aes_by_term:
                aes_by_term[ae_key] = []
            aes_by_term[ae_key].append(ae)

        for ae_key, ae_list in aes_by_term.items():
            term, event_type = ae_key
            header = f"  • {_nz(term)} [{_nz(event_type)}]"
            lines.append(header)
            for ae in ae_list:
                line = f"    - {_nz(ae['group_title'])}: {_nz(ae['affected'])}/{_nz(ae['at_risk'])}"
                if ae.get("rate_pct") is not None:
                    line += f" ({_nz(ae['rate_pct'])}%)"
                lines.append(line)

    return "\n".join(lines)


# =============================================================================
# ENUMS
# =============================================================================

class TrialPhase(str, Enum):
    """Trial phases - values match ctgov database exactly."""
    NA = "NA"
    PHASE_1 = "PHASE1"
    PHASE_1_2 = "PHASE1/PHASE2"
    PHASE_2 = "PHASE2"
    PHASE_2_3 = "PHASE2/PHASE3"
    PHASE_3 = "PHASE3"
    PHASE_4 = "PHASE4"
    EARLY_PHASE_1 = "EARLY_PHASE1"

    @classmethod
    def from_display(cls, display_name: str) -> "TrialPhase | None":
        if not display_name:
            return None
        normalized = (
            display_name.upper()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )
        mapping = {
            "PHASE1": cls.PHASE_1, "PHASE2": cls.PHASE_2, "PHASE3": cls.PHASE_3, "PHASE4": cls.PHASE_4,
            "PHASE1/PHASE2": cls.PHASE_1_2, "PHASE1PHASE2": cls.PHASE_1_2,
            "PHASE2/PHASE3": cls.PHASE_2_3, "PHASE2PHASE3": cls.PHASE_2_3,
            "EARLYPHASE1": cls.EARLY_PHASE_1, "NA": cls.NA, "N/A": cls.NA,
            "PHASEI": cls.PHASE_1, "PHASEII": cls.PHASE_2, "PHASEIII": cls.PHASE_3, "PHASEIV": cls.PHASE_4,
            "I": cls.PHASE_1, "II": cls.PHASE_2, "III": cls.PHASE_3, "IV": cls.PHASE_4,
            "1": cls.PHASE_1, "2": cls.PHASE_2, "3": cls.PHASE_3, "4": cls.PHASE_4,
            "1/2": cls.PHASE_1_2, "I/II": cls.PHASE_1_2,
            "2/3": cls.PHASE_2_3, "II/III": cls.PHASE_2_3,
            "EARLY1": cls.EARLY_PHASE_1,
        }
        return mapping.get(normalized)

    @property
    def display_name(self) -> str:
        display_map = {
            self.NA: "N/A", self.PHASE_1: "Phase 1", self.PHASE_1_2: "Phase 1/Phase 2",
            self.PHASE_2: "Phase 2", self.PHASE_2_3: "Phase 2/Phase 3",
            self.PHASE_3: "Phase 3", self.PHASE_4: "Phase 4", self.EARLY_PHASE_1: "Early Phase 1",
        }
        return display_map.get(self, self.value)


class TrialStatus(str, Enum):
    """Trial statuses - values match ctgov database exactly."""
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"
    RECRUITING = "RECRUITING"
    TERMINATED = "TERMINATED"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    WITHDRAWN = "WITHDRAWN"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    SUSPENDED = "SUSPENDED"

    @classmethod
    def from_display(cls, display_name: str) -> "TrialStatus | None":
        if not display_name:
            return None
        normalized = (
            display_name.upper()
            .replace(",", " ")
            .replace("-", " ")
            .replace("_", " ")
        )
        normalized = "_".join(part for part in normalized.split() if part)
        compact = normalized.replace("_", "")
        mapping = {
            "COMPLETED": cls.COMPLETED, "UNKNOWN": cls.UNKNOWN, "UNKNOWN_STATUS": cls.UNKNOWN,
            "RECRUITING": cls.RECRUITING, "TERMINATED": cls.TERMINATED,
            "NOT_YET_RECRUITING": cls.NOT_YET_RECRUITING, "NOTYETRECRUITING": cls.NOT_YET_RECRUITING,
            "ACTIVE_NOT_RECRUITING": cls.ACTIVE_NOT_RECRUITING, "ACTIVENOTRECRUITING": cls.ACTIVE_NOT_RECRUITING,
            "ACTIVE": cls.ACTIVE_NOT_RECRUITING, "WITHDRAWN": cls.WITHDRAWN,
            "ENROLLING_BY_INVITATION": cls.ENROLLING_BY_INVITATION,
            "ENROLLINGBYINVITATION": cls.ENROLLING_BY_INVITATION, "SUSPENDED": cls.SUSPENDED,
            # Common shorthand/LLM variants
            "NOT_RECRUITING": cls.ACTIVE_NOT_RECRUITING,
            "CLOSED": cls.COMPLETED,
        }
        return mapping.get(normalized) or mapping.get(compact)

    @property
    def display_name(self) -> str:
        display_map = {
            self.COMPLETED: "Completed", self.UNKNOWN: "Unknown status", self.RECRUITING: "Recruiting",
            self.TERMINATED: "Terminated", self.NOT_YET_RECRUITING: "Not yet recruiting",
            self.ACTIVE_NOT_RECRUITING: "Active, not recruiting", self.WITHDRAWN: "Withdrawn",
            self.ENROLLING_BY_INVITATION: "Enrolling by invitation", self.SUSPENDED: "Suspended",
        }
        return display_map.get(self, self.value)

    @classmethod
    def active_statuses(cls) -> list["TrialStatus"]:
        return [cls.RECRUITING, cls.ACTIVE_NOT_RECRUITING, cls.NOT_YET_RECRUITING, cls.ENROLLING_BY_INVITATION]

    @classmethod
    def closed_statuses(cls) -> list["TrialStatus"]:
        return [cls.COMPLETED, cls.TERMINATED, cls.WITHDRAWN, cls.SUSPENDED]


class StudyType(str, Enum):
    """Study types - values match ctgov database exactly."""
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"

    @classmethod
    def from_display(cls, display_name: str) -> "StudyType | None":
        if not display_name:
            return None
        normalized = display_name.upper().replace(" ", "_").replace("-", "_")
        mapping = {
            "INTERVENTIONAL": cls.INTERVENTIONAL, "OBSERVATIONAL": cls.OBSERVATIONAL,
            "EXPANDED_ACCESS": cls.EXPANDED_ACCESS, "EXPANDEDACCESS": cls.EXPANDED_ACCESS,
        }
        return mapping.get(normalized)

    @property
    def display_name(self) -> str:
        return {self.INTERVENTIONAL: "Interventional", self.OBSERVATIONAL: "Observational",
                self.EXPANDED_ACCESS: "Expanded Access"}.get(self, self.value)


class InterventionType(str, Enum):
    """Intervention types - values match ctgov database exactly (lowercase)."""
    DRUG = "drug"
    OTHER = "other"
    DEVICE = "device"
    BEHAVIORAL = "behavioral"
    PROCEDURE = "procedure"
    BIOLOGICAL = "biological"
    DIAGNOSTIC_TEST = "diagnostic_test"
    DIETARY_SUPPLEMENT = "dietary_supplement"
    RADIATION = "radiation"
    COMBINATION_PRODUCT = "combination_product"
    GENETIC = "genetic"

    @classmethod
    def from_display(cls, display_name: str) -> "InterventionType | None":
        if not display_name:
            return None
        normalized = display_name.lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(normalized)
        except ValueError:
            return None

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


class SearchStrategy(str, Enum):
    TRIGRAM = "trigram"
    FULLTEXT = "fulltext"
    EXACT = "exact"
    COMBINED = "combined"


class SortField(str, Enum):
    RELEVANCE = "relevance"
    START_DATE = "start_date"
    COMPLETION_DATE = "completion_date"
    ENROLLMENT = "enrollment"
    NCT_ID = "nct_id"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# OUTPUT MODELS WITH PRETTY PRINT
# =============================================================================

@dataclass
class TrialSearchHit:
    """A single search result with match metadata."""
    nct_id: str
    score: float
    brief_title: str
    phase: str | None = None
    status: str | None = None
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    lead_sponsor: str | None = None
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    intervention_types: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    
    # Match metadata
    match_reasons: list[str] = field(default_factory=list)
    condition_score: float | None = None
    intervention_score: float | None = None
    keyword_score: float | None = None
    relevance_breakdown: dict[str, float] = field(default_factory=dict)
    
    # Full data (optional)
    study_json: dict[str, Any] | None = None
    rendered_summary: str | None = None

    @property
    def phase_display(self) -> str:
        if not self.phase:
            return "N/A"
        try:
            return TrialPhase(self.phase).display_name
        except ValueError:
            return self.phase

    @property
    def status_display(self) -> str:
        if not self.status:
            return "Unknown"
        try:
            return TrialStatus(self.status).display_name
        except ValueError:
            return self.status

    def pretty_print(self, console: Console | None = None, show_summary: bool = False) -> None:
        """Pretty print a single search hit."""
        console = console or Console()
        
        # Status color mapping
        status_colors = {
            "RECRUITING": "green",
            "ACTIVE_NOT_RECRUITING": "yellow",
            "COMPLETED": "blue",
            "TERMINATED": "red",
            "WITHDRAWN": "red",
            "NOT_YET_RECRUITING": "cyan",
            "SUSPENDED": "orange3",
            "UNKNOWN": "dim",
        }
        status_color = status_colors.get(self.status, "white")
        
        # Phase color
        phase_colors = {
            "PHASE3": "green bold",
            "PHASE2": "yellow",
            "PHASE1": "cyan",
            "PHASE4": "blue",
        }
        phase_color = phase_colors.get(self.phase, "white")
        
        # Build the tree
        title = f"[bold cyan]🔬 {self.nct_id}[/bold cyan] [dim](score: {self.score:.3f})[/dim]"
        tree = Tree(title)
        
        # Title branch
        tree.add(f"[bold]{_truncate(self.brief_title, 100)}[/bold]")
        
        # Metadata branch
        meta = tree.add("[bold yellow]📋 Metadata[/bold yellow]")
        meta.add(f"[dim]Phase:[/dim] [{phase_color}]{self.phase_display}[/{phase_color}]")
        meta.add(f"[dim]Status:[/dim] [{status_color}]{self.status_display}[/{status_color}]")
        if self.enrollment:
            meta.add(f"[dim]Enrollment:[/dim] {self.enrollment:,}")
        if self.start_date:
            meta.add(f"[dim]Start Date:[/dim] {self.start_date}")
        if self.completion_date:
            meta.add(f"[dim]Completion Date:[/dim] {self.completion_date}")
        if self.lead_sponsor:
            meta.add(f"[dim]Sponsor:[/dim] {_truncate(self.lead_sponsor, 50)}")
        
        # Match reasons
        if self.match_reasons:
            match_str = ", ".join(self.match_reasons)
            meta.add(f"[dim]Matched on:[/dim] [magenta]{match_str}[/magenta]")
        
        # Conditions
        if self.conditions:
            cond_branch = tree.add(f"[bold green]🏥 Conditions ({len(self.conditions)})[/bold green]")
            for cond in self.conditions[:5]:
                cond_branch.add(f"[white]• {cond}[/white]")
            if len(self.conditions) > 5:
                cond_branch.add(f"[dim]... and {len(self.conditions) - 5} more[/dim]")
        
        # Interventions
        if self.interventions:
            int_branch = tree.add(f"[bold blue]💊 Interventions ({len(self.interventions)})[/bold blue]")
            for intv in self.interventions[:5]:
                int_branch.add(f"[white]• {intv}[/white]")
            if len(self.interventions) > 5:
                int_branch.add(f"[dim]... and {len(self.interventions) - 5} more[/dim]")
        
        # Intervention types
        if self.intervention_types:
            tree.add(f"[dim]Types: {', '.join(self.intervention_types)}[/dim]")
        
        # Countries
        if self.countries:
            tree.add(f"[dim]🌍 Countries: {', '.join(self.countries[:5])}"
                    + (f" (+{len(self.countries)-5} more)" if len(self.countries) > 5 else "") + "[/dim]")
        
        console.print(tree)
        
        # Show rendered summary if requested
        if show_summary and self.rendered_summary:
            console.print(Panel(
                self.rendered_summary[:2000] + ("..." if len(self.rendered_summary) > 2000 else ""),
                title="[bold]Full Summary[/bold]",
                border_style="dim",
            ))

    def __str__(self) -> str:
        return f"TrialSearchHit(nct_id='{self.nct_id}', score={self.score:.3f}, title='{_truncate(self.brief_title, 40)}')"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ClinicalTrialsSearchOutput:
    """Search results container."""
    status: Literal["success", "not_found", "error"]
    total_hits: int = 0
    hits: list[TrialSearchHit] = field(default_factory=list)
    error: str | None = None
    
    # Query echo
    query_summary: str = ""
    filters_applied: list[str] = field(default_factory=list)
    
    # Pagination info
    limit: int = 50
    offset: int = 0
    has_more: bool = False

    def get_nct_ids(self) -> list[str]:
        return [hit.nct_id for hit in self.hits]

    def get_top_hits(self, n: int = 5) -> list[TrialSearchHit]:
        return sorted(self.hits, key=lambda h: h.score, reverse=True)[:n]

    def pretty_print(
        self,
        console: Console | None = None,
        max_hits: int = 10,
        show_table: bool = True,
        show_details: bool = False,
    ) -> None:
        """
        Pretty print search results.
        
        Args:
            console: Rich console instance
            max_hits: Maximum number of hits to display
            show_table: Show results as a table (vs tree view)
            show_details: Show detailed info for each hit
        """
        console = console or Console()
        
        # Status header
        status_icons = {"success": "✅", "not_found": "🔍", "error": "❌"}
        status_colors = {"success": "green", "not_found": "yellow", "error": "red"}
        icon = status_icons.get(self.status, "❓")
        color = status_colors.get(self.status, "white")
        
        # Header panel
        header_text = Text()
        header_text.append(f"{icon} Status: ", style="bold")
        header_text.append(f"{self.status.upper()}\n", style=f"bold {color}")
        header_text.append("Query: ", style="dim")
        header_text.append(f"{self.query_summary}\n", style="white")
        
        if self.filters_applied:
            header_text.append("Filters: ", style="dim")
            header_text.append(f"{', '.join(self.filters_applied)}\n", style="cyan")
        
        header_text.append("Results: ", style="dim")
        header_text.append(f"{self.total_hits} hits", style="bold white")
        if self.has_more:
            header_text.append(" (more available)", style="dim")
        
        console.print(Panel(header_text, title="[bold]Clinical Trials Search Results[/bold]", border_style="blue"))
        
        # Handle errors
        if self.error:
            console.print(Panel(f"[red]{self.error}[/red]", title="Error", border_style="red"))
            return
        
        # No results
        if not self.hits:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        # Show results
        if show_table:
            self._print_table(console, max_hits)
        else:
            self._print_trees(console, max_hits, show_details)
        
        # Pagination info
        if self.has_more:
            console.print(f"\n[dim]Showing {min(len(self.hits), max_hits)} of {self.total_hits} results. "
                         f"Use offset={self.offset + self.limit} to see more.[/dim]")

    def _print_table(self, console: Console, max_hits: int) -> None:
        """Print results as a table."""
        table = Table(
            title=None,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            row_styles=["", "dim"],
        )
        
        table.add_column("#", style="dim", width=3)
        table.add_column("NCT ID", style="cyan", width=12)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Title", width=40, overflow="fold")
        table.add_column("Phase", width=10)
        table.add_column("Status", width=15)
        table.add_column("Enrollment", justify="right", width=10)
        table.add_column("Matched", width=15)
        
        for i, hit in enumerate(self.hits[:max_hits], 1):
            # Status styling
            status_style = {
                "RECRUITING": "green",
                "COMPLETED": "blue",
                "TERMINATED": "red",
                "ACTIVE_NOT_RECRUITING": "yellow",
            }.get(hit.status, "white")
            
            # Phase styling
            phase_style = {
                "PHASE3": "green",
                "PHASE2": "yellow",
                "PHASE1": "cyan",
            }.get(hit.phase, "white")
            
            table.add_row(
                str(i),
                hit.nct_id,
                f"{hit.score:.2f}",
                _truncate(hit.brief_title, 40),
                f"[{phase_style}]{hit.phase_display}[/{phase_style}]",
                f"[{status_style}]{hit.status_display}[/{status_style}]",
                f"{hit.enrollment:,}" if hit.enrollment else "—",
                ", ".join(hit.match_reasons) if hit.match_reasons else "—",
            )
        
        console.print(table)

    def _print_trees(self, console: Console, max_hits: int, show_details: bool) -> None:
        """Print results as trees."""
        for i, hit in enumerate(self.hits[:max_hits], 1):
            console.print(f"\n[bold white]─── Result {i}/{min(len(self.hits), max_hits)} ───[/bold white]")
            hit.pretty_print(console, show_summary=show_details)

    def __str__(self) -> str:
        return f"ClinicalTrialsSearchOutput(status='{self.status}', hits={self.total_hits}, query='{_truncate(self.query_summary, 30)}')"

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# INPUT MODEL
# =============================================================================

class ClinicalTrialsSearchInput(BaseModel):
    """Flexible search input with multiple query types and filters."""
    
    # Search Queries
    condition: str | None = Field(default=None, description="Condition/disease to search")
    intervention: str | None = Field(default=None, description="Drug/intervention to search")
    keyword: str | None = Field(default=None, description="Free-text keyword search")
    nct_ids: list[str] | None = Field(default=None, description="Specific NCT IDs to retrieve")
    sponsor: str | None = Field(default=None, description="Sponsor name to filter")
    
    # Filters
    status: list[TrialStatus] | None = Field(default=None)
    phase: list[TrialPhase] | None = Field(default=None)
    study_type: StudyType | None = Field(default=None)
    intervention_type: list[InterventionType] | None = Field(default=None)
    outcome_type: Literal["primary", "secondary", "all"] | None = Field(default=None)
    eligibility_gender: Literal["male", "female", "all"] | None = Field(default=None)
    eligibility_age_range: tuple[int, int] | None = Field(default=None)
    country: list[str] | None = Field(default=None)
    
    start_date_from: date | None = Field(default=None)
    start_date_to: date | None = Field(default=None)
    completion_date_from: date | None = Field(default=None)
    completion_date_to: date | None = Field(default=None)
    
    min_enrollment: int | None = Field(default=None, ge=0)
    max_enrollment: int | None = Field(default=None, ge=0)
    
    has_results: bool | None = Field(default=None)
    is_fda_regulated: bool | None = Field(default=None)
    
    # Search Options
    strategy: SearchStrategy = Field(default=SearchStrategy.COMBINED)
    match_all: bool = Field(default=False)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Output Options
    sort_by: SortField = Field(default=SortField.RELEVANCE)
    sort_order: SortOrder = Field(default=SortOrder.DESC)
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    include_study_json: bool = Field(default=True)
    condition_semantic_terms: list[str] | None = Field(default=None, exclude=True)
    
    # Output Sections
    output_eligibility: bool = True
    output_groups: bool = True
    output_baseline_measurements: bool = True
    output_results: bool = True
    output_adverse_effects: bool = True
    output_sponsors: bool = True
    output_countries: bool = True

    @field_validator('condition', 'intervention', 'keyword', 'sponsor', mode='before')
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    @field_validator('nct_ids', mode='before')
    @classmethod
    def normalize_nct_ids(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [nct.strip().upper() for nct in v if nct and nct.strip()]
        return v

    @field_validator('country', mode='before')
    @classmethod
    def normalize_countries(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            cleaned = [c.strip() for c in v if isinstance(c, str) and c.strip()]
            return cleaned or None
        return v

    @field_validator('eligibility_age_range', mode='before')
    @classmethod
    def normalize_age_range(cls, v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                return (int(v[0]), int(v[1]))
            except Exception:
                return v
        return v

    @field_validator('status', mode='before')
    @classmethod
    def normalize_status(cls, v):
        if v is None:
            return None
        def to_enum(val) -> TrialStatus:
            if isinstance(val, TrialStatus):
                return val
            if isinstance(val, str):
                try:
                    return TrialStatus(val.upper())
                except ValueError:
                    pass
                result = TrialStatus.from_display(val)
                if result:
                    return result
                allowed = ", ".join(s.display_name for s in TrialStatus)
                raise ValueError(f"Invalid status '{val}'. Allowed: {allowed}")
            allowed = ", ".join(s.display_name for s in TrialStatus)
            raise ValueError(f"Cannot convert {type(val)} to TrialStatus. Allowed: {allowed}")
        if isinstance(v, (list, tuple)):
            out: list[TrialStatus] = []
            for item in v:
                parsed = to_enum(item)
                if parsed not in out:
                    out.append(parsed)
            return out
        return [to_enum(v)]

    @field_validator('phase', mode='before')
    @classmethod
    def normalize_phase(cls, v):
        if v is None:
            return None
        def to_enum(val) -> TrialPhase:
            if isinstance(val, TrialPhase):
                return val
            if isinstance(val, str):
                normalized = (
                    val.upper()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )
                try:
                    return TrialPhase(normalized)
                except ValueError:
                    pass
                result = TrialPhase.from_display(val)
                if result:
                    return result
                allowed = ", ".join(p.display_name for p in TrialPhase)
                raise ValueError(f"Invalid phase '{val}'. Allowed: {allowed}")
            allowed = ", ".join(p.display_name for p in TrialPhase)
            raise ValueError(f"Cannot convert {type(val)} to TrialPhase. Allowed: {allowed}")
        if isinstance(v, (list, tuple)):
            out: list[TrialPhase] = []
            for item in v:
                parsed = to_enum(item)
                if parsed not in out:
                    out.append(parsed)
            return out
        return [to_enum(v)]

    @field_validator('study_type', mode='before')
    @classmethod
    def normalize_study_type(cls, v):
        if v is None:
            return None
        if isinstance(v, StudyType):
            return v
        if isinstance(v, str):
            try:
                return StudyType(v.upper().replace(" ", "_"))
            except ValueError:
                result = StudyType.from_display(v)
                if result:
                    return result
                allowed = ", ".join(s.display_name for s in StudyType)
                raise ValueError(f"Invalid study_type '{v}'. Allowed: {allowed}")
        allowed = ", ".join(s.display_name for s in StudyType)
        raise ValueError(f"Cannot convert {type(v)} to StudyType. Allowed: {allowed}")

    @field_validator('intervention_type', mode='before')
    @classmethod
    def normalize_intervention_type(cls, v):
        if v is None:
            return None
        def to_enum(val) -> InterventionType:
            if isinstance(val, InterventionType):
                return val
            if isinstance(val, str):
                try:
                    return InterventionType(val.lower().replace(" ", "_"))
                except ValueError:
                    result = InterventionType.from_display(val)
                    if result:
                        return result
                    allowed = ", ".join(i.display_name for i in InterventionType)
                    raise ValueError(f"Invalid intervention_type '{val}'. Allowed: {allowed}")
            allowed = ", ".join(i.display_name for i in InterventionType)
            raise ValueError(f"Cannot convert {type(val)} to InterventionType. Allowed: {allowed}")
        if isinstance(v, (list, tuple)):
            return [to_enum(x) for x in v]
        return [to_enum(v)]

    def has_text_query(self) -> bool:
        return any([self.condition, self.intervention, self.keyword, self.sponsor])

    def has_any_query(self) -> bool:
        return self.has_text_query() or bool(self.nct_ids)

    def validate_inputs(self) -> list[str]:
        errors: list[str] = []
        if self.nct_ids and self.has_text_query():
            errors.append(
                "nct_ids cannot be combined with condition, intervention, keyword, or sponsor. "
                "Use nct_ids alone for direct lookup."
            )
        return errors

    def pretty_print(self, console: Console | None = None) -> None:
        """Pretty print the search input parameters."""
        console = console or Console()
        
        tree = Tree("[bold cyan]🔎 Clinical Trials Search Query[/bold cyan]")
        
        # Search terms
        if self.has_text_query() or self.nct_ids:
            search_branch = tree.add("[bold yellow]Search Terms[/bold yellow]")
            if self.condition:
                search_branch.add(f"[dim]Condition:[/dim] [green]{self.condition}[/green]")
            if self.intervention:
                search_branch.add(f"[dim]Intervention:[/dim] [blue]{self.intervention}[/blue]")
            if self.keyword:
                search_branch.add(f"[dim]Keyword:[/dim] [magenta]{self.keyword}[/magenta]")
            if self.sponsor:
                search_branch.add(f"[dim]Sponsor:[/dim] [cyan]{self.sponsor}[/cyan]")
            if self.nct_ids:
                search_branch.add(f"[dim]NCT IDs:[/dim] {', '.join(self.nct_ids)}")
        
        # Filters
        filters = []
        if self.status:
            filters.append(f"Status: {', '.join(s.display_name for s in self.status)}")
        if self.phase:
            filters.append(f"Phase: {', '.join(p.display_name for p in self.phase)}")
        if self.study_type:
            filters.append(f"Study Type: {self.study_type.display_name}")
        if self.intervention_type:
            filters.append(f"Intervention Type: {', '.join(i.display_name for i in self.intervention_type)}")
        if self.outcome_type:
            filters.append(f"Outcome Type: {self.outcome_type}")
        if self.eligibility_gender:
            filters.append(f"Eligibility Gender: {self.eligibility_gender}")
        if self.eligibility_age_range:
            filters.append(f"Eligibility Age Range: {self.eligibility_age_range[0]}-{self.eligibility_age_range[1]}")
        if self.country:
            filters.append(f"Country: {', '.join(self.country)}")
        if self.start_date_from or self.start_date_to:
            filters.append(f"Start Date: {self.start_date_from or '...'} to {self.start_date_to or '...'}")
        if self.min_enrollment is not None or self.max_enrollment is not None:
            filters.append(f"Enrollment: {self.min_enrollment or 0} to {self.max_enrollment or '∞'}")
        if self.has_results is not None:
            filters.append(f"Has Results: {self.has_results}")
        if self.is_fda_regulated is not None:
            filters.append(f"FDA Regulated: {self.is_fda_regulated}")
        
        if filters:
            filter_branch = tree.add("[bold green]Filters[/bold green]")
            for f in filters:
                filter_branch.add(f"[white]• {f}[/white]")
        
        # Options
        options_branch = tree.add("[bold blue]Options[/bold blue]")
        options_branch.add(f"[dim]Strategy:[/dim] {self.strategy.value}")
        options_branch.add(f"[dim]Match All:[/dim] {self.match_all}")
        options_branch.add(f"[dim]Sort:[/dim] {self.sort_by.value} {self.sort_order.value}")
        options_branch.add(f"[dim]Limit:[/dim] {self.limit} (offset: {self.offset})")
        
        console.print(tree)

    def __str__(self) -> str:
        parts = []
        if self.condition:
            parts.append(f"condition='{self.condition}'")
        if self.intervention:
            parts.append(f"intervention='{self.intervention}'")
        if self.keyword:
            parts.append(f"keyword='{self.keyword}'")
        if self.nct_ids:
            parts.append(f"nct_ids={self.nct_ids}")
        return f"ClinicalTrialsSearchInput({', '.join(parts) or 'empty'})"


# =============================================================================
# QUERY BUILDER
# =============================================================================

class ClinicalTrialQueryBuilder:
    """Builds dynamic SQL queries for clinical trial search."""
    
    def __init__(self, search_input: ClinicalTrialsSearchInput):
        self.input = search_input
        self.params: list[Any] = []
        self.param_idx = 0

    def _add_param(self, value: Any) -> str:
        self.param_idx += 1
        self.params.append(value)
        return f"${self.param_idx}"

    @staticmethod
    def _age_to_years_expr(column_name: str) -> str:
        return f"""
            CASE
                WHEN {column_name} IS NULL THEN NULL
                WHEN {column_name} ILIKE '%Year%' THEN split_part({column_name}, ' ', 1)::float
                WHEN {column_name} ILIKE '%Month%' THEN split_part({column_name}, ' ', 1)::float / 12.0
                WHEN {column_name} ILIKE '%Week%' THEN split_part({column_name}, ' ', 1)::float / 52.0
                WHEN {column_name} ILIKE '%Day%' THEN split_part({column_name}, ' ', 1)::float / 365.0
                ELSE NULL
            END
        """

    def build(self) -> tuple[str, list[Any]]:
        if self.input.nct_ids and not self.input.has_text_query():
            return self._build_nct_lookup()
        return self._build_search_query()

    def _build_nct_lookup(self) -> tuple[str, list[Any]]:
        placeholders = ", ".join(self._add_param(nct) for nct in self.input.nct_ids)
        sql = f"""
            SELECT 
                rs.nct_id, 1.0::real AS score, rs.brief_title, rs.phase, rs.overall_status,
                rs.enrollment, rs.start_date::text, rs.completion_date::text, rs.lead_sponsor,
                rs.conditions, rs.interventions, rs.intervention_types, rs.countries,
                ARRAY['nct_id_match']::text[] AS match_reasons,
                NULL::real AS condition_score, NULL::real AS intervention_score, NULL::real AS keyword_score
                {', c.study_json' if self.input.include_study_json else ''}
            FROM public.rag_study_search rs
            {'LEFT JOIN public.rag_study_corpus c USING (nct_id)' if self.input.include_study_json else ''}
            WHERE rs.nct_id IN ({placeholders})
        """
        return sql, self.params

    def _build_search_query(self) -> tuple[str, list[Any]]:
        ctes = []
        if self.input.condition:
            ctes.append(self._build_condition_cte())
        if self.input.intervention:
            ctes.append(self._build_intervention_cte())
        if self.input.keyword:
            ctes.append(self._build_keyword_cte())
        if self.input.sponsor:
            ctes.append(self._build_sponsor_cte())
        return self._build_main_query(ctes), self.params

    def _build_condition_cte(self) -> str:
        terms: list[str] = [self.input.condition or ""]
        if self.input.condition_semantic_terms:
            terms.extend(self.input.condition_semantic_terms)
        normalized_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = normalize_semantic_text(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_terms.append(normalized)
        if not normalized_terms:
            normalized_terms = [normalize_semantic_text(self.input.condition or "")]

        value_rows: list[str] = []
        for term in normalized_terms[:5]:
            term_ph = self._add_param(term)
            ts_ph = self._add_param(" & ".join(term.split()))
            value_rows.append(f"({term_ph}, {ts_ph})")
        terms_cte = f"condition_terms(term, tsquery) AS (VALUES {', '.join(value_rows)})"

        if self.input.strategy == SearchStrategy.FULLTEXT:
            return f"""
                {terms_cte},
                condition_matches AS (
                    SELECT rs.nct_id,
                           MAX(ts_rank(rs.terms_tsv, to_tsquery('english', ct.tsquery)))::real AS score
                    FROM public.rag_study_search rs
                    JOIN condition_terms ct ON TRUE
                    WHERE rs.terms_tsv @@ to_tsquery('english', ct.tsquery)
                    GROUP BY rs.nct_id
                )"""
        elif self.input.strategy == SearchStrategy.TRIGRAM:
            return f"""
                {terms_cte},
                condition_matches AS (
                    SELECT rs.nct_id,
                           MAX(GREATEST(
                               similarity(rs.conditions_norm, ct.term),
                               similarity(rs.mesh_conditions_norm, ct.term)
                           ))::real AS score
                    FROM public.rag_study_search rs
                    JOIN condition_terms ct ON TRUE
                    WHERE rs.conditions_norm % ct.term OR rs.mesh_conditions_norm % ct.term
                    GROUP BY rs.nct_id
                )"""
        else:
            return f"""
                {terms_cte},
                condition_matches AS (
                    SELECT rs.nct_id, MAX(GREATEST(
                        COALESCE(similarity(rs.conditions_norm, ct.term), 0),
                        COALESCE(similarity(rs.mesh_conditions_norm, ct.term), 0),
                        COALESCE(ts_rank(rs.terms_tsv, to_tsquery('english', ct.tsquery)) * 0.5, 0)
                    ))::real AS score
                    FROM public.rag_study_search rs
                    JOIN condition_terms ct ON TRUE
                    WHERE rs.conditions_norm % ct.term
                       OR rs.mesh_conditions_norm % ct.term
                       OR rs.terms_tsv @@ to_tsquery('english', ct.tsquery)
                    GROUP BY rs.nct_id
                )"""

    def _build_intervention_cte(self) -> str:
        query_norm = self._add_param(self.input.intervention.lower())
        if self.input.strategy == SearchStrategy.FULLTEXT:
            tsquery = self._add_param(" & ".join(self.input.intervention.split()))
            return f"""
                intervention_matches AS (
                    SELECT nct_id, ts_rank(terms_tsv, to_tsquery('english', {tsquery}))::real AS score
                    FROM public.rag_study_search WHERE terms_tsv @@ to_tsquery('english', {tsquery})
                )"""
        elif self.input.strategy == SearchStrategy.TRIGRAM:
            return f"""
                intervention_matches AS (
                    SELECT nct_id, GREATEST(similarity(interventions_norm, {query_norm}), similarity(mesh_interventions_norm, {query_norm}))::real AS score
                    FROM public.rag_study_search WHERE interventions_norm % {query_norm} OR mesh_interventions_norm % {query_norm}
                )"""
        else:
            tsquery = self._add_param(" & ".join(self.input.intervention.split()))
            return f"""
                intervention_matches AS (
                    SELECT nct_id, GREATEST(
                        COALESCE(similarity(interventions_norm, {query_norm}), 0),
                        COALESCE(similarity(mesh_interventions_norm, {query_norm}), 0),
                        COALESCE(ts_rank(terms_tsv, to_tsquery('english', {tsquery})) * 0.5, 0)
                    )::real AS score
                    FROM public.rag_study_search
                    WHERE interventions_norm % {query_norm} OR mesh_interventions_norm % {query_norm} OR terms_tsv @@ to_tsquery('english', {tsquery})
                )"""

    def _build_keyword_cte(self) -> str:
        tsquery = self._add_param(" & ".join(self.input.keyword.split()))
        return f"""
            keyword_matches AS (
                SELECT nct_id, ts_rank(title_description_tsv, to_tsquery('english', {tsquery}))::real AS score
                FROM public.rag_study_search WHERE title_description_tsv @@ to_tsquery('english', {tsquery})
            )"""

    def _build_sponsor_cte(self) -> str:
        sponsor_pattern = self._add_param(f"%{self.input.sponsor.lower()}%")
        return f"""
            sponsor_matches AS (
                SELECT nct_id, 1.0::real AS score FROM public.rag_study_search
                WHERE lead_sponsor ILIKE {sponsor_pattern} OR sponsors_norm ILIKE {sponsor_pattern}
            )"""

    def _build_main_query(self, ctes: list[str]) -> str:
        has_condition = self.input.condition is not None
        has_intervention = self.input.intervention is not None
        has_keyword = self.input.keyword is not None
        has_sponsor = self.input.sponsor is not None

        score_parts = []
        if has_condition: score_parts.append("COALESCE(cm.score, 0)")
        if has_intervention: score_parts.append("COALESCE(im.score, 0)")
        if has_keyword: score_parts.append("COALESCE(km.score, 0)")
        if has_sponsor: score_parts.append("COALESCE(sm.score, 0)")
        if not score_parts: score_parts = ["1.0"]

        score_expr = f"LEAST({', '.join(score_parts)})" if self.input.match_all else f"({' + '.join(score_parts)}) / {len(score_parts)}"

        reason_parts = []
        if has_condition: reason_parts.append("CASE WHEN cm.score > 0 THEN 'condition' END")
        if has_intervention: reason_parts.append("CASE WHEN im.score > 0 THEN 'intervention' END")
        if has_keyword: reason_parts.append("CASE WHEN km.score > 0 THEN 'keyword' END")
        if has_sponsor: reason_parts.append("CASE WHEN sm.score > 0 THEN 'sponsor' END")
        reasons_expr = f"ARRAY_REMOVE(ARRAY[{', '.join(reason_parts)}], NULL)" if reason_parts else "ARRAY[]::text[]"

        filters = self._build_filters()
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        join_type = "JOIN" if self.input.match_all else "LEFT JOIN"
        joins = []
        if has_condition: joins.append(f"{join_type} condition_matches cm USING (nct_id)")
        if has_intervention: joins.append(f"{join_type} intervention_matches im USING (nct_id)")
        if has_keyword: joins.append(f"{join_type} keyword_matches km USING (nct_id)")
        if has_sponsor: joins.append(f"{join_type} sponsor_matches sm USING (nct_id)")

        if not self.input.match_all and joins:
            match_conditions = []
            if has_condition: match_conditions.append("cm.nct_id IS NOT NULL")
            if has_intervention: match_conditions.append("im.nct_id IS NOT NULL")
            if has_keyword: match_conditions.append("km.nct_id IS NOT NULL")
            if has_sponsor: match_conditions.append("sm.nct_id IS NOT NULL")
            if match_conditions:
                match_clause = f"({' OR '.join(match_conditions)})"
                where_clause = f"{where_clause} AND {match_clause}" if where_clause else f"WHERE {match_clause}"

        eligibility_join = ""
        if self.input.eligibility_gender or self.input.eligibility_age_range:
            eligibility_join = "LEFT JOIN ctgov_eligibilities e ON rs.nct_id = e.nct_id"

        limit_param = self._add_param(self.input.limit + 1)
        offset_param = self._add_param(self.input.offset)
        order_clause = self._build_order_clause()
        cte_section = "WITH " + ",\n".join(ctes) if ctes else ""

        return f"""
            {cte_section}
            SELECT rs.nct_id, ({score_expr})::real AS score, rs.brief_title, rs.phase, rs.overall_status,
                rs.enrollment, rs.start_date::text, rs.completion_date::text, rs.lead_sponsor,
                rs.conditions, rs.interventions, rs.intervention_types, rs.countries,
                {reasons_expr} AS match_reasons,
                {'cm.score' if has_condition else 'NULL'}::real AS condition_score,
                {'im.score' if has_intervention else 'NULL'}::real AS intervention_score,
                {'km.score' if has_keyword else 'NULL'}::real AS keyword_score
                {', c.study_json' if self.input.include_study_json else ''}
            FROM public.rag_study_search rs {' '.join(joins)}
            {eligibility_join}
            {'LEFT JOIN public.rag_study_corpus c ON rs.nct_id = c.nct_id' if self.input.include_study_json else ''}
            {where_clause} {order_clause} LIMIT {limit_param} OFFSET {offset_param}
        """

    def _build_filters(self) -> list[str]:
        filters = []
        if self.input.status:
            placeholders = ", ".join(self._add_param(s.value) for s in self.input.status)
            filters.append(f"rs.overall_status IN ({placeholders})")
        if self.input.phase:
            placeholders = ", ".join(self._add_param(p.value) for p in self.input.phase)
            filters.append(f"rs.phase IN ({placeholders})")
        if self.input.study_type:
            filters.append(f"rs.study_type = {self._add_param(self.input.study_type.value)}")
        if self.input.intervention_type:
            param = self._add_param([i.value for i in self.input.intervention_type])
            filters.append(f"rs.intervention_types && {param}::text[]")
        if self.input.outcome_type and self.input.outcome_type != "all":
            outcome = self._add_param(self.input.outcome_type.upper())
            filters.append(
                f"EXISTS (SELECT 1 FROM ctgov_outcomes o WHERE o.nct_id = rs.nct_id AND o.outcome_type = {outcome})"
            )
        if self.input.start_date_from:
            filters.append(f"rs.start_date_parsed >= {self._add_param(self.input.start_date_from)}")
        if self.input.start_date_to:
            filters.append(f"rs.start_date_parsed <= {self._add_param(self.input.start_date_to)}")
        if self.input.completion_date_from:
            filters.append(f"rs.completion_date_parsed >= {self._add_param(self.input.completion_date_from)}")
        if self.input.completion_date_to:
            filters.append(f"rs.completion_date_parsed <= {self._add_param(self.input.completion_date_to)}")
        if self.input.min_enrollment is not None:
            filters.append(f"rs.enrollment >= {self._add_param(self.input.min_enrollment)}")
        if self.input.max_enrollment is not None:
            filters.append(f"rs.enrollment <= {self._add_param(self.input.max_enrollment)}")
        if self.input.has_results is not None:
            filters.append("rs.results_first_submitted_date IS NOT NULL" if self.input.has_results else "rs.results_first_submitted_date IS NULL")
        if self.input.is_fda_regulated is not None:
            if self.input.is_fda_regulated:
                filters.append("(rs.is_fda_regulated_drug = true OR rs.is_fda_regulated_device = true)")
            else:
                filters.append("(rs.is_fda_regulated_drug IS NOT TRUE AND rs.is_fda_regulated_device IS NOT TRUE)")
        if self.input.country:
            normalized = [c.lower() for c in self.input.country]
            param = self._add_param(normalized)
            filters.append(
                f"EXISTS (SELECT 1 FROM unnest(rs.countries) AS c WHERE LOWER(c) = ANY({param}::text[]))"
            )
        if self.input.eligibility_gender and self.input.eligibility_gender != "all":
            gender = self._add_param(self.input.eligibility_gender.upper())
            filters.append(f"(e.gender = {gender} OR e.gender = 'ALL')")
        if self.input.eligibility_age_range:
            min_age, max_age = self.input.eligibility_age_range
            min_param = self._add_param(min_age)
            max_param = self._add_param(max_age)
            min_expr = self._age_to_years_expr("e.minimum_age")
            max_expr = self._age_to_years_expr("e.maximum_age")
            filters.append(f"({min_expr} IS NULL OR {min_expr} <= {max_param})")
            filters.append(f"({max_expr} IS NULL OR {max_expr} >= {min_param})")
        return filters

    def _build_order_clause(self) -> str:
        direction = "DESC" if self.input.sort_order == SortOrder.DESC else "ASC"
        field_map = {
            SortField.RELEVANCE: "score", SortField.START_DATE: "rs.start_date_parsed",
            SortField.COMPLETION_DATE: "rs.completion_date_parsed",
            SortField.ENROLLMENT: "rs.enrollment", SortField.NCT_ID: "rs.nct_id",
        }
        sort_field = field_map.get(self.input.sort_by, "score")
        if self.input.sort_by == SortField.RELEVANCE:
            direction = "DESC"
        return f"ORDER BY {sort_field} {direction} NULLS LAST, rs.nct_id ASC"


# =============================================================================
# SEARCH EXECUTOR
# =============================================================================

class ClinicalTrialSearcher:
    """Executes clinical trial searches."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self._async_config: AsyncDatabaseConfig | None = None

    async def _get_conn(self) -> AsyncDatabaseConfig:
        if self._async_config is None:
            self._async_config = await get_async_connection(self.db_config)
        return self._async_config

    async def _get_semantic_condition_terms(self, condition: str, top_k: int = 3) -> list[str]:
        """Return semantically-nearest indication terms for condition fallback."""
        try:
            query_vector = encode_query_vector(condition)
            conn = await self._get_conn()
            rows = await conn.execute_query(
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
            terms: list[str] = []
            for row in rows:
                candidate = (row.get("preferred_name") or "").strip()
                if not candidate:
                    continue
                if normalize_semantic_text(candidate) == base:
                    continue
                terms.append(candidate)
                if len(terms) >= top_k:
                    break
            return terms
        except Exception:
            return []

    async def search(self, search_input: ClinicalTrialsSearchInput) -> ClinicalTrialsSearchOutput:
        errors = search_input.validate_inputs()
        if errors:
            return ClinicalTrialsSearchOutput(
                status="error",
                error="; ".join(errors),
            )
        if not search_input.has_any_query() and not self._has_filters(search_input):
            return ClinicalTrialsSearchOutput(
                status="error",
                error="No search criteria provided. Specify condition, intervention, keyword, nct_ids, or filters."
            )
        
        try:
            enriched_input = search_input
            if search_input.condition:
                semantic_terms = await self._get_semantic_condition_terms(search_input.condition)
                if semantic_terms:
                    enriched_input = search_input.model_copy(
                        update={"condition_semantic_terms": semantic_terms}
                    )
            builder = ClinicalTrialQueryBuilder(enriched_input)
            sql, params = builder.build()
            
            conn = await self._get_conn()
            rows = await conn.execute_query(sql, *params)
            
            if not rows:
                return ClinicalTrialsSearchOutput(
                    status="not_found",
                    query_summary=self._build_query_summary(search_input),
                    filters_applied=self._build_filter_summary(search_input),
                )
            
            has_more = len(rows) > search_input.limit
            if has_more:
                rows = rows[:search_input.limit]
            
            hits = []
            for row in rows:
                relevance_breakdown = {}
                if row.get('condition_score') is not None:
                    relevance_breakdown["condition"] = float(row.get('condition_score') or 0)
                if row.get('intervention_score') is not None:
                    relevance_breakdown["intervention"] = float(row.get('intervention_score') or 0)
                if row.get('keyword_score') is not None:
                    relevance_breakdown["keyword"] = float(row.get('keyword_score') or 0)
                if row.get('match_reasons') and "sponsor" in (row.get('match_reasons') or []):
                    relevance_breakdown["sponsor"] = 1.0

                hit = TrialSearchHit(
                    nct_id=row['nct_id'], score=float(row['score']),
                    brief_title=row['brief_title'] or "", phase=row['phase'], status=row['overall_status'],
                    enrollment=row['enrollment'], start_date=row['start_date'],
                    completion_date=row['completion_date'], lead_sponsor=row['lead_sponsor'],
                    conditions=row['conditions'] or [], interventions=row['interventions'] or [],
                    intervention_types=row['intervention_types'] or [], countries=row['countries'] or [],
                    match_reasons=row['match_reasons'] or [],
                    condition_score=row.get('condition_score'),
                    intervention_score=row.get('intervention_score'),
                    keyword_score=row.get('keyword_score'),
                    relevance_breakdown=relevance_breakdown,
                    study_json=row.get('study_json'),
                )
                
                if hit.study_json and search_input.include_study_json:
                    hit.rendered_summary = render_study_text_full(
                        _as_json_dict(hit.study_json), nct_id=hit.nct_id,
                        output_eligibility=search_input.output_eligibility,
                        output_groups=search_input.output_groups,
                        output_baseline_measurements=search_input.output_baseline_measurements,
                        output_results=search_input.output_results,
                        output_adverse_effects=search_input.output_adverse_effects,
                        output_sponsors=search_input.output_sponsors,
                        output_countries=search_input.output_countries,
                    )
                hits.append(hit)
            
            return ClinicalTrialsSearchOutput(
                status="success", total_hits=len(hits), hits=hits,
                query_summary=self._build_query_summary(search_input),
                filters_applied=self._build_filter_summary(search_input),
                limit=search_input.limit, offset=search_input.offset, has_more=has_more,
            )
        except Exception as e:
            return ClinicalTrialsSearchOutput(
                status="error", error=f"{type(e).__name__}: {e}",
                query_summary=self._build_query_summary(search_input),
            )

    def _has_filters(self, inp: ClinicalTrialsSearchInput) -> bool:
        return any([
            inp.status, inp.phase, inp.study_type, inp.intervention_type,
            inp.start_date_from, inp.start_date_to, inp.completion_date_from, inp.completion_date_to,
            inp.min_enrollment is not None, inp.max_enrollment is not None,
            inp.has_results is not None, inp.is_fda_regulated is not None,
            inp.outcome_type, inp.eligibility_gender, inp.eligibility_age_range, inp.country,
        ])

    def _build_query_summary(self, inp: ClinicalTrialsSearchInput) -> str:
        parts = []
        if inp.condition: parts.append(f"condition='{inp.condition}'")
        if inp.intervention: parts.append(f"intervention='{inp.intervention}'")
        if inp.keyword: parts.append(f"keyword='{inp.keyword}'")
        if inp.sponsor: parts.append(f"sponsor='{inp.sponsor}'")
        if inp.nct_ids: parts.append(f"nct_ids={inp.nct_ids}")
        return " AND ".join(parts) if parts else "all trials"

    def _build_filter_summary(self, inp: ClinicalTrialsSearchInput) -> list[str]:
        filters = []
        if inp.status: filters.append(f"status in [{', '.join(s.display_name for s in inp.status)}]")
        if inp.phase: filters.append(f"phase in [{', '.join(p.display_name for p in inp.phase)}]")
        if inp.study_type: filters.append(f"study_type={inp.study_type.display_name}")
        if inp.intervention_type: filters.append(f"intervention_type in [{', '.join(i.display_name for i in inp.intervention_type)}]")
        if inp.outcome_type: filters.append(f"outcome_type={inp.outcome_type}")
        if inp.start_date_from: filters.append(f"start_date >= {inp.start_date_from}")
        if inp.start_date_to: filters.append(f"start_date <= {inp.start_date_to}")
        if inp.completion_date_from: filters.append(f"completion_date >= {inp.completion_date_from}")
        if inp.completion_date_to: filters.append(f"completion_date <= {inp.completion_date_to}")
        if inp.min_enrollment is not None: filters.append(f"enrollment >= {inp.min_enrollment}")
        if inp.max_enrollment is not None: filters.append(f"enrollment <= {inp.max_enrollment}")
        if inp.has_results is not None: filters.append(f"has_results={inp.has_results}")
        if inp.is_fda_regulated is not None: filters.append(f"fda_regulated={inp.is_fda_regulated}")
        if inp.eligibility_gender: filters.append(f"eligibility_gender={inp.eligibility_gender}")
        if inp.eligibility_age_range: filters.append(f"eligibility_age_range={inp.eligibility_age_range}")
        if inp.country: filters.append(f"country={', '.join(inp.country)}")
        return filters


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def clinical_trials_search_async(
    db_config: DatabaseConfig,
    search_input: ClinicalTrialsSearchInput,
) -> ClinicalTrialsSearchOutput:
    """
    Execute a clinical trial search using the provided input model.

    Args:
        db_config: Database configuration for the PostgreSQL source.
        search_input: ClinicalTrialsSearchInput with queries, filters, and options.

    Returns:
        ClinicalTrialsSearchOutput with status, hits, and summary metadata.
    """
    searcher = ClinicalTrialSearcher(db_config)
    return await searcher.search(search_input)