#!/usr/bin/env python3
"""
CureBench Data App Module

This module provides search and query functionality for the drug databases.
All search functions have been separated from ingestion logic for cleaner organization.
"""

from .openfda_and_dailymed_searches import unified_search_async, UnifiedSearchInput, EnrichedSearchOutput
from .clinical_trial_searches import clinical_trials_search, ClinicalTrialsSearchInput, ClinicalTrialsSearchOutput, SearchKind


__all__ = [
    # Synchronous search modules
    "unified_search_async",
    "UnifiedSearchInput",
    "EnrichedSearchOutput",
    "clinical_trials_search",
    "ClinicalTrialsSearchInput",
    "ClinicalTrialsSearchOutput",
    "SearchKind",
]
