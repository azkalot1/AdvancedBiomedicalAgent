#!/usr/bin/env python3
"""
CureBench Data App Module

This module provides search and query functionality for the drug databases.
All search functions have been separated from ingestion logic for cleaner organization.
"""

from .openfda_and_dailymed_search import dailymed_and_openfda_search_async, DailyMedAndOpenFDAInput, DailyMedAndOpenFDASearchOutput
from .clinical_trial_search import clinical_trials_search_async, ClinicalTrialsSearchInput, ClinicalTrialsSearchOutput
from .target_search import target_search_async, TargetSearchInput, TargetSearchOutput


__all__ = [
    # Synchronous search modules
    "dailymed_and_openfda_search_async",
    "DailyMedAndOpenFDAInput",
    "DailyMedAndOpenFDASearchOutput",
    "clinical_trials_search_async",
    "ClinicalTrialsSearchInput",
    "ClinicalTrialsSearchOutput", 
    "target_search_async",
    "TargetSearchInput",
    "TargetSearchOutput",
]
