#!/usr/bin/env python3
"""
Search module exports for biomedical data sources.

Re-exports async search functions and their input/output models to provide a
single import surface for database-backed search capabilities.
"""

from .openfda_and_dailymed_search import dailymed_and_openfda_search_async, DailyMedAndOpenFDAInput, DailyMedAndOpenFDASearchOutput
from .clinical_trial_search import clinical_trials_search_async, ClinicalTrialsSearchInput, ClinicalTrialsSearchOutput
from .target_search import target_search_async, TargetSearchInput, TargetSearchOutput
from .molecule_trial_search import molecule_trial_search_async, MoleculeTrialSearchInput, MoleculeTrialSearchOutput
from .adverse_events_search import adverse_events_search_async, AdverseEventsSearchInput, AdverseEventsSearchOutput
from .outcomes_search import outcomes_search_async, OutcomesSearchInput, OutcomesSearchOutput
from .orange_book_search import orange_book_search_async, OrangeBookSearchInput, OrangeBookSearchOutput
from .cross_db_lookup import cross_database_lookup_async, CrossDatabaseLookupInput, CrossDatabaseLookupOutput
from .biotherapeutic_sequence_search import (
    biotherapeutic_sequence_search_async,
    BiotherapeuticSearchInput,
    BiotherapeuticSearchOutput,
)


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
    "molecule_trial_search_async",
    "MoleculeTrialSearchInput",
    "MoleculeTrialSearchOutput",
    "adverse_events_search_async",
    "AdverseEventsSearchInput",
    "AdverseEventsSearchOutput",
    "outcomes_search_async",
    "OutcomesSearchInput",
    "OutcomesSearchOutput",
    "orange_book_search_async",
    "OrangeBookSearchInput",
    "OrangeBookSearchOutput",
    "cross_database_lookup_async",
    "CrossDatabaseLookupInput",
    "CrossDatabaseLookupOutput",
    "biotherapeutic_sequence_search_async",
    "BiotherapeuticSearchInput",
    "BiotherapeuticSearchOutput",
]
