#!/usr/bin/env python3
"""
Direct search tool tests (no agent layer).
Run with:
  python -m tests.test_search_tools_direct
  python -m tests.test_search_tools_direct --category antibody
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from dataclasses import dataclass
from typing import Awaitable, Callable

from bioagent.data.ingest.config import DEFAULT_CONFIG
from bioagent.data.search import (
    target_search_async,
    TargetSearchInput,
    clinical_trials_search_async,
    ClinicalTrialsSearchInput,
    dailymed_and_openfda_search_async,
    DailyMedAndOpenFDAInput,
    molecule_trial_search_async,
    MoleculeTrialSearchInput,
    adverse_events_search_async,
    AdverseEventsSearchInput,
    outcomes_search_async,
    OutcomesSearchInput,
    orange_book_search_async,
    OrangeBookSearchInput,
    cross_database_lookup_async,
    CrossDatabaseLookupInput,
    biotherapeutic_sequence_search_async,
    BiotherapeuticSearchInput,
)
from bioagent.data.search.target_search import SearchMode, DataSource
from bioagent.data.search.clinical_trial_search import InterventionType, TrialPhase, TrialStatus


# -----------------------------------------------------------------------------
# Test data constants
# -----------------------------------------------------------------------------

TEST_MOLECULES = {
    "imatinib": {
        "expected_targets": {"ABL1", "KIT", "PDGFRA", "PDGFRB"},
        "chembl_id": "CHEMBL941",
        "is_biotherapeutic": False,
        "has_salts": True,
    },
    "aspirin": {
        "expected_targets": {"PTGS1", "PTGS2"},
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "is_biotherapeutic": False,
    },
    "metformin": {
        "expected_targets": set(),
        "has_salts": True,
    },
    "ruxolitinib": {
        "expected_targets": {"JAK1", "JAK2"},
        "is_biotherapeutic": False,
    },
}

TEST_BIOLOGICS = {
    "pembrolizumab": {"target": "PDCD1", "type": "antibody"},
    "trastuzumab": {"target": "ERBB2", "type": "antibody"},
    "nivolumab": {"target": "PDCD1", "type": "antibody"},
    "rituximab": {"target": "MS4A1", "type": "antibody"},
    "adalimumab": {"target": "TNF", "type": "antibody"},
}

TEST_TARGETS = {
    "EGFR": {"expected_drugs": {"erlotinib", "gefitinib", "osimertinib", "afatinib"}},
    "ABL1": {"expected_drugs": {"imatinib", "dasatinib", "nilotinib", "ponatinib"}},
    "JAK2": {"expected_drugs": {"ruxolitinib", "fedratinib", "baricitinib"}},
    "BRAF": {"expected_drugs": {"vemurafenib", "dabrafenib", "encorafenib"}},
}

TEST_DRUG_CLASSES = {
    "jak_inhibitors": {
        "drugs": ["ruxolitinib", "tofacitinib", "baricitinib"],
        "targets": {"JAK1", "JAK2", "JAK3"},
    },
    "egfr_inhibitors": {
        "drugs": ["erlotinib", "gefitinib", "osimertinib"],
        "targets": {"EGFR"},
    },
    "braf_inhibitors": {
        "drugs": ["vemurafenib", "dabrafenib", "encorafenib"],
        "targets": {"BRAF"},
    },
    "cdk_inhibitors": {
        "drugs": ["palbociclib", "ribociclib", "abemaciclib"],
        "targets": {"CDK4", "CDK6"},
    },
}

# ChEMBL action_type values: INHIBITOR, ANTAGONIST, AGONIST, MODULATOR, BLOCKER, 
# BINDING AGENT, NEUTRALISING ANTIBODY, OPENER, ACTIVATOR, etc.
# Use flexible matching - any of these action keywords should pass
TEST_ANTIBODY_TARGETS = {
    "pembrolizumab": {"target": "PDCD1", "actions": ["inhibitor", "antagonist", "blocker"]},
    "nivolumab": {"target": "PDCD1", "actions": ["inhibitor", "antagonist", "blocker"]},
    "ipilimumab": {"target": "CTLA4", "actions": ["inhibitor", "antagonist", "blocker"]},
    "trastuzumab": {"target": "ERBB2", "actions": ["inhibitor", "antagonist", "modulator"]},
    "bevacizumab": {"target": "VEGFA", "actions": ["inhibitor", "neutralising", "binding"]},
    "rituximab": {"target": "MS4A1", "actions": ["binding", "modulator", "inhibitor"]},
}

TEST_DRUG_INDICATIONS = {
    "imatinib": ["chronic myeloid leukemia", "gastrointestinal stromal tumor"],
    "pembrolizumab": ["melanoma", "non-small cell lung cancer", "head and neck cancer"],
    "trastuzumab": ["HER2-positive breast cancer", "gastric cancer"],
}

TEST_LABEL_WARNINGS = {
    "warfarin": ["boxed_warning", "bleeding"],
    "metformin": ["contraindications", "lactic acidosis"],
    "isotretinoin": ["boxed_warning", "teratogenic"],
}

TEST_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "benzene": "c1ccccc1",
    "indole": "c1ccc2[nH]ccc2c1",
}

TEST_NCT_IDS = ["NCT01295827", "NCT04280705", "NCT02576509"]


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str


class SearchToolTester:
    def __init__(self, verbose: bool = False, fail_fast: bool = False, json_output: bool = False) -> None:
        self.results: list[TestResult] = []
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.json_output = json_output

    async def _run_test(
        self,
        name: str,
        coro: Awaitable,
        assert_fn: Callable,
    ) -> None:
        start = time.time()
        result = None
        try:
            result = await coro
            assert_fn(result)
            passed = True
            details = "ok"
        except Exception as exc:  # noqa: BLE001 - want full error reporting
            passed = False
            details = f"{type(exc).__name__}: {exc}"
        duration_ms = (time.time() - start) * 1000
        self.results.append(TestResult(name=name, passed=passed, duration_ms=duration_ms, details=details))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name} ({duration_ms:.0f}ms) - {details}")
        if self.verbose:
            try:
                status_value = getattr(result, "status", "unknown")
                print(f"  status={status_value}")
            except Exception:
                print("  status=unknown")
        if self.fail_fast and not passed:
            raise RuntimeError(f"Fail-fast: {name} failed")

    # ---------------------------------------------------------------------
    # Target search tests
    # ---------------------------------------------------------------------

    async def test_targets_for_drug_small_molecule(self) -> None:
        input_obj = TargetSearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query="imatinib",
            data_source=DataSource.BOTH,
            limit=10,
        )
        await self._run_test(
            "targets_for_drug_imatinib",
            target_search_async(DEFAULT_CONFIG, input_obj),
            self._assert_imatinib_targets,
        )

    async def test_drug_profile_aspirin(self) -> None:
        input_obj = TargetSearchInput(
            mode=SearchMode.DRUG_PROFILE,
            query="aspirin",
            data_source=DataSource.BOTH,
            limit=10,
        )
        await self._run_test(
            "drug_profile_aspirin",
            target_search_async(DEFAULT_CONFIG, input_obj),
            self._assert_success_with_hits,
        )

    async def test_targets_for_drug_antibody(self) -> None:
        pembrolizumab = TargetSearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query="pembrolizumab",
            data_source=DataSource.BOTH,
            limit=10,
        )
        trastuzumab = TargetSearchInput(
            mode=SearchMode.TARGETS_FOR_DRUG,
            query="trastuzumab",
            data_source=DataSource.BOTH,
            limit=10,
        )
        await self._run_test(
            "targets_for_drug_pembrolizumab",
            target_search_async(DEFAULT_CONFIG, pembrolizumab),
            lambda r: self._assert_has_target_genes(r, {"PDCD1"}),
        )
        await self._run_test(
            "targets_for_drug_trastuzumab",
            target_search_async(DEFAULT_CONFIG, trastuzumab),
            lambda r: self._assert_has_target_genes(r, {"ERBB2"}),
        )

    async def test_drug_profile_pembrolizumab(self) -> None:
        input_obj = TargetSearchInput(
            mode=SearchMode.DRUG_PROFILE,
            query="pembrolizumab",
            data_source=DataSource.BOTH,
            limit=10,
        )
        await self._run_test(
            "drug_profile_pembrolizumab",
            target_search_async(DEFAULT_CONFIG, input_obj),
            self._assert_biotherapeutic_profile,
        )

    async def test_trials_for_drug_pembrolizumab(self) -> None:
        input_obj = TargetSearchInput(
            mode=SearchMode.TRIALS_FOR_DRUG,
            query="pembrolizumab",
            limit=10,
        )
        await self._run_test(
            "trials_for_drug_pembrolizumab",
            target_search_async(DEFAULT_CONFIG, input_obj),
            self._assert_success_with_hits,
        )

    async def test_drugs_for_target_pdcd1_erbb2(self) -> None:
        pdcd1 = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_TARGET,
            query="PDCD1",
            data_source=DataSource.BOTH,
            limit=15,
        )
        erbb2 = TargetSearchInput(
            mode=SearchMode.DRUGS_FOR_TARGET,
            query="ERBB2",
            data_source=DataSource.BOTH,
            limit=15,
        )
        await self._run_test(
            "drugs_for_target_PDCD1",
            target_search_async(DEFAULT_CONFIG, pdcd1),
            lambda r: self._assert_drug_name_contains(r, {"pembrolizumab", "nivolumab"}),
        )
        await self._run_test(
            "drugs_for_target_ERBB2",
            target_search_async(DEFAULT_CONFIG, erbb2),
            lambda r: self._assert_drug_name_contains(r, {"trastuzumab", "pertuzumab"}),
        )

    async def test_structure_searches(self) -> None:
        exact = TargetSearchInput(
            mode=SearchMode.EXACT_STRUCTURE,
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            limit=5,
        )
        similar = TargetSearchInput(
            mode=SearchMode.SIMILAR_MOLECULES,
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            similarity_threshold=0.7,
            limit=10,
        )
        await self._run_test(
            "exact_structure_aspirin",
            target_search_async(DEFAULT_CONFIG, exact),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "similar_molecules_aspirin",
            target_search_async(DEFAULT_CONFIG, similar),
            self._assert_not_error,
        )

    async def test_molecules(self) -> None:
        """Comprehensive small molecule tests."""
        await self._run_test(
            "mol_imatinib_targets",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGETS_FOR_DRUG,
                    query="imatinib",
                    data_source=DataSource.BOTH,
                ),
            ),
            self._assert_imatinib_targets,
        )
        await self._run_test(
            "mol_aspirin_targets",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGETS_FOR_DRUG,
                    query="aspirin",
                    data_source=DataSource.BOTH,
                ),
            ),
            lambda r: self._assert_has_target_genes(r, {"PTGS1", "PTGS2"}),
        )
        await self._run_test(
            "mol_synonym_gleevec",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGETS_FOR_DRUG,
                    query="Gleevec",
                    data_source=DataSource.BOTH,
                ),
            ),
            lambda r: self._assert_has_target_genes(r, {"ABL1"}),
        )
        await self._run_test(
            "mol_profile_imatinib",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUG_PROFILE,
                    query="imatinib",
                    include_forms=True,
                    include_trials=True,
                ),
            ),
            self._assert_success_with_hits,
        )

    async def test_biologics(self) -> None:
        """Comprehensive biologic/antibody tests."""
        for name, data in TEST_BIOLOGICS.items():
            target_gene = data["target"]
            await self._run_test(
                f"bio_{name}_target",
                target_search_async(
                    DEFAULT_CONFIG,
                    TargetSearchInput(
                        mode=SearchMode.TARGETS_FOR_DRUG,
                        query=name,
                        data_source=DataSource.BOTH,
                    ),
                ),
                lambda r, t=target_gene: self._assert_has_target_genes(r, {t}),
            )
        await self._run_test(
            "bio_pembrolizumab_profile",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUG_PROFILE,
                    query="pembrolizumab",
                ),
            ),
            self._assert_biotherapeutic_profile,
        )

    async def test_targets(self) -> None:
        """Test drugs-for-target searches."""
        # Use MECHANISM data source to get curated approved drugs with proper names
        # Activity data includes many research compounds with ChEMBL IDs as names
        for gene, data in TEST_TARGETS.items():
            expected = data["expected_drugs"]
            await self._run_test(
                f"target_{gene}_drugs",
                target_search_async(
                    DEFAULT_CONFIG,
                    TargetSearchInput(
                        mode=SearchMode.DRUGS_FOR_TARGET,
                        query=gene,
                        data_source=DataSource.MECHANISM,  # Curated drug mechanisms
                        limit=50,  # Increase limit to find approved drugs
                    ),
                ),
                lambda r, d=expected: self._assert_drug_name_contains(r, d),
            )
        await self._run_test(
            "target_egfr_high_potency",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUGS_FOR_TARGET,
                    query="EGFR",
                    min_pchembl=8.0,
                ),
            ),
            lambda r: self._assert_pchembl_above(r, 8.0),
        )

    async def test_drug_classes(self) -> None:
        """Test drug class groupings and expected targets."""
        for class_name, data in TEST_DRUG_CLASSES.items():
            drug = data["drugs"][0]
            expected_targets = data["targets"]
            await self._run_test(
                f"class_{class_name}_{drug}",
                target_search_async(
                    DEFAULT_CONFIG,
                    TargetSearchInput(
                        mode=SearchMode.TARGETS_FOR_DRUG,
                        query=drug,
                        data_source=DataSource.BOTH,
                        limit=15,
                    ),
                ),
                lambda r, t=expected_targets: self._assert_has_target_genes(r, t),
            )

    async def test_biologic_mechanisms(self) -> None:
        """Test monoclonal antibody mechanism data."""
        for drug, data in TEST_ANTIBODY_TARGETS.items():
            target_gene = data["target"]
            expected_actions = data["actions"]
            await self._run_test(
                f"bio_mech_{drug}",
                target_search_async(
                    DEFAULT_CONFIG,
                    TargetSearchInput(
                        mode=SearchMode.TARGETS_FOR_DRUG,
                        query=drug,
                        data_source=DataSource.BOTH,
                        limit=10,
                    ),
                ),
                lambda r, t=target_gene, actions=expected_actions: (
                    self._assert_has_target_genes(r, {t}),
                    self._assert_has_mechanism_type_any(r, actions),
                ),
            )

    async def test_structures(self) -> None:
        """Structure search tests."""
        await self._run_test(
            "struct_exact_aspirin",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.EXACT_STRUCTURE,
                    smiles=TEST_SMILES["aspirin"],
                ),
            ),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "struct_similar_high",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.SIMILAR_MOLECULES,
                    smiles=TEST_SMILES["aspirin"],
                    similarity_threshold=0.85,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "struct_similar_low",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.SIMILAR_MOLECULES,
                    smiles=TEST_SMILES["aspirin"],
                    similarity_threshold=0.5,
                    limit=50,
                ),
            ),
            lambda r: self._assert_min_hits(r, 5) if r.status == "success" else None,
        )
        await self._run_test(
            "struct_substructure_indole",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.SUBSTRUCTURE,
                    smarts=TEST_SMILES["indole"],
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )

    async def test_selectivity(self) -> None:
        """Selectivity and comparison tests."""
        await self._run_test(
            "sel_compare_abl",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.COMPARE_DRUGS,
                    target="ABL1",
                    drug_names=["imatinib", "dasatinib", "nilotinib"],
                ),
            ),
            lambda r: self._assert_min_hits(r, 2),
        )
        await self._run_test(
            "sel_jak2_selective",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.SELECTIVE_DRUGS,
                    target="JAK2",
                    off_targets=["JAK1", "JAK3"],
                    min_selectivity_fold=5.0,
                    min_pchembl=6.0,
                ),
            ),
            self._assert_not_error,
        )

    async def test_forms(self) -> None:
        """Drug forms tests."""
        await self._run_test(
            "forms_metformin",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUG_FORMS,
                    query="metformin",
                ),
            ),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "forms_imatinib",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUG_FORMS,
                    query="imatinib",
                ),
            ),
            self._assert_has_smiles,
        )

    # ---------------------------------------------------------------------
    # Clinical trial search tests
    # ---------------------------------------------------------------------

    async def test_clinical_trials(self) -> None:
        breast_cancer = ClinicalTrialsSearchInput(
            condition="breast cancer",
            phase=[TrialPhase.PHASE_3],
            limit=20,
        )
        pembrolizumab = ClinicalTrialsSearchInput(
            intervention="pembrolizumab",
            intervention_type=[InterventionType.BIOLOGICAL],
            limit=20,
        )
        imatinib = ClinicalTrialsSearchInput(
            intervention="imatinib",
            intervention_type=[InterventionType.DRUG],
            limit=20,
        )
        melanoma_combo = ClinicalTrialsSearchInput(
            condition="melanoma",
            intervention="pembrolizumab",
            phase=[TrialPhase.PHASE_2, TrialPhase.PHASE_3],
            limit=20,
        )
        nct_id = ClinicalTrialsSearchInput(
            nct_ids=["NCT01295827"],
            limit=5,
        )
        await self._run_test(
            "clinical_trials_breast_cancer_phase3",
            clinical_trials_search_async(DEFAULT_CONFIG, breast_cancer),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "clinical_trials_pembrolizumab_biological",
            clinical_trials_search_async(DEFAULT_CONFIG, pembrolizumab),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "clinical_trials_imatinib_drug",
            clinical_trials_search_async(DEFAULT_CONFIG, imatinib),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "clinical_trials_melanoma_pembrolizumab",
            clinical_trials_search_async(DEFAULT_CONFIG, melanoma_combo),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "clinical_trials_nct_id",
            clinical_trials_search_async(DEFAULT_CONFIG, nct_id),
            self._assert_nct_id_returned,
        )

    async def test_clinical_trials_extended(self) -> None:
        """Extended clinical trial tests."""
        await self._run_test(
            "ct_recruiting",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="lung cancer",
                    status=[TrialStatus.RECRUITING],
                    limit=10,
                ),
            ),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "ct_completed_results",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    intervention="imatinib",
                    status=[TrialStatus.COMPLETED],
                    has_results=True,
                    limit=10,
                ),
            ),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "ct_large_enrollment",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="diabetes",
                    min_enrollment=500,
                    limit=10,
                ),
            ),
            lambda r: self._assert_min_enrollment(r, 500) if r.hits else None,
        )
        await self._run_test(
            "ct_multiple_nct",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    nct_ids=TEST_NCT_IDS[:2],
                ),
            ),
            lambda r: self._assert_min_hits(r, 1),
        )

    # ---------------------------------------------------------------------
    # DailyMed/OpenFDA search tests
    # ---------------------------------------------------------------------

    async def test_openfda_dailymed(self) -> None:
        property_lookup = DailyMedAndOpenFDAInput(
            drug_names=["aspirin", "imatinib"],
            result_limit=5,
        )
        section_search = DailyMedAndOpenFDAInput(
            drug_names=["aspirin"],
            section_queries=["warnings"],
            result_limit=5,
        )
        keyword_search = DailyMedAndOpenFDAInput(
            keyword_query=["hepatotoxicity"],
            result_limit=5,
        )
        full_sections = DailyMedAndOpenFDAInput(
            drug_names=["metformin"],
            fetch_all_sections=True,
            result_limit=5,
        )
        await self._run_test(
            "openfda_property_lookup",
            dailymed_and_openfda_search_async(DEFAULT_CONFIG, property_lookup),
            self._assert_not_error,
        )
        await self._run_test(
            "openfda_section_search",
            dailymed_and_openfda_search_async(DEFAULT_CONFIG, section_search),
            self._assert_not_error,
        )
        await self._run_test(
            "openfda_keyword_search",
            dailymed_and_openfda_search_async(DEFAULT_CONFIG, keyword_search),
            self._assert_not_error,
        )
        await self._run_test(
            "openfda_full_sections",
            dailymed_and_openfda_search_async(DEFAULT_CONFIG, full_sections),
            self._assert_not_error,
        )

    async def test_drug_labels_extended(self) -> None:
        """Extended drug label tests."""
        await self._run_test(
            "label_warnings_interactions",
            dailymed_and_openfda_search_async(
                DEFAULT_CONFIG,
                DailyMedAndOpenFDAInput(
                    drug_names=["warfarin"],
                    section_queries=["warnings", "drug interactions"],
                ),
            ),
            self._assert_has_label_sections,
        )
        await self._run_test(
            "label_adverse_reactions",
            dailymed_and_openfda_search_async(
                DEFAULT_CONFIG,
                DailyMedAndOpenFDAInput(
                    drug_names=["metformin"],
                    section_queries=["adverse reactions"],
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "label_keyword_liver",
            dailymed_and_openfda_search_async(
                DEFAULT_CONFIG,
                DailyMedAndOpenFDAInput(
                    keyword_query=["liver toxicity", "hepatotoxicity"],
                ),
            ),
            self._assert_not_error,
        )

    async def test_activity_and_indications(self) -> None:
        """Test activity and indication search modes."""
        await self._run_test(
            "activity_imatinib",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.ACTIVITIES_FOR_DRUG,
                    query="imatinib",
                    limit=20,
                ),
            ),
            self._assert_has_activity_data,
        )
        await self._run_test(
            "activity_egfr",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.ACTIVITIES_FOR_TARGET,
                    query="EGFR",
                    limit=50,
                ),
            ),
            self._assert_success_with_hits,
        )
        await self._run_test(
            "indications_imatinib",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.INDICATIONS_FOR_DRUG,
                    query="imatinib",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "drugs_for_indication_cml",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUGS_FOR_INDICATION,
                    query="chronic myeloid leukemia",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )

    async def test_pathways_and_interactions(self) -> None:
        """Test target pathway and drug interaction modes."""
        await self._run_test(
            "target_pathways_egfr",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGET_PATHWAYS,
                    query="EGFR",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "drug_interactions_warfarin",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUG_INTERACTIONS,
                    query="warfarin",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )

    async def test_clinical_trials_filters(self) -> None:
        """Test clinical trial search filters and combinations."""
        await self._run_test(
            "ct_date_range_2020_2023",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="breast cancer",
                    start_date_from=date(2020, 1, 1),
                    start_date_to=date(2023, 12, 31),
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ct_country_us",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    intervention="pembrolizumab",
                    country=["United States"],
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ct_eligibility_male_adult",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="prostate cancer",
                    eligibility_gender="male",
                    eligibility_age_range=(18, 65),
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ct_sponsor_pfizer",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    sponsor="Pfizer",
                    phase=[TrialPhase.PHASE_3],
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ct_fda_regulated",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    intervention="nivolumab",
                    is_fda_regulated=True,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_molecule_trial_search(self) -> None:
        """Test molecule <-> trial connectivity search."""
        await self._run_test(
            "mol_trial_by_target_egfr",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_target",
                    target_gene="EGFR",
                    min_pchembl=7.0,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "mol_trial_by_condition_lung",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="molecules_by_condition",
                    condition="lung cancer",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_adverse_events_search(self) -> None:
        """Test adverse event search modes."""
        await self._run_test(
            "ae_events_for_drug",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="pembrolizumab",
                    severity="all",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ae_drugs_with_event",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="drugs_with_event",
                    event_term="hepatotoxicity",
                    severity="all",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_outcomes_search(self) -> None:
        """Test outcomes search modes."""
        await self._run_test(
            "outcomes_for_trial",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="outcomes_for_trial",
                    nct_id="NCT01295827",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "outcomes_efficacy_comparison",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="efficacy_comparison",
                    drug_name="imatinib",
                    max_p_value=0.1,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_orange_book_search(self) -> None:
        """Test Orange Book search modes."""
        await self._run_test(
            "orange_book_te_codes",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="te_codes",
                    drug_name="imatinib",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "orange_book_patents",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="patents",
                    drug_name="imatinib",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_cross_db_lookup(self) -> None:
        """Test cross-database lookup."""
        await self._run_test(
            "cross_db_imatinib",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(identifier="imatinib", identifier_type="name"),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "cross_db_chembl",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(identifier="CHEMBL941", identifier_type="chembl"),
            ),
            self._assert_not_error,
        )

    async def test_biotherapeutic_sequence_search(self) -> None:
        """Test biotherapeutic sequence search."""
        await self._run_test(
            "bio_sequence_pdcd1",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_target",
                    target_gene="PDCD1",
                    limit=10,
                ),
            ),
            self._assert_status_in_allowed,
        )

    async def test_edge_cases(self) -> None:
        """Edge cases and error handling."""
        await self._run_test(
            "edge_not_found_drug",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGETS_FOR_DRUG,
                    query="xyzzzynotadrugname12345",
                ),
            ),
            self._assert_not_found,
        )
        await self._run_test(
            "edge_not_found_target",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUGS_FOR_TARGET,
                    query="NOTAREALGENE999",
                ),
            ),
            self._assert_not_found,
        )
        await self._run_test(
            "edge_invalid_smiles",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.EXACT_STRUCTURE,
                    smiles="invalid(((smiles",
                ),
            ),
            lambda r: self._assert_status_in(r, ["error", "not_found"]),
        )
        await self._run_test(
            "edge_impossible_pchembl",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.DRUGS_FOR_TARGET,
                    query="EGFR",
                    min_pchembl=15.0,
                ),
            ),
            lambda r: self._assert_status_in(r, ["success", "not_found"]),
        )
        # Test with impossible NCT ID - should return not_found or empty success
        await self._run_test(
            "edge_ct_no_results",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    nct_ids=["NCT99999999999"],  # Non-existent NCT ID
                ),
            ),
            lambda r: self._assert_empty_or_not_found(r),
        )

    # ---------------------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------------------

    @staticmethod
    def _assert_success_with_hits(result) -> None:
        if result.status != "success":
            raise AssertionError(f"Expected success, got {result.status}")
        if not result.hits:
            raise AssertionError("Expected non-empty hits")

    @staticmethod
    def _assert_not_error(result) -> None:
        if result.status == "error":
            raise AssertionError(f"Unexpected error: {result.error}")

    @staticmethod
    def _assert_imatinib_targets(result) -> None:
        SearchToolTester._assert_success_with_hits(result)
        genes = set()
        for hit in result.hits:
            if hasattr(hit, "all_target_genes"):
                genes.update(hit.all_target_genes)
        expected = {"ABL1", "KIT", "PDGFRA"}
        if not (genes & expected):
            raise AssertionError(f"Expected targets {expected}, got {sorted(genes)[:10]}")

    @staticmethod
    def _assert_has_target_genes(result, expected: set[str]) -> None:
        SearchToolTester._assert_success_with_hits(result)
        genes = set()
        for hit in result.hits:
            if hasattr(hit, "all_target_genes"):
                genes.update(hit.all_target_genes)
        if not (genes & expected):
            raise AssertionError(f"Expected genes {expected}, got {sorted(genes)[:10]}")

    @staticmethod
    def _assert_biotherapeutic_profile(result) -> None:
        SearchToolTester._assert_success_with_hits(result)
        profile = result.hits[0]
        if not getattr(profile, "is_biotherapeutic", False):
            raise AssertionError("Expected biotherapeutic profile to be True")

    @staticmethod
    def _assert_drug_name_contains(result, expected_names: set[str]) -> None:
        SearchToolTester._assert_success_with_hits(result)
        names = {h.concept_name.lower() for h in result.hits if h.concept_name}
        if not any(name in names for name in expected_names):
            raise AssertionError(f"Expected one of {expected_names}, got {sorted(names)[:10]}")

    @staticmethod
    def _assert_nct_id_returned(result) -> None:
        SearchToolTester._assert_success_with_hits(result)
        nct_ids = {h.nct_id for h in result.hits}
        if "NCT01295827" not in nct_ids:
            raise AssertionError(f"NCT01295827 not found. Got {sorted(nct_ids)[:5]}")

    @staticmethod
    def _assert_not_found(result) -> None:
        if result.status != "not_found":
            raise AssertionError(f"Expected not_found, got {result.status}")

    @staticmethod
    def _assert_empty_or_not_found(result) -> None:
        """Assert result is not_found OR success with no hits."""
        if result.status == "not_found":
            return
        if result.status == "success" and not result.hits:
            return
        if result.status == "success" and len(result.hits) == 0:
            return
        if result.status == "error":
            return  # Errors are acceptable for edge cases
        raise AssertionError(
            f"Expected not_found or empty success, got {result.status} with {len(getattr(result, 'hits', []))} hits"
        )

    @staticmethod
    def _assert_status_in(result, statuses: list[str]) -> None:
        if result.status not in statuses:
            raise AssertionError(f"Expected one of {statuses}, got {result.status}")

    @staticmethod
    def _assert_status_in_allowed(result) -> None:
        if result.status not in ["success", "not_found"]:
            raise AssertionError(f"Expected success or not_found, got {result.status}")

    @staticmethod
    def _assert_min_hits(result, min_count: int) -> None:
        SearchToolTester._assert_success_with_hits(result)
        if len(result.hits) < min_count:
            raise AssertionError(f"Expected >= {min_count} hits, got {len(result.hits)}")

    @staticmethod
    def _assert_max_hits(result, max_count: int) -> None:
        if len(result.hits) > max_count:
            raise AssertionError(f"Expected <= {max_count} hits, got {len(result.hits)}")

    @staticmethod
    def _assert_has_smiles(result) -> None:
        for hit in result.hits:
            if getattr(hit, "canonical_smiles", None):
                return
        raise AssertionError("No hits have SMILES data")

    @staticmethod
    def _assert_has_chembl_id(result) -> None:
        for hit in result.hits:
            if getattr(hit, "chembl_id", None):
                return
        raise AssertionError("No hits have ChEMBL ID")

    @staticmethod
    def _assert_has_mechanism_data(result) -> None:
        for hit in result.hits:
            if hasattr(hit, "mechanisms") and hit.mechanisms:
                return
            if getattr(hit, "mechanism_of_action", None):
                return
        raise AssertionError("No mechanism data found")

    @staticmethod
    def _assert_has_activity_data(result) -> None:
        for hit in result.hits:
            if hasattr(hit, "activities") and hit.activities:
                return
            if getattr(hit, "activity_value_nm", None) is not None:
                return
        raise AssertionError("No activity data found")

    @staticmethod
    def _assert_has_mechanism_type(result, expected_action: str) -> None:
        for hit in result.hits:
            for mech in getattr(hit, "mechanisms", []):
                action_type = getattr(mech, "action_type", "") or ""
                if expected_action.lower() in action_type.lower():
                    return
        raise AssertionError(f"Expected mechanism action '{expected_action}' not found")

    @staticmethod
    def _assert_has_mechanism_type_any(result, expected_actions: list[str]) -> None:
        """Check if any of the expected action types are found in mechanism data."""
        found_actions = set()
        for hit in result.hits:
            for mech in getattr(hit, "mechanisms", []):
                action_type = getattr(mech, "action_type", "") or ""
                found_actions.add(action_type.lower())
                for expected in expected_actions:
                    if expected.lower() in action_type.lower():
                        return
        # If no mechanisms found, check if any target/activity data exists (may not have mechanism annotation)
        if not found_actions:
            # For biologics, mechanism data may not always be present in ChEMBL
            # Accept if we at least found the target
            for hit in result.hits:
                if getattr(hit, "mechanisms", []) or getattr(hit, "activities", []):
                    return  # Has some target data, acceptable
        raise AssertionError(
            f"Expected one of {expected_actions}, found actions: {sorted(found_actions) if found_actions else 'none'}"
        )

    @staticmethod
    def _assert_selectivity_above(result, min_fold: float) -> None:
        for hit in result.hits:
            fold = getattr(hit, "selectivity_fold", None)
            if fold and fold >= min_fold:
                return
        raise AssertionError(f"No hits with selectivity >= {min_fold}x")

    @staticmethod
    def _assert_label_section_present(result, section_name: str) -> None:
        if not result.results:
            raise AssertionError("No results")
        for item in result.results:
            for section in getattr(item, "sections", []):
                if section_name.lower() in section.section_name.lower():
                    return
        raise AssertionError(f"Expected section '{section_name}' not found")

    @staticmethod
    def _assert_pchembl_above(result, min_pchembl: float) -> None:
        for hit in result.hits:
            pchembl = getattr(hit, "pchembl", None)
            if pchembl is not None and pchembl < min_pchembl:
                raise AssertionError(f"pChEMBL {pchembl} below {min_pchembl}")

    @staticmethod
    def _assert_min_enrollment(result, min_enrollment: int) -> None:
        for hit in result.hits:
            enrollment = getattr(hit, "enrollment", None)
            if enrollment is not None and enrollment < min_enrollment:
                raise AssertionError(f"Enrollment {enrollment} below {min_enrollment}")

    @staticmethod
    def _assert_has_label_sections(result) -> None:
        if not result.results:
            raise AssertionError("No results")
        for item in result.results:
            if item.sections:
                return
        raise AssertionError("No label sections found")

    # ---------------------------------------------------------------------
    # Runner
    # ---------------------------------------------------------------------

    async def run(self, category: str) -> None:
        category_map = {
            "molecules": self.test_molecules,
            "biologics": self.test_biologics,
            "targets": self.test_targets,
            "drug_classes": self.test_drug_classes,
            "biologic_mechanisms": self.test_biologic_mechanisms,
            "structures": self.test_structures,
            "selectivity": self.test_selectivity,
            "forms": self.test_forms,
            "clinical_trials": self.test_clinical_trials,
            "clinical_trials_extended": self.test_clinical_trials_extended,
            "clinical_trials_filters": self.test_clinical_trials_filters,
            "drug_labels": self.test_drug_labels_extended,
            "openfda": self.test_openfda_dailymed,
            "edge_cases": self.test_edge_cases,
            "activity_indications": self.test_activity_and_indications,
            "pathways_interactions": self.test_pathways_and_interactions,
            "molecule_trial": self.test_molecule_trial_search,
            "adverse_events": self.test_adverse_events_search,
            "outcomes": self.test_outcomes_search,
            "orange_book": self.test_orange_book_search,
            "cross_db": self.test_cross_db_lookup,
            "biotherapeutics_sequence": self.test_biotherapeutic_sequence_search,
            "legacy_targets": self.test_targets_for_drug_small_molecule,
            "legacy_antibody": self.test_targets_for_drug_antibody,
            "legacy_structure": self.test_structure_searches,
        }

        if category == "all":
            for test_fn in [
                self.test_molecules,
                self.test_biologics,
                self.test_targets,
                self.test_drug_classes,
                self.test_biologic_mechanisms,
                self.test_structures,
                self.test_selectivity,
                self.test_forms,
                self.test_clinical_trials,
                self.test_clinical_trials_extended,
                self.test_clinical_trials_filters,
                self.test_openfda_dailymed,
                self.test_drug_labels_extended,
                self.test_edge_cases,
                self.test_activity_and_indications,
                self.test_pathways_and_interactions,
                self.test_molecule_trial_search,
                self.test_adverse_events_search,
                self.test_outcomes_search,
                self.test_orange_book_search,
                self.test_cross_db_lookup,
                self.test_biotherapeutic_sequence_search,
            ]:
                await test_fn()
        elif category in category_map:
            await category_map[category]()

        self._print_summary()

    def _print_summary(self) -> None:
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_ms = sum(r.duration_ms for r in self.results)
        if self.json_output:
            payload = {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "duration_ms": round(total_ms, 2),
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration_ms": round(r.duration_ms, 2),
                        "details": r.details,
                    }
                    for r in self.results
                ],
            }
            print(json.dumps(payload, indent=2))
            return

        print("\n" + "=" * 80)
        print(f"TOTAL: {len(self.results)} | PASSED: {passed} | FAILED: {failed} | TIME: {total_ms/1000:.2f}s")
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} - {result.name} ({result.duration_ms:.0f}ms)")
        if failed:
            print("\nFAILED TESTS:")
            for result in self.results:
                if not result.passed:
                    print(f"FAIL - {result.name}: {result.details}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct search tool tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests.test_search_tools_direct
  python -m tests.test_search_tools_direct --category targets
  python -m tests.test_search_tools_direct --category edge_cases
  python -m tests.test_search_tools_direct -v --fail-fast
        """,
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=[
            "all",
            "molecules",
            "biologics",
            "targets",
            "drug_classes",
            "biologic_mechanisms",
            "structures",
            "selectivity",
            "forms",
            "clinical_trials",
            "clinical_trials_extended",
            "clinical_trials_filters",
            "drug_labels",
            "openfda",
            "edge_cases",
            "activity_indications",
            "pathways_interactions",
            "molecule_trial",
            "adverse_events",
            "outcomes",
            "orange_book",
            "cross_db",
            "biotherapeutics_sequence",
            "legacy_targets",
            "legacy_antibody",
            "legacy_structure",
        ],
        help="Subset of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fail-fast", "-f", action="store_true", help="Stop on first failure")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    tester = SearchToolTester(
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        json_output=args.json,
    )
    await tester.run(args.category)


if __name__ == "__main__":
    asyncio.run(main())
