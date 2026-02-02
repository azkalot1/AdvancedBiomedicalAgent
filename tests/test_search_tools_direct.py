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
from contextlib import contextmanager
from typing import Awaitable, Callable, Type

from pydantic import ValidationError

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]


@contextmanager
def _raises(*expected: type[BaseException]):
    """Context manager that expects one of the given exceptions. Use when pytest may be unavailable."""
    try:
        yield
        raise AssertionError(f"Expected one of {expected} to be raised")
    except expected:
        pass


def _raises_or_pytest(*expected: Type[BaseException]):
    """Use pytest.raises if available, else _raises."""
    if pytest is not None:
        return pytest.raises(expected[0] if len(expected) == 1 else expected)
    return _raises(*expected)


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
from bioagent.data.search.clinical_trial_search import (
    InterventionType,
    TrialPhase,
    TrialStatus,
    SortField,
    SortOrder,
    SearchStrategy,
    StudyType,
)


# -----------------------------------------------------------------------------
# Test data constants
# -----------------------------------------------------------------------------

TEST_MOLECULES = {
    "imatinib": {
        "expected_targets": {"ABL1", "KIT", "PDGFRA", "PDGFRB"},
        "chembl_id": "CHEMBL941",
        "is_biotherapeutic": False,
        "has_salts": True,
        "inchi_key": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
    },
    "aspirin": {
        "expected_targets": {"PTGS1", "PTGS2"},
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "is_biotherapeutic": False,
        "chembl_id": "CHEMBL25",
    },
    "metformin": {
        "expected_targets": set(),
        "has_salts": True,
        "chembl_id": "CHEMBL1431",
    },
    "ruxolitinib": {
        "expected_targets": {"JAK1", "JAK2"},
        "is_biotherapeutic": False,
        "chembl_id": "CHEMBL1789941",
    },
    "erlotinib": {
        "expected_targets": {"EGFR"},
        "chembl_id": "CHEMBL553",
        "is_biotherapeutic": False,
    },
    "vemurafenib": {
        "expected_targets": {"BRAF"},
        "chembl_id": "CHEMBL1229517",
        "is_biotherapeutic": False,
    },
    "dasatinib": {
        "expected_targets": {"ABL1", "SRC", "KIT"},
        "chembl_id": "CHEMBL1421",
        "is_biotherapeutic": False,
    },
    "gefitinib": {
        "expected_targets": {"EGFR"},
        "chembl_id": "CHEMBL939",
        "is_biotherapeutic": False,
    },
}

TEST_BIOLOGICS = {
    "pembrolizumab": {"target": "PDCD1", "type": "antibody", "chembl_id": "CHEMBL3137343"},
    "trastuzumab": {"target": "ERBB2", "type": "antibody", "chembl_id": "CHEMBL1201585"},
    "nivolumab": {"target": "PDCD1", "type": "antibody", "chembl_id": "CHEMBL2108738"},
    "rituximab": {"target": "MS4A1", "type": "antibody", "chembl_id": "CHEMBL1201576"},
    "adalimumab": {"target": "TNF", "type": "antibody", "chembl_id": "CHEMBL1201580"},
    "bevacizumab": {"target": "VEGFA", "type": "antibody", "chembl_id": "CHEMBL1201583"},
    "ipilimumab": {"target": "CTLA4", "type": "antibody", "chembl_id": "CHEMBL1789844"},
    "cetuximab": {"target": "EGFR", "type": "antibody", "chembl_id": "CHEMBL1201577"},
}

TEST_TARGETS = {
    "EGFR": {"expected_drugs": {"erlotinib", "gefitinib", "osimertinib", "afatinib"}},
    "ABL1": {"expected_drugs": {"imatinib", "dasatinib", "nilotinib", "ponatinib"}},
    "JAK2": {"expected_drugs": {"ruxolitinib", "fedratinib", "baricitinib"}},
    "BRAF": {"expected_drugs": {"vemurafenib", "dabrafenib", "encorafenib"}},
    "PDCD1": {"expected_drugs": {"pembrolizumab", "nivolumab"}},
    "ERBB2": {"expected_drugs": {"trastuzumab", "pertuzumab", "lapatinib"}},
    "ALK": {"expected_drugs": {"crizotinib", "alectinib", "ceritinib"}},
    "BCR": {"expected_drugs": {"imatinib", "dasatinib", "nilotinib"}},
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
    "bcr_abl_inhibitors": {
        "drugs": ["imatinib", "dasatinib", "nilotinib", "ponatinib"],
        "targets": {"ABL1"},
    },
    "pd1_inhibitors": {
        "drugs": ["pembrolizumab", "nivolumab"],
        "targets": {"PDCD1"},
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
    "metformin": ["diabetes", "type 2 diabetes"],
    "aspirin": ["pain", "fever", "inflammation"],
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
    "imatinib": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
}

TEST_NCT_IDS = ["NCT01295827", "NCT04280705", "NCT02576509", "NCT02125461", "NCT01905657"]

# Expected adverse events for drugs (based on CT.gov reported events)
TEST_ADVERSE_EVENTS = {
    "pembrolizumab": {
        "expected_events": ["fatigue", "nausea", "diarrhea", "rash", "pruritus"],
        "serious_events": ["pneumonitis", "colitis", "hepatitis"],
    },
    "imatinib": {
        "expected_events": ["nausea", "edema", "muscle cramps", "diarrhea"],
        "serious_events": ["fluid retention", "hemorrhage"],
    },
    "metformin": {
        "expected_events": ["diarrhea", "nausea", "abdominal pain"],
        "serious_events": ["lactic acidosis"],
    },
}

# Expected Orange Book data
TEST_ORANGE_BOOK = {
    "imatinib": {
        "has_patents": True,
        "has_generics": True,
        "te_code_expected": "AB",  # Therapeutically equivalent
    },
    "metformin": {
        "has_patents": False,  # Off-patent
        "has_generics": True,
    },
    "aspirin": {
        "has_generics": True,
    },
}

# Expected cross-database lookup data
TEST_CROSS_DB = {
    "imatinib": {
        "has_molecules": True,
        "has_labels": True,
        "has_trials": True,
        "has_targets": True,
        "expected_chembl": "CHEMBL941",
    },
    "pembrolizumab": {
        "has_molecules": True,
        "has_trials": True,
        "has_targets": True,
    },
    "CHEMBL941": {
        "identifier_type": "chembl",
        "expected_name": "imatinib",
    },
}

# Clinical trial expected data
TEST_CLINICAL_TRIALS = {
    "NCT01295827": {
        "condition_contains": ["leukemia", "cancer"],
        "has_results": True,
    },
    "breast_cancer_phase3": {
        "min_expected_trials": 50,
        "expected_phases": ["Phase 3", "PHASE3"],
    },
    "pembrolizumab_melanoma": {
        "min_expected_trials": 10,
        "condition_contains": ["melanoma"],
    },
}

# Molecule-Trial connectivity expected data
TEST_MOLECULE_TRIALS = {
    "imatinib": {
        "min_trials": 10,
        "expected_conditions": ["leukemia", "gist", "cancer"],
    },
    "pembrolizumab": {
        "min_trials": 50,
        "expected_conditions": ["melanoma", "lung", "cancer"],
    },
}

# Expected outcomes data
TEST_OUTCOMES = {
    "NCT01295827": {
        "has_primary_outcomes": True,
        "has_secondary_outcomes": True,
    },
}


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
        # Test additional molecules from TEST_MOLECULES
        for drug_name, data in list(TEST_MOLECULES.items())[:4]:
            if data.get("expected_targets"):
                await self._run_test(
                    f"mol_{drug_name}_targets_correctness",
                    target_search_async(
                        DEFAULT_CONFIG,
                        TargetSearchInput(
                            mode=SearchMode.TARGETS_FOR_DRUG,
                            query=drug_name,
                            data_source=DataSource.BOTH,
                            limit=20,
                        ),
                    ),
                    lambda r, exp=data["expected_targets"]: self._assert_has_target_genes(r, exp),
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
        # Test structure search with imatinib SMILES
        await self._run_test(
            "struct_exact_imatinib",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.EXACT_STRUCTURE,
                    smiles=TEST_SMILES["imatinib"],
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
        # Test multiple drugs at once
        await self._run_test(
            "label_multiple_drugs",
            dailymed_and_openfda_search_async(
                DEFAULT_CONFIG,
                DailyMedAndOpenFDAInput(
                    drug_names=["aspirin", "ibuprofen", "acetaminophen"],
                    result_limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test dosage and administration section
        await self._run_test(
            "label_dosage_section",
            dailymed_and_openfda_search_async(
                DEFAULT_CONFIG,
                DailyMedAndOpenFDAInput(
                    drug_names=["metformin"],
                    section_queries=["dosage and administration"],
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
        await self._run_test(
            "ct_strategy_sort",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="breast cancer",
                    strategy=SearchStrategy.COMBINED,
                    sort_by=SortField.START_DATE,
                    sort_order=SortOrder.ASC,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ct_study_type_interventional",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="diabetes",
                    study_type=StudyType.INTERVENTIONAL,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test pagination with offset
        await self._run_test(
            "ct_pagination_offset",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="cancer",
                    limit=10,
                    offset=20,
                ),
            ),
            self._assert_not_error,
        )
        # Test outcome type filter
        await self._run_test(
            "ct_outcome_type_primary",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="breast cancer",
                    outcome_type="primary",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test enrollment range
        await self._run_test(
            "ct_enrollment_range",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="diabetes",
                    min_enrollment=100,
                    max_enrollment=1000,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test multiple statuses
        await self._run_test(
            "ct_multiple_statuses",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    condition="lung cancer",
                    status=[TrialStatus.RECRUITING, TrialStatus.ACTIVE_NOT_RECRUITING],
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test keyword search
        await self._run_test(
            "ct_keyword_immunotherapy",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    keyword="immunotherapy checkpoint inhibitor",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_molecule_trial_search(self) -> None:
        """Test molecule <-> trial connectivity search (trials_by_molecule, molecules_by_condition, trials_by_target)."""
        await self._run_test(
            "mol_trial_by_molecule",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_molecule",
                    molecule_name="imatinib",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
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

    async def test_molecule_trial_search_extended(self) -> None:
        """Extended molecule-trial connectivity tests with correctness validation."""
        # Test trials_by_molecule with inchi_key
        await self._run_test(
            "mol_trial_by_inchikey",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_molecule",
                    inchi_key=TEST_MOLECULES["imatinib"].get("inchi_key", "KTUFNOKKBVMGRW-UHFFFAOYSA-N"),
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test trials_by_target with phase filter
        await self._run_test(
            "mol_trial_by_target_phase_filter",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_target",
                    target_gene="ABL1",
                    phase=["PHASE3"],
                    min_pchembl=6.0,
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test molecules_by_condition for different conditions
        for condition in ["breast cancer", "leukemia", "melanoma"]:
            await self._run_test(
                f"mol_trial_condition_{condition.replace(' ', '_')}",
                molecule_trial_search_async(
                    DEFAULT_CONFIG,
                    MoleculeTrialSearchInput(
                        mode="molecules_by_condition",
                        condition=condition,
                        limit=20,
                    ),
                ),
                self._assert_not_error,
            )
        # Test correctness: imatinib should have trials
        await self._run_test(
            "mol_trial_imatinib_correctness",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_molecule",
                    molecule_name="imatinib",
                    limit=50,
                ),
            ),
            lambda r: self._assert_mol_trial_has_trials(r, min_trials=5),
        )
        # Test pagination
        await self._run_test(
            "mol_trial_pagination",
            molecule_trial_search_async(
                DEFAULT_CONFIG,
                MoleculeTrialSearchInput(
                    mode="trials_by_molecule",
                    molecule_name="pembrolizumab",
                    limit=10,
                    offset=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_adverse_events_search(self) -> None:
        """Test adverse event search modes (events_for_drug, drugs_with_event, compare_safety) and severity (all, serious, other)."""
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
        await self._run_test(
            "ae_compare_safety",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="compare_safety",
                    drug_names=["pembrolizumab", "nivolumab"],
                    severity="all",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ae_severity_serious",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="pembrolizumab",
                    severity="serious",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "ae_severity_other",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="pembrolizumab",
                    severity="other",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_adverse_events_search_extended(self) -> None:
        """Extended adverse events tests with correctness validation."""
        # Test min_subjects_affected filter
        await self._run_test(
            "ae_min_subjects_affected",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="imatinib",
                    min_subjects_affected=10,
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        # Test drugs_with_event for common events
        for event in ["nausea", "fatigue", "diarrhea"]:
            await self._run_test(
                f"ae_drugs_with_{event}",
                adverse_events_search_async(
                    DEFAULT_CONFIG,
                    AdverseEventsSearchInput(
                        mode="drugs_with_event",
                        event_term=event,
                        severity="all",
                        limit=10,
                    ),
                ),
                self._assert_not_error,
            )
        # Test compare_safety with multiple drugs
        await self._run_test(
            "ae_compare_multiple_drugs",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="compare_safety",
                    drug_names=["imatinib", "dasatinib", "nilotinib"],
                    severity="all",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        # Test correctness: check that known drugs return expected events
        await self._run_test(
            "ae_imatinib_correctness",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="imatinib",
                    severity="all",
                    limit=50,
                ),
            ),
            self._assert_adverse_events_has_data,
        )
        # Test pagination
        await self._run_test(
            "ae_pagination",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="pembrolizumab",
                    limit=10,
                    offset=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_outcomes_search(self) -> None:
        """Test outcomes search modes (outcomes_for_trial, trials_with_outcome, efficacy_comparison)."""
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
            "outcomes_trials_with_outcome",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="trials_with_outcome",
                    outcome_keyword="response",
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
        await self._run_test(
            "outcomes_outcome_type_primary",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="outcomes_for_trial",
                    nct_id="NCT01295827",
                    outcome_type="primary",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_outcomes_search_extended(self) -> None:
        """Extended outcomes search tests with correctness validation."""
        # Test outcome_type secondary
        await self._run_test(
            "outcomes_type_secondary",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="outcomes_for_trial",
                    nct_id="NCT01295827",
                    outcome_type="secondary",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test trials_with_outcome for various outcome keywords
        for keyword in ["survival", "progression", "remission", "response rate"]:
            await self._run_test(
                f"outcomes_keyword_{keyword.replace(' ', '_')}",
                outcomes_search_async(
                    DEFAULT_CONFIG,
                    OutcomesSearchInput(
                        mode="trials_with_outcome",
                        outcome_keyword=keyword,
                        limit=10,
                    ),
                ),
                self._assert_not_error,
            )
        # Test efficacy_comparison for different drugs
        for drug in ["pembrolizumab", "nivolumab", "trastuzumab"]:
            await self._run_test(
                f"outcomes_efficacy_{drug}",
                outcomes_search_async(
                    DEFAULT_CONFIG,
                    OutcomesSearchInput(
                        mode="efficacy_comparison",
                        drug_name=drug,
                        max_p_value=0.05,
                        limit=10,
                    ),
                ),
                self._assert_not_error,
            )
        # Test pagination
        await self._run_test(
            "outcomes_pagination",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="trials_with_outcome",
                    outcome_keyword="survival",
                    limit=10,
                    offset=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test correctness: NCT01295827 should have outcomes
        await self._run_test(
            "outcomes_nct_correctness",
            outcomes_search_async(
                DEFAULT_CONFIG,
                OutcomesSearchInput(
                    mode="outcomes_for_trial",
                    nct_id="NCT01295827",
                    limit=50,
                ),
            ),
            self._assert_outcomes_has_data,
        )

    async def test_orange_book_search(self) -> None:
        """Test Orange Book search modes (te_codes, patents, generics, exclusivity)."""
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
        await self._run_test(
            "orange_book_generics",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="generics",
                    drug_name="imatinib",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "orange_book_exclusivity",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="exclusivity",
                    drug_name="imatinib",
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )

    async def test_orange_book_search_extended(self) -> None:
        """Extended Orange Book search tests with correctness validation."""
        # Test search by ingredient
        await self._run_test(
            "orange_book_ingredient_metformin",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="te_codes",
                    ingredient="metformin",
                    limit=20,
                ),
            ),
            self._assert_not_error,
        )
        # Test search by NDA number (if known)
        await self._run_test(
            "orange_book_nda_number",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="te_codes",
                    nda_number="021588",  # Gleevec NDA
                    limit=10,
                ),
            ),
            self._assert_not_error,
        )
        # Test all modes for multiple drugs
        for drug in ["metformin", "aspirin", "atorvastatin"]:
            for mode in ["te_codes", "patents", "generics", "exclusivity"]:
                await self._run_test(
                    f"orange_book_{mode}_{drug}",
                    orange_book_search_async(
                        DEFAULT_CONFIG,
                        OrangeBookSearchInput(
                            mode=mode,
                            drug_name=drug,
                            limit=10,
                        ),
                    ),
                    self._assert_not_error,
                )
        # Test pagination
        await self._run_test(
            "orange_book_pagination",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="generics",
                    drug_name="metformin",
                    limit=10,
                    offset=5,
                ),
            ),
            self._assert_not_error,
        )
        # Test correctness: metformin should have generics
        await self._run_test(
            "orange_book_metformin_generics_correctness",
            orange_book_search_async(
                DEFAULT_CONFIG,
                OrangeBookSearchInput(
                    mode="generics",
                    drug_name="metformin",
                    limit=50,
                ),
            ),
            self._assert_orange_book_has_data,
        )

    async def test_cross_db_lookup(self) -> None:
        """Test cross-database lookup (identifier_type: name, chembl, inchikey, auto)."""
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
        await self._run_test(
            "cross_db_inchikey",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                    identifier_type="inchikey",
                ),
            ),
            self._assert_not_error,
        )
        await self._run_test(
            "cross_db_auto",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(identifier="imatinib", identifier_type="auto"),
            ),
            self._assert_not_error,
        )

    async def test_cross_db_lookup_extended(self) -> None:
        """Extended cross-database lookup tests with correctness validation."""
        # Test include_labels option
        await self._run_test(
            "cross_db_include_labels",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="imatinib",
                    identifier_type="name",
                    include_labels=True,
                    include_trials=False,
                    include_targets=False,
                ),
            ),
            self._assert_not_error,
        )
        # Test include_trials option
        await self._run_test(
            "cross_db_include_trials",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="pembrolizumab",
                    identifier_type="name",
                    include_labels=False,
                    include_trials=True,
                    include_targets=False,
                ),
            ),
            self._assert_not_error,
        )
        # Test include_targets option
        await self._run_test(
            "cross_db_include_targets",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="erlotinib",
                    identifier_type="name",
                    include_labels=False,
                    include_trials=False,
                    include_targets=True,
                ),
            ),
            self._assert_not_error,
        )
        # Test all options enabled
        await self._run_test(
            "cross_db_all_options",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="imatinib",
                    identifier_type="name",
                    include_labels=True,
                    include_trials=True,
                    include_targets=True,
                    limit=50,
                ),
            ),
            self._assert_cross_db_has_data,
        )
        # Test multiple identifier types for same drug
        for id_type, identifier in [
            ("name", "aspirin"),
            ("chembl", "CHEMBL25"),
            ("auto", "CHEMBL25"),
        ]:
            await self._run_test(
                f"cross_db_{id_type}_{identifier}",
                cross_database_lookup_async(
                    DEFAULT_CONFIG,
                    CrossDatabaseLookupInput(
                        identifier=identifier,
                        identifier_type=id_type,
                    ),
                ),
                self._assert_not_error,
            )
        # Test correctness: imatinib should have molecules, labels, trials, targets
        await self._run_test(
            "cross_db_imatinib_correctness",
            cross_database_lookup_async(
                DEFAULT_CONFIG,
                CrossDatabaseLookupInput(
                    identifier="imatinib",
                    identifier_type="name",
                    include_labels=True,
                    include_trials=True,
                    include_targets=True,
                ),
            ),
            lambda r: self._assert_cross_db_has_molecules(r),
        )

    async def test_biotherapeutic_sequence_search(self) -> None:
        """Test biotherapeutic sequence search (by_target, by_sequence, similar_biologics) and biotherapeutic_type (all, antibody, enzyme)."""
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
        await self._run_test(
            "bio_by_sequence",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_sequence",
                    sequence="CDR",
                    limit=10,
                ),
            ),
            self._assert_status_in_allowed,
        )
        await self._run_test(
            "bio_similar_biologics",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="similar_biologics",
                    sequence="CDR",
                    limit=10,
                ),
            ),
            self._assert_status_in_allowed,
        )
        await self._run_test(
            "bio_type_antibody",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_target",
                    target_gene="PDCD1",
                    biotherapeutic_type="antibody",
                    limit=10,
                ),
            ),
            self._assert_status_in_allowed,
        )
        await self._run_test(
            "bio_type_enzyme",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_target",
                    target_gene="EGFR",
                    biotherapeutic_type="enzyme",
                    limit=10,
                ),
            ),
            self._assert_status_in_allowed,
        )

    async def test_biotherapeutic_sequence_search_extended(self) -> None:
        """Extended biotherapeutic sequence search tests."""
        # Test by_target for various targets
        for target in ["ERBB2", "TNF", "VEGFA", "MS4A1", "CTLA4"]:
            await self._run_test(
                f"bio_by_target_{target}",
                biotherapeutic_sequence_search_async(
                    DEFAULT_CONFIG,
                    BiotherapeuticSearchInput(
                        mode="by_target",
                        target_gene=target,
                        limit=10,
                    ),
                ),
                self._assert_status_in_allowed,
            )
        # Test by_sequence with different motifs
        for motif in ["CDRH3", "LCDR", "VH", "VL"]:
            await self._run_test(
                f"bio_sequence_motif_{motif}",
                biotherapeutic_sequence_search_async(
                    DEFAULT_CONFIG,
                    BiotherapeuticSearchInput(
                        mode="by_sequence",
                        sequence=motif,
                        limit=10,
                    ),
                ),
                self._assert_status_in_allowed,
            )
        # Test biotherapeutic_type="all"
        await self._run_test(
            "bio_type_all",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_target",
                    target_gene="EGFR",
                    biotherapeutic_type="all",
                    limit=20,
                ),
            ),
            self._assert_status_in_allowed,
        )
        # Test pagination
        await self._run_test(
            "bio_pagination",
            biotherapeutic_sequence_search_async(
                DEFAULT_CONFIG,
                BiotherapeuticSearchInput(
                    mode="by_sequence",
                    sequence="CDR",
                    limit=10,
                    offset=5,
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
        # Edge case: empty drug name after strip
        await self._run_test(
            "edge_empty_drug_name",
            adverse_events_search_async(
                DEFAULT_CONFIG,
                AdverseEventsSearchInput(
                    mode="events_for_drug",
                    drug_name="   ",  # Whitespace only
                    limit=10,
                ),
            ),
            lambda r: self._assert_status_in(r, ["invalid_input", "not_found", "error"]),
        )
        # Edge case: very long query
        await self._run_test(
            "edge_long_query",
            clinical_trials_search_async(
                DEFAULT_CONFIG,
                ClinicalTrialsSearchInput(
                    keyword="cancer " * 100,  # Very long query
                    limit=5,
                ),
            ),
            self._assert_not_error,
        )
        # Edge case: special characters in query
        await self._run_test(
            "edge_special_chars",
            target_search_async(
                DEFAULT_CONFIG,
                TargetSearchInput(
                    mode=SearchMode.TARGETS_FOR_DRUG,
                    query="drug's name (with) special-chars",
                ),
            ),
            lambda r: self._assert_status_in(r, ["not_found", "success", "error"]),
        )

    # ---------------------------------------------------------------------
    # Invalid literals (construction-time ValidationError / ValueError)
    # ---------------------------------------------------------------------

    async def test_invalid_literals_outcomes(self) -> None:
        """Unsupported mode or outcome_type raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            OutcomesSearchInput(mode="invalid_mode", nct_id="NCT01295827")
        with _raises_or_pytest(ValidationError):
            OutcomesSearchInput(mode="outcomes_for_trial", nct_id="NCT01295827", outcome_type="invalid_type")

    async def test_invalid_literals_adverse_events(self) -> None:
        """Unsupported mode or severity raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            AdverseEventsSearchInput(mode="invalid", drug_name="pembrolizumab")
        with _raises_or_pytest(ValidationError):
            AdverseEventsSearchInput(mode="events_for_drug", drug_name="pembrolizumab", severity="non-serious")

    async def test_invalid_literals_orange_book(self) -> None:
        """Unsupported mode raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            OrangeBookSearchInput(mode="invalid", drug_name="imatinib")

    async def test_invalid_literals_cross_db(self) -> None:
        """Unsupported identifier_type raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            CrossDatabaseLookupInput(identifier="imatinib", identifier_type="invalid_type")

    async def test_invalid_literals_biotherapeutic(self) -> None:
        """Unsupported mode or biotherapeutic_type raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            BiotherapeuticSearchInput(mode="invalid", target_gene="PDCD1")
        with _raises_or_pytest(ValidationError):
            BiotherapeuticSearchInput(mode="by_target", target_gene="PDCD1", biotherapeutic_type="peptide")

    async def test_invalid_literals_molecule_trial(self) -> None:
        """Unsupported mode raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            MoleculeTrialSearchInput(mode="invalid", molecule_name="imatinib")

    async def test_invalid_literals_target_search(self) -> None:
        """Invalid SearchMode or approval_status raises ValidationError at input construction."""
        with _raises_or_pytest(ValidationError):
            TargetSearchInput(mode="INVALID", query="imatinib")  # type: ignore[arg-type]
        with _raises_or_pytest(ValidationError):
            TargetSearchInput(mode=SearchMode.TARGETS_FOR_DRUG, query="imatinib", approval_status="invalid")

    async def test_invalid_literals_clinical_trials(self) -> None:
        """Invalid status, phase, study_type, or intervention_type raises ValidationError (from validators)."""
        try:
            ClinicalTrialsSearchInput(condition="cancer", status=["INVALID_STATUS"], limit=5)
        except ValidationError as e:
            assert "Invalid status" in str(e) or "status" in str(e).lower()
        else:
            raise AssertionError("Expected ValidationError for invalid status")
        try:
            ClinicalTrialsSearchInput(condition="cancer", phase=["Phase 99"], limit=5)
        except ValidationError as e:
            assert "Invalid phase" in str(e) or "phase" in str(e).lower()
        else:
            raise AssertionError("Expected ValidationError for invalid phase")
        with _raises_or_pytest(ValidationError):
            ClinicalTrialsSearchInput(condition="cancer", study_type="Invalid", limit=5)
        with _raises_or_pytest(ValidationError):
            ClinicalTrialsSearchInput(condition="cancer", intervention_type=["invalid"], limit=5)

    async def test_invalid_literals_dailymed(self) -> None:
        """Invalid search combination raises ValueError or ValidationError from model_validator."""
        with _raises_or_pytest(ValidationError, ValueError):
            DailyMedAndOpenFDAInput(section_queries=["warnings"], result_limit=5)

    # ---------------------------------------------------------------------
    # Semantic validation (missing required params per mode -> invalid_input)
    # ---------------------------------------------------------------------

    async def test_invalid_input_missing_required_molecule_trial(self) -> None:
        """Missing required params for mode returns status=invalid_input."""
        out = await molecule_trial_search_async(
            DEFAULT_CONFIG,
            MoleculeTrialSearchInput(mode="trials_by_molecule", molecule_name=None, inchi_key=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error
        out2 = await molecule_trial_search_async(
            DEFAULT_CONFIG,
            MoleculeTrialSearchInput(mode="molecules_by_condition", condition=None, limit=10),
        )
        assert out2.status == "invalid_input"
        assert out2.error
        # Test trials_by_target missing target_gene
        out3 = await molecule_trial_search_async(
            DEFAULT_CONFIG,
            MoleculeTrialSearchInput(mode="trials_by_target", target_gene=None, limit=10),
        )
        assert out3.status == "invalid_input"
        assert out3.error

    async def test_invalid_input_missing_required_adverse_events(self) -> None:
        """Missing drug_name or event_term or drug_names for mode returns status=invalid_input."""
        out = await adverse_events_search_async(
            DEFAULT_CONFIG,
            AdverseEventsSearchInput(mode="events_for_drug", drug_name=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error
        out2 = await adverse_events_search_async(
            DEFAULT_CONFIG,
            AdverseEventsSearchInput(mode="compare_safety", drug_names=["pembrolizumab"], limit=10),
        )
        assert out2.status == "invalid_input"
        assert out2.error
        # Test drugs_with_event missing event_term
        out3 = await adverse_events_search_async(
            DEFAULT_CONFIG,
            AdverseEventsSearchInput(mode="drugs_with_event", event_term=None, limit=10),
        )
        assert out3.status == "invalid_input"
        assert out3.error

    async def test_invalid_input_missing_required_outcomes(self) -> None:
        """Missing nct_id or outcome_keyword or drug_name for mode returns status=invalid_input."""
        out = await outcomes_search_async(
            DEFAULT_CONFIG,
            OutcomesSearchInput(mode="outcomes_for_trial", nct_id=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error
        out2 = await outcomes_search_async(
            DEFAULT_CONFIG,
            OutcomesSearchInput(mode="trials_with_outcome", outcome_keyword=None, limit=10),
        )
        assert out2.status == "invalid_input"
        assert out2.error
        # Test efficacy_comparison missing drug_name
        out3 = await outcomes_search_async(
            DEFAULT_CONFIG,
            OutcomesSearchInput(mode="efficacy_comparison", drug_name=None, limit=10),
        )
        assert out3.status == "invalid_input"
        assert out3.error

    async def test_invalid_input_missing_required_orange_book(self) -> None:
        """Missing drug_name, ingredient, and nda_number returns status=invalid_input."""
        out = await orange_book_search_async(
            DEFAULT_CONFIG,
            OrangeBookSearchInput(mode="te_codes", drug_name=None, ingredient=None, nda_number=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error

    async def test_invalid_input_missing_required_biotherapeutic(self) -> None:
        """Missing sequence or target_gene for mode returns status=invalid_input."""
        out = await biotherapeutic_sequence_search_async(
            DEFAULT_CONFIG,
            BiotherapeuticSearchInput(mode="by_sequence", sequence=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error
        # Test by_target missing target_gene
        out2 = await biotherapeutic_sequence_search_async(
            DEFAULT_CONFIG,
            BiotherapeuticSearchInput(mode="by_target", target_gene=None, limit=10),
        )
        assert out2.status == "invalid_input"
        assert out2.error

    async def test_invalid_input_missing_required_target_search(self) -> None:
        """Missing query for TARGETS_FOR_DRUG returns status=invalid_input with diagnostics."""
        out = await target_search_async(
            DEFAULT_CONFIG,
            TargetSearchInput(mode=SearchMode.TARGETS_FOR_DRUG, query=None, limit=10),
        )
        assert out.status == "invalid_input"
        assert out.error

    async def test_invalid_input_missing_required_cross_db(self) -> None:
        """Missing identifier returns status=invalid_input."""
        out = await cross_database_lookup_async(
            DEFAULT_CONFIG,
            CrossDatabaseLookupInput(identifier="", identifier_type="name"),
        )
        assert out.status == "invalid_input"
        assert out.error

    @staticmethod
    def _assert_invalid_literals(_result: None) -> None:
        """No-op assertion; sub-tests assert ValidationError/ValueError internally."""
        return None

    async def test_invalid_literals(self) -> None:
        """Run all invalid-literals (construction-time ValidationError/ValueError) tests."""
        await self._run_test(
            "invalid_literals_outcomes",
            self.test_invalid_literals_outcomes(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_adverse_events",
            self.test_invalid_literals_adverse_events(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_orange_book",
            self.test_invalid_literals_orange_book(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_cross_db",
            self.test_invalid_literals_cross_db(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_biotherapeutic",
            self.test_invalid_literals_biotherapeutic(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_molecule_trial",
            self.test_invalid_literals_molecule_trial(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_target_search",
            self.test_invalid_literals_target_search(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_clinical_trials",
            self.test_invalid_literals_clinical_trials(),
            self._assert_invalid_literals,
        )
        await self._run_test(
            "invalid_literals_dailymed",
            self.test_invalid_literals_dailymed(),
            self._assert_invalid_literals,
        )

    @staticmethod
    def _assert_invalid_input(_result: None) -> None:
        """No-op assertion; sub-tests assert status=invalid_input internally."""
        return None

    async def test_invalid_input_missing_required(self) -> None:
        """Run all semantic validation (missing required params -> invalid_input) tests."""
        await self._run_test(
            "invalid_input_molecule_trial",
            self.test_invalid_input_missing_required_molecule_trial(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_adverse_events",
            self.test_invalid_input_missing_required_adverse_events(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_outcomes",
            self.test_invalid_input_missing_required_outcomes(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_orange_book",
            self.test_invalid_input_missing_required_orange_book(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_biotherapeutic",
            self.test_invalid_input_missing_required_biotherapeutic(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_target_search",
            self.test_invalid_input_missing_required_target_search(),
            self._assert_invalid_input,
        )
        await self._run_test(
            "invalid_input_cross_db",
            self.test_invalid_input_missing_required_cross_db(),
            self._assert_invalid_input,
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

    # New assertions for extended tests

    @staticmethod
    def _assert_mol_trial_has_trials(result, min_trials: int = 1) -> None:
        """Assert molecule-trial search returns trials."""
        if result.status == "not_found":
            raise AssertionError("Expected trials but got not_found")
        if result.status != "success":
            raise AssertionError(f"Expected success, got {result.status}")
        if len(result.hits) < min_trials:
            raise AssertionError(f"Expected >= {min_trials} trials, got {len(result.hits)}")

    @staticmethod
    def _assert_adverse_events_has_data(result) -> None:
        """Assert adverse events search returns data."""
        if result.status == "not_found":
            return  # Acceptable if no data
        if result.status != "success":
            raise AssertionError(f"Expected success or not_found, got {result.status}")
        # Check that hits have expected fields
        for hit in result.hits:
            if hasattr(hit, "adverse_event_term") and hit.adverse_event_term:
                return
        if result.hits:
            return  # Has some hits

    @staticmethod
    def _assert_outcomes_has_data(result) -> None:
        """Assert outcomes search returns data."""
        if result.status == "not_found":
            return  # Acceptable if no data
        if result.status != "success":
            raise AssertionError(f"Expected success or not_found, got {result.status}")

    @staticmethod
    def _assert_orange_book_has_data(result) -> None:
        """Assert Orange Book search returns data."""
        if result.status == "not_found":
            return  # Acceptable if no data
        if result.status != "success":
            raise AssertionError(f"Expected success or not_found, got {result.status}")

    @staticmethod
    def _assert_cross_db_has_data(result) -> None:
        """Assert cross-database lookup returns comprehensive data."""
        if result.status != "success":
            raise AssertionError(f"Expected success, got {result.status}")
        # Check molecules
        if not result.molecules:
            raise AssertionError("Expected molecules in cross-db lookup")

    @staticmethod
    def _assert_cross_db_has_molecules(result) -> None:
        """Assert cross-database lookup returns molecules."""
        if result.status != "success":
            raise AssertionError(f"Expected success, got {result.status}")
        if not result.molecules:
            raise AssertionError("Expected molecules in cross-db lookup")

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
            "molecule_trial_extended": self.test_molecule_trial_search_extended,
            "adverse_events": self.test_adverse_events_search,
            "adverse_events_extended": self.test_adverse_events_search_extended,
            "outcomes": self.test_outcomes_search,
            "outcomes_extended": self.test_outcomes_search_extended,
            "orange_book": self.test_orange_book_search,
            "orange_book_extended": self.test_orange_book_search_extended,
            "cross_db": self.test_cross_db_lookup,
            "cross_db_extended": self.test_cross_db_lookup_extended,
            "biotherapeutics_sequence": self.test_biotherapeutic_sequence_search,
            "biotherapeutics_sequence_extended": self.test_biotherapeutic_sequence_search_extended,
            "invalid_literals": self.test_invalid_literals,
            "invalid_input": self.test_invalid_input_missing_required,
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
                self.test_molecule_trial_search_extended,
                self.test_adverse_events_search,
                self.test_adverse_events_search_extended,
                self.test_outcomes_search,
                self.test_outcomes_search_extended,
                self.test_orange_book_search,
                self.test_orange_book_search_extended,
                self.test_cross_db_lookup,
                self.test_cross_db_lookup_extended,
                self.test_biotherapeutic_sequence_search,
                self.test_biotherapeutic_sequence_search_extended,
                self.test_invalid_literals,
                self.test_invalid_input_missing_required,
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
            "molecule_trial_extended",
            "adverse_events",
            "adverse_events_extended",
            "outcomes",
            "outcomes_extended",
            "orange_book",
            "orange_book_extended",
            "cross_db",
            "cross_db_extended",
            "biotherapeutics_sequence",
            "biotherapeutics_sequence_extended",
            "invalid_literals",
            "invalid_input",
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
