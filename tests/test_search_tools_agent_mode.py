#!/usr/bin/env python3
"""
Test script for biomedical agent tools.

Runs each tool with natural-language queries and validates:
- the tool was invoked
- response contains expected keywords
- pass/fail summaries are reported
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import dotenv
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from bioagent.agent import get_chat_model
from bioagent.agent.tools import think
from bioagent.agent.tools.dbsearch import DBSEARCH_TOOLS
from bioagent.agent.tools.target_search import TARGET_SEARCH_TOOLS, pharmacology_search


# Load environment variables
dotenv.load_dotenv()


@dataclass
class ToolTestCase:
    query: str
    expected_any: list[str] | None = None
    expected_all: list[str] | None = None
    expect_tool: bool = True


@dataclass
class AgentTestResult:
    tool_name: str
    query: str
    success: bool
    tool_was_called: bool
    expected_content_found: bool
    duration: float
    response_preview: str
    tool_calls: list[str]
    error: str | None = None


def _normalize_text(text: str | None) -> str:
    return (text or "").lower()


def _check_expected(text: str, expected_any: list[str] | None, expected_all: list[str] | None) -> bool:
    if not expected_any and not expected_all:
        return True
    normalized = _normalize_text(text)
    if expected_any:
        if any(token.lower() in normalized for token in expected_any):
            return True
    if expected_all:
        return all(token.lower() in normalized for token in expected_all)
    return False


def _extract_tool_calls(messages: list[Any]) -> list[str]:
    tool_calls: list[str] = []
    for msg in messages:
        # ToolMessage (langchain)
        if getattr(msg, "type", None) == "tool":
            name = getattr(msg, "name", None)
            if name:
                tool_calls.append(name)

        # AIMessage tool calls
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                if isinstance(call, dict):
                    name = call.get("name") or call.get("function", {}).get("name")
                    if name:
                        tool_calls.append(name)
                else:
                    name = getattr(call, "name", None)
                    if name:
                        tool_calls.append(name)

        # Additional kwargs tool calls
        extra = getattr(msg, "additional_kwargs", {}) or {}
        for call in extra.get("tool_calls", []) or []:
            name = call.get("function", {}).get("name") or call.get("name")
            if name:
                tool_calls.append(name)

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for name in tool_calls:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _extract_last_response(messages: list[Any]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content:
            return str(content)
    return ""


def _get_tool_function(tool_name: str):
    for tool in DBSEARCH_TOOLS:
        if tool.name == tool_name:
            return tool
    for tool in TARGET_SEARCH_TOOLS:
        if tool.name == tool_name:
            return tool
    if tool_name == "pharmacology_search":
        return pharmacology_search
    return None


class AgentToolTester:
    """Agent-mode test runner with tool invocation and content checks."""

    def __init__(self, model):
        self.model = model

    async def run_tool_cases(
        self,
        tool_name: str,
        tool_function,
        cases: list[ToolTestCase],
        fail_fast: bool = False,
    ) -> list[AgentTestResult]:
        results: list[AgentTestResult] = []

        checkpointer = InMemorySaver()
        system_prompt = (
            f"You are a biomedical agent testing the {tool_name} tool.\n"
            f"Use the {tool_name} tool to answer user questions directly and concisely.\n\n"
            "When tool outputs exceed ~4000 characters, they are automatically summarized.\n"
            "The full output is stored in files and can be retrieved:\n"
            "- Use list_research_outputs() to browse stored outputs with one-line descriptions.\n"
            "- Use retrieve_full_output(ref_id) to fetch full content when needed.\n"
            "Only retrieve full output if the summary indicates critical details are missing."
        )

        agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=[think, tool_function],
            checkpointer=checkpointer,
            debug=False,
        )

        for case in cases:
            start_time = datetime.now()
            try:
                human_msg = HumanMessage(case.query)
                response = await agent.ainvoke(
                    {"messages": [human_msg]},
                    {"configurable": {"thread_id": f"test_{tool_name}_{start_time.isoformat()}"}},
                )
                duration = (datetime.now() - start_time).total_seconds()

                messages = response.get("messages", []) if isinstance(response, dict) else []
                tool_calls = _extract_tool_calls(messages)
                response_text = _extract_last_response(messages)

                tool_was_called = tool_name in tool_calls
                expected_found = _check_expected(response_text, case.expected_any, case.expected_all)
                tool_ok = tool_was_called if case.expect_tool else not tool_was_called
                success = bool(response_text) and tool_ok and expected_found

                preview = response_text[:400] + ("..." if len(response_text) > 400 else "")
                results.append(
                    AgentTestResult(
                        tool_name=tool_name,
                        query=case.query,
                        success=success,
                        tool_was_called=tool_was_called,
                        expected_content_found=expected_found,
                        duration=duration,
                        response_preview=preview,
                        tool_calls=tool_calls,
                    )
                )

            except Exception as exc:
                duration = (datetime.now() - start_time).total_seconds()
                results.append(
                    AgentTestResult(
                        tool_name=tool_name,
                        query=case.query,
                        success=False,
                        tool_was_called=False,
                        expected_content_found=False,
                        duration=duration,
                        response_preview="",
                        tool_calls=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

            if fail_fast and results[-1].success is False:
                break

        return results


TOOL_TEST_CASES: dict[str, dict[str, Any]] = {
    "search_clinical_trials": {
        "description": "Search ClinicalTrials.gov for clinical studies",
        "cases": [
            ToolTestCase(
                query="Find clinical trials for ruxolitinib in cancer patients that are still recruiting",
                expected_any=["ruxolitinib"],
            ),
            ToolTestCase(
                query="Search for Phase 3 trials in breast cancer with immunotherapy",
                expected_any=["breast", "phase 3"],
            ),
            ToolTestCase(
                query="Find completed trials for metformin in diabetes",
                expected_any=["metformin", "diabetes"],
            ),
            ToolTestCase(
                query="Find Phase 3 breast cancer trials, give me just a brief overview",
                expected_any=["breast", "phase 3"],
            ),
        ],
    },
    "get_clinical_trial_details": {
        "description": "Get detailed info for specific NCT IDs",
        "cases": [
            ToolTestCase(
                query="Get full details for trial NCT04280705",
                expected_any=["NCT04280705"],
            ),
            ToolTestCase(
                query="Show me details for NCT01295827 and NCT03456789",
                expected_any=["NCT01295827"],
            ),
        ],
    },
    "search_drug_labels": {
        "description": "Search FDA drug labels from OpenFDA and DailyMed",
        "cases": [
            ToolTestCase(
                query="What are the warnings and side effects for aspirin?",
                expected_any=["aspirin", "warnings"],
            ),
            ToolTestCase(
                query="Find contraindications for metformin",
                expected_any=["metformin", "contraindications"],
            ),
            ToolTestCase(
                query="What are the drug interactions for warfarin?",
                expected_any=["warfarin", "interactions"],
            ),
        ],
    },
    "check_data_availability": {
        "description": "Check what data is available for an entity",
        "cases": [
            ToolTestCase(
                query="What data do we have about imatinib?",
                expected_any=["imatinib", "clinical trials"],
            ),
            ToolTestCase(
                query="Check data availability for EGFR",
                expected_any=["EGFR", "target"],
            ),
        ],
    },
    "search_molecule_trials": {
        "description": "Link molecules to clinical trials via name, target, or structure",
        "cases": [
            # Mode: trials_by_molecule
            ToolTestCase(
                query="Find clinical trials linked to the drug imatinib",
                expected_any=["imatinib", "nct"],
            ),
            # Mode: molecules_by_condition
            ToolTestCase(
                query="Find molecules being studied for breast cancer",
                expected_any=["breast"],
            ),
            # Mode: trials_by_target - KEY TEST: Target/Gene → Clinical Trials
            ToolTestCase(
                query="Find clinical trials for drugs that target EGFR. Use the trials_by_target mode.",
                expected_any=["egfr", "nct"],
            ),
            ToolTestCase(
                query="What clinical trials exist for drugs targeting JAK2? I want to go from target to trials.",
                expected_any=["jak2", "nct"],
            ),
            ToolTestCase(
                query="Find trials for drugs that modulate the ABL1 target gene",
                expected_any=["abl1", "nct"],
            ),
            # Mode: trials_by_structure - Structure similarity → Trials
            ToolTestCase(
                query="Find clinical trials for molecules similar to this SMILES: CC(=O)Oc1ccccc1C(=O)O. Use trials_by_structure mode.",
                expected_any=["aspirin", "nct", "trial"],
            ),
            # Mode: trials_by_substructure - Substructure → Trials
            ToolTestCase(
                query="Find clinical trials for molecules containing a benzene ring substructure. Use the smiles c1ccccc1.",
                expected_any=["nct", "trial"],
            ),
        ],
    },
    "search_adverse_events": {
        "description": "Search trial adverse event summaries",
        "cases": [
            ToolTestCase(
                query="Show top adverse events for imatinib",
                expected_any=["imatinib"],
            ),
            ToolTestCase(
                query="Find trials reporting neutropenia",
                expected_any=["neutropenia"],
            ),
            ToolTestCase(
                query="Compare safety of imatinib and dasatinib",
                expected_any=["imatinib", "dasatinib"],
            ),
        ],
    },
    "search_trial_outcomes": {
        "description": "Search clinical trial outcomes and analyses",
        "cases": [
            ToolTestCase(
                query="Outcomes for NCT01295827",
                expected_any=["NCT01295827"],
            ),
            ToolTestCase(
                query="Trials with outcome overall survival",
                expected_any=["overall survival"],
            ),
            ToolTestCase(
                query="Efficacy comparison for ruxolitinib",
                expected_any=["ruxolitinib"],
            ),
        ],
    },
    "search_orange_book": {
        "description": "Search FDA Orange Book records",
        "cases": [
            ToolTestCase(
                query="Find TE codes for metformin",
                expected_any=["metformin"],
            ),
            ToolTestCase(
                query="Patents for imatinib",
                expected_any=["imatinib"],
            ),
            ToolTestCase(
                query="Exclusivity for warfarin",
                expected_any=["warfarin"],
            ),
        ],
    },
    "lookup_drug_identifiers": {
        "description": "Resolve drug identifiers across databases",
        "cases": [
            ToolTestCase(
                query="Lookup CHEMBL941",
                expected_any=["CHEMBL941"],
            ),
            ToolTestCase(
                query="Lookup aspirin identifiers",
                expected_any=["aspirin"],
            ),
            ToolTestCase(
                query="Lookup InChIKey BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                expected_any=["BSYNRYMUTXBXSQ"],
            ),
        ],
    },
    "search_biotherapeutics": {
        "description": "Search biologics by sequence motif",
        "cases": [
            ToolTestCase(
                query="Find antibody sequence motif QVQLV",
                expected_any=["QVQLV"],
            ),
            ToolTestCase(
                query="Find similar biologics for sequence EVQLVESGG",
                expected_any=["EVQLVESGG"],
            ),
        ],
    },
    "search_drug_targets": {
        "description": "Find all protein targets for a drug",
        "cases": [
            ToolTestCase(query="What proteins does imatinib target?", expected_any=["imatinib", "ABL1"]),
            ToolTestCase(query="Find all targets for aspirin", expected_any=["aspirin", "PTGS1"]),
            ToolTestCase(query="What are the molecular targets of pembrolizumab?", expected_any=["pembrolizumab", "PDCD1"]),
        ],
    },
    "search_target_drugs": {
        "description": "Find all drugs that modulate a specific protein target",
        "cases": [
            ToolTestCase(query="What drugs inhibit EGFR?", expected_any=["EGFR", "erlotinib"]),
            ToolTestCase(query="Find all compounds that target ABL1", expected_any=["ABL1", "imatinib"]),
            ToolTestCase(query="What medications affect JAK2?", expected_any=["JAK2", "ruxolitinib"]),
            ToolTestCase(query="Find biologics targeting PDCD1", expected_any=["PDCD1"]),
        ],
    },
    "search_similar_molecules": {
        "description": "Find molecules structurally similar to a query compound",
        "cases": [
            ToolTestCase(query="Find molecules similar to aspirin", expected_any=["aspirin"]),
            ToolTestCase(query="What compounds are structurally related to caffeine?", expected_any=["caffeine"]),
            ToolTestCase(query="Find analogs of ibuprofen", expected_any=["ibuprofen"]),
        ],
    },
    "search_exact_structure": {
        "description": "Find an exact structure match for a molecule",
        "cases": [
            ToolTestCase(query="Identify this molecule: CC(=O)Oc1ccccc1C(=O)O", expected_any=["aspirin"]),
            ToolTestCase(query="What is the name of this compound: CN1C=NC2=C1C(=O)N(C(=O)N2C)C", expected_any=["caffeine"]),
            ToolTestCase(query="Find information for SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", expected_any=["ibuprofen"]),
        ],
    },
    "search_substructure": {
        "description": "Find molecules containing a specific substructure",
        "cases": [
            ToolTestCase(query="Find compounds containing a benzene ring", expected_any=["benzene"]),
            ToolTestCase(query="What molecules have an amide group?", expected_any=["amide"]),
            ToolTestCase(query="Find drugs with a pyridine ring", expected_any=["pyridine"]),
        ],
    },
    "get_drug_profile": {
        "description": "Get comprehensive profile for a drug",
        "cases": [
            ToolTestCase(query="Give me a complete profile of imatinib including targets, forms, and trials", expected_any=["imatinib"]),
            ToolTestCase(query="What is the full pharmacological profile of aspirin?", expected_any=["aspirin"]),
            ToolTestCase(query="Comprehensive information about pembrolizumab", expected_any=["pembrolizumab"]),
        ],
    },
    "get_drug_forms": {
        "description": "Get all molecular forms of a drug",
        "cases": [
            ToolTestCase(query="What are all the salt forms of metformin?", expected_any=["metformin"]),
            ToolTestCase(query="Find different formulations of insulin", expected_any=["insulin"]),
            ToolTestCase(query="What are the stereoisomers of thalidomide?", expected_any=["thalidomide"]),
        ],
    },
    "search_drug_trials": {
        "description": "Find clinical trials for a drug",
        "cases": [
            ToolTestCase(query="Find clinical trials for ruxolitinib", expected_any=["ruxolitinib"]),
            ToolTestCase(query="What trials are there for pembrolizumab in cancer?", expected_any=["pembrolizumab"]),
            ToolTestCase(query="Clinical studies involving metformin", expected_any=["metformin"]),
        ],
    },
    "compare_drugs_on_target": {
        "description": "Compare multiple drugs' activity against a single target",
        "cases": [
            ToolTestCase(query="Compare imatinib, dasatinib, and nilotinib on ABL1", expected_any=["ABL1", "imatinib"]),
            ToolTestCase(query="How do gefitinib, erlotinib, and osimertinib compare on EGFR?", expected_any=["EGFR", "gefitinib"]),
            ToolTestCase(query="Compare different JAK inhibitors on JAK2", expected_any=["JAK2"]),
        ],
    },
    "search_selective_drugs": {
        "description": "Find drugs that are selective for one target over others",
        "cases": [
            ToolTestCase(query="Find JAK2-selective inhibitors that spare JAK1 and JAK3", expected_any=["JAK2"]),
            ToolTestCase(query="Find EGFR inhibitors that don't affect ERBB2", expected_any=["EGFR"]),
            ToolTestCase(query="Selective CDK4 inhibitors that spare CDK6", expected_any=["CDK4"]),
        ],
    },
    "search_drug_activities": {
        "description": "Retrieve activity measurements for a drug",
        "cases": [
            ToolTestCase(query="Show activity measurements for imatinib", expected_any=["imatinib"]),
            ToolTestCase(query="Activity data for aspirin", expected_any=["aspirin"]),
            ToolTestCase(query="Activities for pembrolizumab", expected_any=["pembrolizumab"]),
        ],
    },
    "search_target_activities": {
        "description": "Retrieve activity measurements for a target gene",
        "cases": [
            ToolTestCase(query="Activity measurements for EGFR", expected_any=["EGFR"]),
            ToolTestCase(query="Activity measurements for ABL1", expected_any=["ABL1"]),
            ToolTestCase(query="Activity measurements for JAK2", expected_any=["JAK2"]),
        ],
    },
    "search_drug_indications": {
        "description": "Find indications for a drug",
        "cases": [
            ToolTestCase(query="Indications for imatinib", expected_any=["imatinib"]),
            ToolTestCase(query="Indications for aspirin", expected_any=["aspirin"]),
            ToolTestCase(query="Indications for pembrolizumab", expected_any=["pembrolizumab"]),
        ],
    },
    "search_indication_drugs": {
        "description": "Find drugs associated with an indication",
        "cases": [
            ToolTestCase(query="Drugs for chronic myeloid leukemia", expected_any=["leukemia"]),
            ToolTestCase(query="Drugs for rheumatoid arthritis", expected_any=["arthritis"]),
            ToolTestCase(query="Drugs for melanoma", expected_any=["melanoma"]),
        ],
    },
    "search_target_pathways": {
        "description": "Find pathway annotations for a target gene",
        "cases": [
            ToolTestCase(query="Pathways for EGFR", expected_any=["EGFR"]),
            ToolTestCase(query="Pathways for MAPK1", expected_any=["MAPK1"]),
            ToolTestCase(query="Pathways for JAK2", expected_any=["JAK2"]),
        ],
    },
    "search_drug_interactions": {
        "description": "Find known drug-drug interactions",
        "cases": [
            ToolTestCase(query="Drug interactions for warfarin", expected_any=["warfarin"]),
            ToolTestCase(query="Drug interactions for aspirin", expected_any=["aspirin"]),
            ToolTestCase(query="Drug interactions for rifampin", expected_any=["rifampin"]),
        ],
    },
    "pharmacology_search": {
        "description": "Unified pharmacology search tool",
        "cases": [
            ToolTestCase(query="Find all targets for imatinib using unified search", expected_any=["imatinib"]),
            ToolTestCase(query="Search for drugs targeting EGFR with unified tool", expected_any=["EGFR"]),
            ToolTestCase(query="Get comprehensive drug profile for aspirin using unified search", expected_any=["aspirin"]),
            ToolTestCase(query="Find indications for imatinib using unified search", expected_any=["imatinib"]),
            ToolTestCase(query="What pathways involve EGFR? Use pharmacology search.", expected_any=["EGFR"]),
            ToolTestCase(query="Find drug interactions for warfarin using unified search", expected_any=["warfarin"]),
        ],
    },
}


async def main() -> int:
    parser = argparse.ArgumentParser(description="Agent-mode tool tests")
    parser.add_argument("tools", nargs="*", help="Tool names to test (default: all)")
    parser.add_argument("--json", dest="json_path", default=None, help="Write JSON results to file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--model", default="google/gemini-3-flash-preview", help="Model to use for testing")
    args = parser.parse_args()

    try:
        model = get_chat_model(args.model, "openrouter", model_parameters={"temperature": 0.5})
    except Exception as exc:
        print(f"Failed to initialize model: {exc}")
        return 1

    tester = AgentToolTester(model)
    all_tool_names = list(TOOL_TEST_CASES.keys())
    tools_to_test = args.tools or all_tool_names

    results: list[AgentTestResult] = []
    for tool_name in tools_to_test:
        if tool_name not in TOOL_TEST_CASES:
            print(f"Unknown tool '{tool_name}'. Available: {all_tool_names}")
            return 1

        tool_function = _get_tool_function(tool_name)
        if not tool_function:
            print(f"Tool function not found for '{tool_name}'")
            return 1

        cases = TOOL_TEST_CASES[tool_name]["cases"]
        print(f"\n{'=' * 80}")
        print(f"TESTING TOOL: {tool_name}")
        print(f"DESCRIPTION: {TOOL_TEST_CASES[tool_name]['description']}")
        print(f"{'=' * 80}")

        tool_results = await tester.run_tool_cases(
            tool_name=tool_name,
            tool_function=tool_function,
            cases=cases,
            fail_fast=args.fail_fast,
        )
        results.extend(tool_results)

        for result in tool_results:
            status = "PASS" if result.success else "FAIL"
            print(f"\n[{status}] {result.query}")
            print(f"  tool_called={result.tool_was_called} | expected_found={result.expected_content_found} | duration={result.duration:.2f}s")
            if result.tool_calls:
                print(f"  tool_calls={result.tool_calls}")
            if result.error:
                print(f"  error={result.error}")
            print(f"  response_preview={result.response_preview}")

        if args.fail_fast and any(not r.success for r in tool_results):
            break

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    print(f"\n{'=' * 80}")
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 80}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Wrote JSON results to {args.json_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
