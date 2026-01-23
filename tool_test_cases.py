#!/usr/bin/env python3
"""
Test script for biomedical agent tools.
Tests each tool individually with appropriate test cases.
Run this in the biomedagent conda environment.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# Import biomed agent - should work if running from project root or if package is installed
from bioagent.agent import get_chat_model
from bioagent.agent.tools import think
from bioagent.agent.tools.dbsearch import DBSEARCH_TOOLS
from bioagent.agent.tools.target_search import TARGET_SEARCH_TOOLS, pharmacology_search
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Test cases for each tool
TOOL_TEST_CASES = {
    # Clinical trials tools
    "search_clinical_trials": {
        "description": "Search ClinicalTrials.gov for clinical studies",
        "test_queries": [
            "Find clinical trials for ruxolitinib in cancer patients that are still recruiting",
            "Search for Phase 3 trials in breast cancer with immunotherapy",
            "Find completed trials for metformin in diabetes"
        ]
    },

    "search_drug_labels": {
        "description": "Search FDA drug labels from OpenFDA and DailyMed",
        "test_queries": [
            "What are the warnings and side effects for aspirin?",
            "Find contraindications for metformin",
            "What are the drug interactions for warfarin?"
        ]
    },

    # Pharmacology tools
    "search_drug_targets": {
        "description": "Find all protein targets for a drug",
        "test_queries": [
            "What proteins does imatinib target?",
            "Find all targets for aspirin",
            "What are the molecular targets of pembrolizumab?"
        ]
    },

    "search_target_drugs": {
        "description": "Find all drugs that modulate a specific protein target",
        "test_queries": [
            "What drugs inhibit EGFR?",
            "Find all compounds that target ABL1",
            "What medications affect JAK2?"
        ]
    },

    "search_similar_molecules": {
        "description": "Find molecules structurally similar to a query compound",
        "test_queries": [
            "Find molecules similar to aspirin",
            "What compounds are structurally related to caffeine?",
            "Find analogs of ibuprofen"
        ]
    },

    "search_exact_structure": {
        "description": "Find an exact structure match for a molecule",
        "test_queries": [
            "Identify this molecule: CC(=O)Oc1ccccc1C(=O)O",
            "What is the name of this compound: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Find information for SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        ]
    },

    "search_substructure": {
        "description": "Find molecules containing a specific substructure",
        "test_queries": [
            "Find compounds containing a benzene ring",
            "What molecules have an amide group?",
            "Find drugs with a pyridine ring"
        ]
    },

    "get_drug_profile": {
        "description": "Get comprehensive profile for a drug",
        "test_queries": [
            "Give me a complete profile of imatinib including targets, forms, and trials",
            "What is the full pharmacological profile of aspirin?",
            "Comprehensive information about pembrolizumab"
        ]
    },

    "get_drug_forms": {
        "description": "Get all molecular forms of a drug",
        "test_queries": [
            "What are all the salt forms of metformin?",
            "Find different formulations of insulin",
            "What are the stereoisomers of thalidomide?"
        ]
    },

    "search_drug_trials": {
        "description": "Find clinical trials for a drug",
        "test_queries": [
            "Find clinical trials for ruxolitinib",
            "What trials are there for pembrolizumab in cancer?",
            "Clinical studies involving metformin"
        ]
    },

    "compare_drugs_on_target": {
        "description": "Compare multiple drugs' activity against a single target",
        "test_queries": [
            "Compare imatinib, dasatinib, and nilotinib on ABL1",
            "How do gefitinib, erlotinib, and osimertinib compare on EGFR?",
            "Compare different JAK inhibitors on JAK2"
        ]
    },

    "search_selective_drugs": {
        "description": "Find drugs that are selective for one target over others",
        "test_queries": [
            "Find JAK2-selective inhibitors that spare JAK1 and JAK3",
            "Find EGFR inhibitors that don't affect ERBB2",
            "Selective CDK4 inhibitors that spare CDK6"
        ]
    },

    "pharmacology_search": {
        "description": "Unified pharmacology search tool",
        "test_queries": [
            "Find all targets for imatinib using unified search",
            "Search for drugs targeting EGFR with unified tool",
            "Get comprehensive drug profile for aspirin using unified search"
        ]
    }
}

async def test_single_tool(tool_name: str, tool_function, test_queries: List[str], model):
    """Test a single tool with multiple queries."""
    print(f"\n{'='*80}")
    print(f"TESTING TOOL: {tool_name}")
    print(f"DESCRIPTION: {TOOL_TEST_CASES[tool_name]['description']}")
    print(f"{'='*80}")

    # Create agent with only this tool
    checkpointer = InMemorySaver()
    system_prompt = f"""
    You are a biomedical agent testing the {tool_name} tool.
    Your task is to use the {tool_name} tool to answer user questions.
    Be direct and use the tool appropriately.
    """

    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[think, tool_function],
        checkpointer=checkpointer,
        debug=False  # Disable debug for cleaner output
    )

    config = {"configurable": {"thread_id": f"test_{tool_name}_{datetime.now().isoformat()}"}}

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")

        try:
            human_msg = HumanMessage(query)
            start_time = datetime.now()

            response = await agent.ainvoke({'messages': [human_msg]}, config)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"Duration: {duration:.2f} seconds")

            # Print the final response
            if response and 'messages' in response:
                last_message = response['messages'][-1]
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    # Truncate very long responses for readability
                    if len(content) > 2000:
                        print(f"Response (truncated): {content[:2000]}...")
                    else:
                        print(f"Response: {content}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

def get_tool_function(tool_name: str):
    """Get the tool function by name."""
    # Check DBSEARCH_TOOLS
    for tool in DBSEARCH_TOOLS:
        if tool.name == tool_name:
            return tool

    # Check TARGET_SEARCH_TOOLS
    for tool in TARGET_SEARCH_TOOLS:
        if tool.name == tool_name:
            return tool

    # Special case for unified pharmacology search
    if tool_name == "pharmacology_search":
        return pharmacology_search

    return None

async def main():
    """Main test function."""
    print("Setting up biomedical agent tool tests...")

    # Initialize the model
    try:
        model = get_chat_model("google/gemini-2.5-flash", "openrouter", model_parameters={"temperature": 0.5})
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return

    # Get all tool names to test
    all_tools = list(TOOL_TEST_CASES.keys())

    # Allow command line selection of specific tools
    if len(sys.argv) > 1:
        requested_tools = sys.argv[1:]
        tools_to_test = [t for t in requested_tools if t in all_tools]
        if not tools_to_test:
            print(f"No valid tools specified. Available: {all_tools}")
            return
    else:
        tools_to_test = all_tools

    print(f"Testing tools: {tools_to_test}")

    # Test each tool
    for tool_name in tools_to_test:
        tool_function = get_tool_function(tool_name)
        if not tool_function:
            print(f"WARNING: Could not find function for tool '{tool_name}'")
            continue

        test_queries = TOOL_TEST_CASES[tool_name]["test_queries"]
        await test_single_tool(tool_name, tool_function, test_queries, model)

    print(f"\n{'='*80}")
    print("TOOL TESTING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())