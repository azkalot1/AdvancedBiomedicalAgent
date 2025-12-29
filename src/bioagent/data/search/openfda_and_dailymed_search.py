# openfda_and_dailymed_searches.py
import asyncio
import re
from collections import defaultdict
from typing import Any
import os
import sys
from contextlib import redirect_stderr
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Assuming these are in the user's environment as specified
from bioagent.data.ingest.async_config import AsyncDatabaseConfig, get_async_connection
from bioagent.data.ingest.config import DatabaseConfig
# from bioagent.api import dailymed_find  # TODO: Fix this import


async def dailymed_find(drug_name: str, limit: int = 10) -> list[dict]:
    """Placeholder function for dailymed_find API call."""
    # TODO: Implement actual DailyMed API search
    return []


SIMILARITY_THRESHOLD = 0.87
SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def _strip_chemical_suffixes(drug_name: str) -> str:
    """
    Removes common chemical suffixes from a drug name to improve search recall.
    For example, "Lisinopril Dihydrate" becomes "Lisinopril".
    Also handles cases like "%drugname% (topical)" -> "%drugname%".
    """
    # A list of common suffixes to strip. This list can be expanded.
    suffixes = [
        "HCL", "hydrochloride", "sulfate", "sodium", "potassium",
        "acetate", "maleate", "tartrate", "phosphate", "dihydrate",
        "monohydrate", "besylate", "mesylate", "succinate", "fumarate",
        "topical", "ointment", "cream", "gel", "spray", "solution", "suspension", "elixir", "tablet", "capsule", "pill", 
    ]
    
    # First, remove parenthetical suffixes like "(topical)", "(cream)", etc.
    # This handles cases like "%drugname% (topical)" -> "%drugname%"
    parenthetical_pattern = r"\s*\([^)]*\)\s*$"
    stripped_name = re.sub(parenthetical_pattern, "", drug_name, flags=re.IGNORECASE)
    
    # Build a case-insensitive regex pattern to match whole words
    # The \b ensures that we don't accidentally strip parts of words.
    pattern = r"\b(" + "|".join(suffixes) + r")\b"
    
    # Substitute the found suffixes with an empty string
    stripped_name = re.sub(pattern, "", stripped_name, flags=re.IGNORECASE)
    
    # Clean up any resulting double spaces or leading/trailing whitespace
    return " ".join(stripped_name.strip().split())

# --- 1. Pydantic Models ---


class DrugProperties(BaseModel):
    smiles: str | None = None
    formula: str | None = None
    inchi_key: str | None = None
    name: str | None = None


class SearchResultItem(BaseModel):
    section_name: str
    text: str
    source: str


class DailyMedAndOpenFDASearchResult(BaseModel):
    # This name now corresponds to the user's search term when drug_names are provided.
    product_name: str
    properties: DrugProperties | None = Field(default=None)
    sections: list[SearchResultItem] = Field(default_factory=list)

    def pretty_print(self, console: Console | None = None) -> None:
        """Pretty print a single search result."""
        console = console or Console()
        
        tree = Tree(f"[bold cyan]📦 {self.product_name}[/bold cyan]")
        
        if self.properties:
            props_branch = tree.add("[bold yellow]Properties[/bold yellow]")
            for key, value in self.properties.model_dump().items():
                if value is not None:
                    props_branch.add(f"[dim]{key}:[/dim] {value}")
        
        if self.sections:
            sections_branch = tree.add(f"[bold green]Sections ({len(self.sections)})[/bold green]")
            for section in self.sections:
                section_data = section.model_dump()
                section_name = section_data.get('section_name', 'Untitled')
                section_data = section_data.get('text', 'Untitled')[:100] + "..." if len(section_data.get('text', 'Untitled')) > 100 else section_data.get('text', 'Untitled')
                sections_branch.add(f"[white]• {section_name}[/white]: {section_data}")
        
        console.print(tree)

    def __str__(self) -> str:
        return f"SearchResult(product_name='{self.product_name}', sections={len(self.sections)})"



class DailyMedAndOpenFDASearchOutput(BaseModel):
    status: str
    results: list[DailyMedAndOpenFDASearchResult] | None = None
    error: str | None = None

    def pretty_print(self, console: Console | None = None, verbose: bool = False) -> None:
        """Pretty print the search output with optional verbosity."""
        console = console or Console()
        
        # Status header
        status_color = "green" if self.status == "success" else "red"
        console.print(Panel(
            f"[bold {status_color}]{self.status.upper()}[/bold {status_color}]",
            title="🔍 Search Results",
            border_style=status_color
        ))
        
        # Error handling
        if self.error:
            console.print(f"[bold red]❌ Error:[/bold red] {self.error}")
            return
        
        # Results
        if not self.results:
            console.print("[dim]No results found.[/dim]")
            return
        
        console.print(f"\n[bold]Found {len(self.results)} result(s):[/bold]\n")
        
        for i, result in enumerate(self.results, 1):
            console.print(f"[bold magenta]━━━ Result {i} ━━━[/bold magenta]")
            result.pretty_print(console)
            console.print()

    def to_table(self, console: Console | None = None) -> None:
        """Display results as a table."""
        console = console or Console()
        
        table = Table(title="DailyMed & OpenFDA Search Results", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Product Name", style="cyan", no_wrap=True)
        table.add_column("Has Properties", justify="center")
        table.add_column("Sections", justify="right", style="green")
        
        if self.results:
            for i, result in enumerate(self.results, 1):
                table.add_row(
                    str(i),
                    result.product_name,
                    "✓" if result.properties else "✗",
                    str(len(result.sections))
                )
        
        console.print(table)

    def to_json(self, indent: int = 2) -> str:
        """Return formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def __str__(self) -> str:
        count = len(self.results) if self.results else 0
        return f"SearchOutput(status='{self.status}', results={count})"



class DailyMedAndOpenFDAInput(BaseModel):
    drug_names: list[str] | None = Field(default=None, min_length=1)
    section_queries: list[str] | None = Field(default=None, min_length=1)
    keyword_query: list[str] | None = Field(default=None, min_length=1)
    fetch_all_sections: bool = Field(default=False)
    top_n_drugs: int = Field(default=3, ge=1, le=10)
    sections_per_query: int = Field(default=2, ge=1, le=5)
    result_limit: int = Field(default=10, ge=1, le=50)
    aggressive_deduplication: bool = Field(default=True)

    @model_validator(mode='after')
    def check_valid_use_case(self) -> 'DailyMedAndOpenFDAInput':
        is_property_lookup = (
            self.drug_names and not self.section_queries and not self.keyword_query and not self.fetch_all_sections
        )
        is_get_all_info = self.drug_names and self.fetch_all_sections
        is_global_keyword_search = self.keyword_query and not self.drug_names and not self.section_queries
        is_advanced_label_search = (self.keyword_query or self.section_queries) and not self.fetch_all_sections
        if not (is_property_lookup or is_get_all_info or is_global_keyword_search or is_advanced_label_search):
            raise ValueError("Invalid search combination. Please consult the tool's documentation for valid patterns.")
        return self


    def pretty_print(self, console: Console | None = None) -> None:
        """Pretty print a single search result."""
        console = console or Console()
        
        tree = Tree(f"[bold cyan]📦 {self.product_name}[/bold cyan]")
        
        if self.properties:
            props_branch = tree.add("[bold yellow]Properties[/bold yellow]")
            for key, value in self.properties.model_dump().items():
                if value is not None:
                    props_branch.add(f"[dim]{key}:[/dim] {value}")
        
        if self.sections:
            sections_branch = tree.add(f"[bold green]Sections ({len(self.sections)})[/bold green]")
            for section in self.sections:
                section_data = section.model_dump()
                title = section_data.get('title', 'Untitled')
                sections_branch.add(f"[blue]• {title}[/blue]")
        
        console.print(tree)

    def __str__(self) -> str:
        return f"SearchResult(product_name='{self.product_name}', sections={len(self.sections)})"



# --- 2. Helper Functions ---


def _sanitize_for_tsquery(text: str) -> str:
    text = str(text)
    text = re.sub(r"[!&|():*<>\"'\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _deduplicate_by_embedding_sync(results: list[SearchResultItem], model: SentenceTransformer) -> list[SearchResultItem]:
    if not results:
        return []
    sorted_results = sorted(results, key=lambda item: len(item.text), reverse=True)
    all_embeddings = model.encode([r.text for r in sorted_results], show_progress_bar=False)
    unique_results, kept_embeddings = [], []
    for i, item in enumerate(sorted_results):
        embedding = all_embeddings[i].reshape(1, -1)
        if not (kept_embeddings and np.any(cosine_similarity(embedding, np.vstack(kept_embeddings)) > SIMILARITY_THRESHOLD)):
            unique_results.append(item)
            kept_embeddings.append(embedding)
    return unique_results


# --- 3. Core Data Fetching Logic ---


async def _find_drug_properties_async(async_config: AsyncDatabaseConfig, drug_name: str) -> DrugProperties | None:
    sanitized_name = _sanitize_for_tsquery(drug_name)
    ts_query = " & ".join(sanitized_name.strip().split())
    if not ts_query:
        return None
    query = "SELECT smiles, formula, inchi_key, name FROM drugcentral_drugs WHERE name_vector @@ to_tsquery('english', $1) OR synonyms_vector @@ to_tsquery('english', $1) ORDER BY ts_rank(name_vector, to_tsquery('english', $1)) + ts_rank(synonyms_vector, to_tsquery('english', $1)) DESC LIMIT 1;"
    row = await async_config.execute_one(query, ts_query)
    return DrugProperties(**row) if row else None


async def _search_structured_sections_async(
    async_config: AsyncDatabaseConfig, set_id_ids: list[int], search_input: DailyMedAndOpenFDAInput, model: SentenceTransformer
) -> list:
    if not set_id_ids:
        return []
    params: list[Any] = [set_id_ids]
    query = "SELECT DISTINCT lm.title AS product_name, sn.section_name, s.text, 'openfda' AS source FROM sections s JOIN section_names sn ON s.section_name_id = sn.id JOIN labels_meta lm ON s.set_id_id = lm.set_id_id WHERE s.set_id_id = ANY($1)"

    if not search_input.fetch_all_sections:
        if search_input.section_queries:
            all_relevant_section_ids = set()
            for s_query in search_input.section_queries:
                query_embedding = model.encode(s_query, show_progress_bar=False)
                rows = await async_config.execute_query(
                    "SELECT id FROM section_names ORDER BY name_embedding <=> $1 LIMIT $2;",
                    query_embedding.tolist(),
                    search_input.sections_per_query,
                )
                all_relevant_section_ids.update(row['id'] for row in rows)
            if not all_relevant_section_ids:
                return []
            query += f" AND s.section_name_id = ANY(${len(params) + 1})"
            params.append(list(all_relevant_section_ids))
        if search_input.keyword_query:
            keyword_tsquery = " | ".join(
                [
                    f"({' & '.join(_sanitize_for_tsquery(phrase).split())})" 
                    for phrase in search_input.keyword_query
                ]
            )
            query += f" AND s.text_vector @@ to_tsquery('english', ${len(params) + 1})"
            params.append(keyword_tsquery)

    query += f" LIMIT ${len(params) + 1};"
    params.append(search_input.result_limit)
    return await async_config.execute_query(query, *params)


async def _search_dailymed_structured_async(
    async_config: AsyncDatabaseConfig, set_ids: list[str], search_input: DailyMedAndOpenFDAInput, model: SentenceTransformer
) -> list:
    if not set_ids:
        return []
    params: list[Any] = [set_ids]
    query = """
        SELECT dp.product_name, dsn.section_name, ds.text, 'dailymed' AS source 
        FROM dailymed_sections ds 
        JOIN dailymed_section_names dsn ON ds.section_name_id = dsn.id 
        JOIN dailymed_products dp ON ds.set_id = dp.set_id 
        WHERE ds.set_id = ANY($1)
    """

    if not search_input.fetch_all_sections:
        if search_input.section_queries:
            all_relevant_section_ids = set()
            for s_query in search_input.section_queries:
                query_embedding = model.encode(s_query, show_progress_bar=False)  # TO DO: make async
                rows = await async_config.execute_query(
                    "SELECT id FROM dailymed_section_names ORDER BY embedding <=> $1 LIMIT $2;",
                    query_embedding.tolist(),
                    search_input.sections_per_query,
                )
                all_relevant_section_ids.update(row['id'] for row in rows)
            if not all_relevant_section_ids:
                return []
            query += f" AND ds.section_name_id = ANY(${len(params) + 1})"
            params.append(list(all_relevant_section_ids))
        if search_input.keyword_query:
            keyword_tsquery = " | ".join(
                [
                    f"({' & '.join(_sanitize_for_tsquery(phrase).split())})" 
                    for phrase in search_input.keyword_query
                ]
            )
            query += f" AND ds.text_vector @@ to_tsquery('english', ${len(params) + 1})"
            params.append(keyword_tsquery)

    query += f" LIMIT ${len(params) + 1};"
    params.append(search_input.result_limit)
    return await async_config.execute_query(query, *params)


async def _find_ids_for_single_drug_async(
    async_config: AsyncDatabaseConfig, drug_name: str, top_n: int
) -> tuple[list[int], list[str]]:
    chemically_stripped_name = _strip_chemical_suffixes(drug_name)
    sanitized_name = _sanitize_for_tsquery(drug_name)
    sanitized_name = _sanitize_for_tsquery(chemically_stripped_name)
    drug_tsquery = " & ".join(sanitized_name.strip().split())

    if not drug_tsquery:
        return [], []

    openfda_query = "WITH ranked AS (SELECT set_id_id, 1 AS p FROM mapping_brand_name WHERE to_tsvector('english', brand_name) @@ to_tsquery('english', $1) UNION ALL SELECT set_id_id, 2 AS p FROM mapping_generic_name WHERE to_tsvector('english', generic_name) @@ to_tsquery('english', $1) UNION ALL SELECT set_id_id, 3 AS p FROM mapping_substance_name WHERE to_tsvector('english', substance_name) @@ to_tsquery('english', $1)), dist AS (SELECT DISTINCT ON (set_id_id) set_id_id, p FROM ranked ORDER BY set_id_id, p) SELECT d.set_id_id, s.set_id FROM dist d JOIN set_ids s ON d.set_id_id = s.id ORDER BY d.p LIMIT $2;"
    dailymed_query = "SELECT set_id FROM dailymed_products WHERE name_vector @@ to_tsquery('english', $1) LIMIT $2;"

    openfda_rows, dailymed_rows, dailymed_api_search = await asyncio.gather(
        async_config.execute_query(openfda_query, drug_tsquery, top_n),
        async_config.execute_query(dailymed_query, drug_tsquery, top_n),
        dailymed_find(drug_name, limit=top_n),
    )
    set_ids_api_search = [x['setid'] for x in dailymed_api_search]
    set_id_ids = [row['set_id_id'] for row in openfda_rows]
    set_ids = {row['set_id'] for row in openfda_rows}
    set_ids.update(row['set_id'] for row in dailymed_rows)
    set_ids.update(set_ids_api_search)
    return set_id_ids, list(set_ids)


async def _discover_drug_ids_by_keyword_async(
    async_config: AsyncDatabaseConfig, search_input: DailyMedAndOpenFDAInput
) -> tuple[list[int], list[str]]:
    if not search_input.keyword_query:
        return [], []
    keyword_tsquery = " | ".join(
        [
            f"({' & '.join(_sanitize_for_tsquery(phrase).split())})" 
            for phrase in search_input.keyword_query
        ]
    )

    openfda_discover_query = "SELECT DISTINCT s.set_id_id, si.set_id FROM sections s JOIN set_ids si ON s.set_id_id = si.id WHERE s.text_vector @@ to_tsquery('english', $1) LIMIT $2;"
    dailymed_discover_query = (
        "SELECT DISTINCT set_id FROM dailymed_sections WHERE text_vector @@ to_tsquery('english', $1) LIMIT $2;"
    )

    openfda_rows, dailymed_rows = await asyncio.gather(
        async_config.execute_query(openfda_discover_query, keyword_tsquery, search_input.top_n_drugs),
        async_config.execute_query(dailymed_discover_query, keyword_tsquery, search_input.top_n_drugs),
    )
    set_id_ids = [row['set_id_id'] for row in openfda_rows]
    set_ids = {row['set_id'] for row in openfda_rows}
    set_ids.update(row['set_id'] for row in dailymed_rows)
    return set_id_ids, list(set_ids)


# --- 4. Main Search Orchestration Logic ---


async def _group_and_enrich_discovery_results(
    raw_results: list, async_config: AsyncDatabaseConfig, model: SentenceTransformer
) -> list[DailyMedAndOpenFDASearchResult]:
    """Groups results from a discovery search by the product name found in the database."""
    grouped_by_drug: dict[str, list[SearchResultItem]] = defaultdict(list)
    for row in raw_results:
        product_name = str(row['product_name']).strip() if row.get('product_name') else 'Unknown Product'
        grouped_by_drug[product_name].append(SearchResultItem(**row))

    enriched_results = []
    for product_name, sections in grouped_by_drug.items():
        deduplicated_sections = _deduplicate_by_embedding_sync(sections, model)
        if not deduplicated_sections:
            continue
        properties = await _find_drug_properties_async(async_config, product_name)
        enriched_results.append(
            DailyMedAndOpenFDASearchResult(product_name=product_name, properties=properties, sections=deduplicated_sections)
        )
    return enriched_results


async def _handle_property_lookup(async_config: AsyncDatabaseConfig, drug_names: list[str]) -> list[DailyMedAndOpenFDASearchResult]:
    tasks = [_find_drug_properties_async(async_config, name) for name in drug_names]
    all_properties = await asyncio.gather(*tasks)
    return [DailyMedAndOpenFDASearchResult(product_name=p.name, properties=p) for p in all_properties if p and p.name]


async def _handle_label_search(
    async_config: AsyncDatabaseConfig, search_input: DailyMedAndOpenFDAInput, model: SentenceTransformer
) -> list[DailyMedAndOpenFDASearchResult]:
    if search_input.drug_names:
        # **NEW AGGREGATION LOGIC**: Iterate and build one result object per searched drug name.
        final_results = []
        for drug_name in search_input.drug_names:
            set_id_ids, set_ids = await _find_ids_for_single_drug_async(async_config, drug_name, search_input.top_n_drugs)
            if not set_id_ids and not set_ids:
                continue

            structured_task = _search_structured_sections_async(async_config, set_id_ids, search_input, model)
            dailymed_task = _search_dailymed_structured_async(async_config, set_ids, search_input, model)
            results_structured, results_dailymed = await asyncio.gather(structured_task, dailymed_task)

            all_sections_for_this_drug = [SearchResultItem(**r) for r in results_structured + results_dailymed]
            if not all_sections_for_this_drug:
                continue

            deduplicated_sections = _deduplicate_by_embedding_sync(all_sections_for_this_drug, model)
            properties = await _find_drug_properties_async(async_config, drug_name)

            aggregated_result = DailyMedAndOpenFDASearchResult(
                product_name=drug_name, properties=properties, sections=deduplicated_sections
            )
            final_results.append(aggregated_result)
        return final_results
    else:
        # **DISCOVERY SEARCH**: Find drugs by keyword and group by database product name.
        set_id_ids, set_ids = await _discover_drug_ids_by_keyword_async(async_config, search_input)
        if not set_id_ids and not set_ids:
            return []

        structured_task = _search_structured_sections_async(async_config, set_id_ids, search_input, model)
        dailymed_task = _search_dailymed_structured_async(async_config, set_ids, search_input, model)
        results_structured, results_dailymed = await asyncio.gather(structured_task, dailymed_task)

        raw_results = results_structured + results_dailymed
        if not raw_results:
            return []

        return await _group_and_enrich_discovery_results(raw_results, async_config, model)


async def dailymed_and_openfda_search_async(db_config: DatabaseConfig, search_input: DailyMedAndOpenFDAInput) -> DailyMedAndOpenFDASearchOutput:
    try:
        async_config = await get_async_connection(db_config)

        if (
            search_input.drug_names
            and not search_input.section_queries
            and not search_input.keyword_query
            and not search_input.fetch_all_sections
        ):
            results = await _handle_property_lookup(async_config, search_input.drug_names)
        else:
            results = await _handle_label_search(async_config, search_input, SENTENCE_MODEL)

        status = "not_found" if not results else "success"
        error = "No documents matched all criteria." if not results else None
        return DailyMedAndOpenFDASearchOutput(status=status, results=results or None, error=error)
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        return DailyMedAndOpenFDASearchOutput(status="error", error=f"An unexpected server error occurred: {type(e).__name__}")
