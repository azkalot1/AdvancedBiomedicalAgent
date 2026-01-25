# Advanced Biomedical Agent

A comprehensive biomedical research agent that integrates multiple data sources (OpenFDA, DailyMed, ClinicalTrials.gov, ChEMBL, BindingDB, DrugCentral, Orange Book) into a unified PostgreSQL database with LLM-friendly search tools.

## Overview

The Advanced Biomedical Agent provides:

- **Unified Database**: Integrates 9 major biomedical data sources into a single PostgreSQL database
- **Intelligent Search Tools**: Both retrieval-based and agent-based tools for querying biomedical data
- **LLM Integration**: Tools designed for use with language models, with automatic summarization and formatting
- **Comprehensive Coverage**: Clinical trials, drug labels, molecular structures, target interactions, adverse events, and more

## Features

### Data Sources Integrated

1. **OpenFDA** - Drug labels with normalized structure
2. **DailyMed** - SPL drug labels with semantic search
3. **ClinicalTrials.gov** - Clinical trials from CTTI AACT dump
4. **Orange Book** - FDA therapeutic equivalence data
5. **BindingDB** - Molecular targets & binding affinity
6. **ChEMBL** - Biochemical annotations
7. **DrugCentral** - Molecular structures & chemical data
8. **Target Mapping** - Canonical target mapping (ChEMBL + BindingDB)
9. **Molecule Mapping** - Unified molecules + ClinicalTrials.gov/product mappings

### Tool Categories

#### Retrieval-Based Tools (`@search`)
Direct database access functions that return structured data:
- Clinical trials search with flexible filtering
- Drug labels search (OpenFDA + DailyMed)
- Molecule-trial connectivity
- Adverse events search
- Clinical trial outcomes
- Orange Book search
- Cross-database identifier lookup
- Biotherapeutic sequence search
- Target/drug pharmacology search

#### Agent-Based Tools (`@tools`)
LLM-friendly wrappers with formatting and summarization:
- Automatic output summarization for long results
- Input normalization for robust LLM integration
- Helpful error messages with suggestions
- Research output management (store/retrieve full results)
- Web search capabilities
- Thinking/reasoning tool

### Key Capabilities

- **Clinical Trial Search**: Search 500,000+ clinical trials with flexible querying
- **Drug Information**: Access FDA drug labels, indications, warnings, and interactions
- **Pharmacology**: Drug-target interactions, mechanisms of action, activity data (IC50, Ki, Kd)
- **Molecular Search**: Structure similarity, substructure search, exact structure matching
- **Safety Data**: Adverse events, outcomes, and safety comparisons
- **Regulatory Data**: Orange Book patents, exclusivity, and therapeutic equivalence
- **Cross-Database Lookup**: Resolve identifiers across multiple databases

## Quick Start

### Prerequisites

- **PostgreSQL 14+** (recommended: 16)
- **Python 3.12+**
- **~50 GB disk space** for full database
- **8+ GB RAM** recommended for ingestion

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AdvancedBiomedicalAgent

# Install the package
pip install -e .

# Install optional dependencies
pip install -e ".[chem]"  # For RDKit support
pip install -e ".[dev]"   # For development
```

### Database Setup

See the [Setup Guide](docs/SETUP_GUIDE.md) for detailed instructions.

```bash
# Quick setup
biomedagent-db setup-db
biomedagent-db install-extensions
```

### Data Ingestion

See the [Data Population Guide](docs/README_POPULATE.md) for detailed instructions.

```bash
# Full ingestion (all sources)
biomedagent-db ingest

# Selective ingestion
biomedagent-db ingest --skip-openfda --skip-ctgov

# Use local files (skip download)
biomedagent-db ingest --openfda-use-local-files
```

### Using the Tools

```python
from bioagent.agent.tools import get_summarized_tools
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Get tools with summarization
tools = get_summarized_tools(
    summarizer_llm=llm,
    session_id="my-session"
)

# Use tools in your agent
# Tools are automatically registered and ready to use
```

## Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Complete database setup instructions
- **[Database Schema](docs/DATABASE_SCHEMA.md)** - Detailed database structure documentation
- **[Data Population Guide](docs/README_POPULATE.md)** - How to ingest data from all sources
- **[Molecular Mapping Guide](docs/MOLECULAR_MAPPING_GUIDE.md)** - Understanding molecular mappings
- **[Tools Documentation](docs/tools.md)** - Comprehensive tool reference

## Project Structure

```
AdvancedBiomedicalAgent/
├── src/bioagent/
│   ├── agent/
│   │   └── tools/          # Agent-based tools (@tools)
│   │       ├── dbsearch.py
│   │       ├── target_search.py
│   │       ├── web_search.py
│   │       ├── summarizing.py
│   │       └── thinking.py
│   ├── data/
│   │   ├── search/         # Retrieval-based tools (@search)
│   │   │   ├── clinical_trial_search.py
│   │   │   ├── target_search.py
│   │   │   ├── openfda_and_dailymed_search.py
│   │   │   └── ...
│   │   └── ingest/         # Data ingestion scripts
│   └── cli/                 # Command-line interface
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
└── tests/                   # Test suite
```

## Tool Architecture

The system uses a two-layer architecture:

1. **Retrieval Tools** (`@search`): Direct database access functions
   - Return structured Pydantic models
   - Async/await for performance
   - Located in `src/bioagent/data/search/`

2. **Agent Tools** (`@tools`): LLM-friendly wrappers
   - Format outputs for LLM consumption
   - Automatic summarization for long outputs
   - Input normalization and validation
   - Located in `src/bioagent/agent/tools/`

See [Tools Documentation](docs/tools.md) for complete details.

## Example Usage

### Search Clinical Trials

```python
# Using agent tool
result = await search_clinical_trials(
    condition="non-small cell lung cancer",
    intervention="pembrolizumab",
    phase="Phase 3",
    status="Recruiting",
    limit=10
)
```

### Search Drug Targets

```python
# Find all targets for a drug
result = await search_drug_targets(
    drug_name="imatinib",
    min_pchembl=6.0,
    data_source="both"
)
```

### Search Drug Labels

```python
# Get drug label information
result = await search_drug_labels(
    drug_names="warfarin",
    section_queries=["warnings", "drug interactions"],
    fetch_all_sections=False
)
```

## CLI Commands

```bash
# Database management
biomedagent-db setup-db              # Create database
biomedagent-db install-extensions     # Install pgvector, RDKit
biomedagent-db verify-deps            # Verify dependencies

# Data ingestion
biomedagent-db ingest                 # Full ingestion
biomedagent-db ingest --skip-<source> # Skip specific source

# Database utilities
biomedagent-db vacuum                 # Optimize database
biomedagent-db stats                  # Show database statistics
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## Requirements

- PostgreSQL 14+ with pgvector and RDKit extensions
- Python 3.12+
- ~50 GB disk space for full database
- 8+ GB RAM for ingestion

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please see the documentation for:
- Database schema details
- Tool development guidelines
- Data ingestion processes

## Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review the [Setup Guide](docs/SETUP_GUIDE.md) for common issues
3. Check the [Database Schema](docs/DATABASE_SCHEMA.md) for data structure details

---

**Note**: This project integrates large-scale biomedical datasets. Ensure you have adequate disk space and follow the setup guide carefully.
