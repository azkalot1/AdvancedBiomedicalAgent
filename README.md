# Advanced Biomedical Agent

Biomedical research agent that integrates OpenFDA, DailyMed, ClinicalTrials.gov, ChEMBL, BindingDB, DrugCentral, Orange Book and related sources into a unified PostgreSQL database, with LLM-friendly search tools and a **chat interface**.

## Quick Start

**Prerequisites:** PostgreSQL 14+, Python 3.12+, ~50 GB disk, 8+ GB RAM for full ingestion.

```bash
git clone <repository-url>
cd AdvancedBiomedicalAgent
pip install -e .
# Optional: pip install -e ".[chem]"  # RDKit; pip install -e ".[dev]"  # dev
```

**Database:** See [Setup Guide](docs/SETUP_GUIDE.md).

```bash
biomedagent-db setup-db
biomedagent-db install-extensions
biomedagent-db ingest   # full ingestion; see docs/README_POPULATE.md for options
```

## Chat (recommended)

Terminal chat with the agent (LangGraph + SQLite checkpoints, per-thread research outputs):

```bash
export OPENROUTER_API_KEY=...   # or use .env; or --provider openai with OPENAI_API_KEY
biomedagent-db chat
```

**Example session:**

```
Biomedical Agent — terminal chat
Commands: /new, /thread [id], /list, /history, /quit
Thread: <uuid>

You> What phase 3 trials are recruiting for NSCLC and pembrolizumab?
Agent> [uses search_clinical_trials, summarizes] ...
  [Refs: search_clinical_trials_abc123]

You> /list
  **Stored Research Outputs (current thread):**
  - search_clinical_trials_abc123 (12,340 chars)
  Use retrieve_full_output(ref_id) in a message to fetch full content.

You> /quit
Bye.
```

**Chat options:** `biomedagent-db chat --help` (e.g. `--thread-id`, `--checkpoint-db`, `--model`, `--provider`, `--debug`).

## Using tools in code

```python
from bioagent.agent.tools import get_summarized_tools
from bioagent.agent import get_chat_model

llm = get_chat_model("google/gemini-2.5-flash", "openrouter", {"temperature": 0.5})
tools = get_summarized_tools(llm, session_id="my-session")
# Use tools with your agent (see docs/tools.md)
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Setup Guide](docs/SETUP_GUIDE.md) | PostgreSQL setup, extensions, troubleshooting |
| [Data Population](docs/README_POPULATE.md) | Ingestion and source options |
| [Database Schema](docs/DATABASE_SCHEMA.md) | Table layout |
| [Tools](docs/tools.md) | Retrieval vs agent tools, API reference |
| [Molecular Mapping](docs/MOLECULAR_MAPPING_GUIDE.md) | Molecule/target mappings |

## CLI summary

```bash
biomedagent-db setup-db              # Create DB
biomedagent-db install-extensions    # pgvector, RDKit
biomedagent-db verify-deps          # Check dependencies
biomedagent-db ingest [--skip-*]     # Ingest data
biomedagent-db chat [options]        # Chat REPL
biomedagent-db vacuum               # Optimize DB
biomedagent-db stats                 # DB statistics
```

## License

See [LICENSE](LICENSE). For schema, tool development, and ingestion details, see the docs above.
