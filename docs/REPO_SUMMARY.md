# Advanced Biomedical Agent Repo Summary

This document summarizes what exists in `AdvancedBiomedicalAgent`, how it is deployed on Railway, what is strong about it, what problems it can solve, and how users can interact with it.

## 1) What Data Is There

### Core integrated biomedical sources

- ClinicalTrials.gov (CT.gov / AACT): studies, conditions, interventions, outcomes, adverse events.
- OpenFDA: normalized drug label/product data.
- DailyMed: structured SPL drug label sections.
- FDA Orange Book: approvals, products, ingredients, therapeutic equivalence context.
- ChEMBL: molecules, targets, activities, and biotherapeutics.
- BindingDB: molecule-target binding activity.
- DrugCentral: drug structures and identifier mappings.

### Integrated model and search layers

- Unified molecule model: `dm_molecule_concept`, `dm_molecule_stereo`, `dm_molecule`, synonym/mapping tables.
- Biotherapeutics layer: `dm_biotherapeutic` and related component/synonym tables.
- Molecule-trial and product-molecule mappings: `map_ctgov_molecules`, `map_product_molecules`.
- Canonical target layer: `dm_target` plus gene/uniprot mapping tables.
- CT.gov semantic/search helpers: `rag_study_corpus` and `rag_study_keys` for retrieval-oriented lookups.
- Denormalized CT.gov search table: `ctgov_enriched_search` for indexed filtering and search.

### Runtime/user data

- Thread/run state persisted by Aegra using Postgres when `AEGRA_POSTGRES_URI` is configured.
- Tool-generated reports persisted in `bioagent_reports` (scoped by `user_id` and `thread_id`).

### Repository data artifacts

- Large DB backup artifact: `bioagent.dump` (~10.7 GB file size).
- Evaluation assets: `tests/ground_truth/*.yaml` with curated biomedical QA targets.
- Test result artifacts: `results*.json` and tool evaluation outputs.
- Notebook research outputs under `notebooks/research_outputs/`.

## 2) How It Is Deployed (Railway)

### Backend runtime model

- Backend is migrated to Aegra runtime (not the previous licensed runtime).
- Startup is handled by `scripts/run_aegra.sh`.
- Graph wiring from `langgraph.json`: `co_scientist` -> `src/bioagent/server/graph.py:graph`.
- HTTP app wiring from `langgraph.json`: `src/bioagent/server/webapp.py:app`.

### Railway service requirements

Set these on the Railway backend service:

```bash
AEGRA_POSTGRES_URI=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
AEGRA_REDIS_URL=redis://default:PASSWORD@redis.railway.internal:6379
AEGRA_CONFIG_PATH=/app/langgraph.json
```

Also keep aliases for compatibility where needed:

```bash
DATABASE_URL=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
APP_DATABASE_URL=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
POSTGRES_URI=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
```

`scripts/run_aegra.sh` auto-maps `DATABASE_URL` / `APP_DATABASE_URL` / `POSTGRES_URI` to `AEGRA_POSTGRES_URI` if `AEGRA_POSTGRES_URI` is not already set.

### Health checks for Railway

- Aegra endpoints: `/health`, `/ready`, `/live`
- Custom app endpoint: `/ok` (and `/v1/ok`)
- Recommended Railway health check target: `/health` or `/ok`

### Auth boundary

- Custom bearer-token middleware protects `/v1/*` app routes.
- Aegra protocol routes (`/threads`, `/runs`, etc.) are managed by Aegra; add Aegra-side auth (for example JWT mode) if required.

## 3) What Is Good About It

- Strong data integration: joins trials, labels, regulatory, molecule, target, and activity domains in one queryable system.
- Two-layer tool design: retrieval tools provide typed/direct DB access, while agent tools add LLM-friendly formatting, summarization, and tool-status streaming.
- Scientific search breadth: text/metadata trial search, safety/outcome comparison, cross-database ID resolution, target/pharmacology search, structure and substructure workflows.
- Production-aware runtime: Aegra-native deployment, Postgres-backed persistence, explicit health endpoints, and user/thread-scoped report persistence.
- Multiple consumption surfaces (API, CLI, web GUI) share one backend capability set.
- Test/eval assets exist for direct tools, agent mode behavior, web API smoke checks, and ground-truth based evaluation.

## 4) What Problems It Can Solve

- Find relevant clinical trials for a condition, intervention, molecule, or target.
- Connect molecules to trials (including similarity/substructure-driven discovery).
- Compare safety signals (adverse events) across drugs and studies.
- Compare efficacy/outcome evidence across trials.
- Retrieve and cross-check label/regulatory content (OpenFDA, DailyMed, Orange Book).
- Resolve identifiers and normalize entities across multiple biomedical databases.
- Generate reusable research reports from long tool outputs, then retrieve/report/manage them per thread.

## 5) How Users Can Interact With It

### Web Workbench (Next.js)

- Login-based GUI (`/login`) with thread/chat workflow.
- Streams run events/tokens from backend.
- Supports context cards, report browsing/loading, and file/text quick-add workflows.
- Uses backend proxy routes and user identity from NextAuth session.

### CLI Chat

- `biomedagent-db chat` for threaded conversations against Aegra backend.
- Built-in commands for thread management, context management, reports, and model selection.
- Convenience launcher: `./scripts/run_aegra_and_chat.sh`.

### API / Programmatic

- Aegra protocol routes for threads/runs/state.
- Custom v1 endpoints for health, identity, reports CRUD, and report content pagination.
- Suitable for custom clients and integration tests.

### Direct developer usage

- Ingestion and maintenance through `biomedagent-db` CLI.
- Tool behavior and retrieval logic exercised directly in `tests/` for reproducible development and evaluation.
