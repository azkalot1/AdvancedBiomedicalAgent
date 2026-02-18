# Advanced Biomedical Agent

Biomedical research agent that integrates OpenFDA, DailyMed, ClinicalTrials.gov, ChEMBL, BindingDB, DrugCentral, Orange Book and related sources into a unified PostgreSQL database, with LLM-friendly search tools and a **chat interface**.

## Quick Start

**Prerequisites:** PostgreSQL 14+, Python 3.12+, ~50 GB disk, 8+ GB RAM for full ingestion.

### 1. Install PostgreSQL

- **Ubuntu/Debian:** `sudo apt-get install postgresql postgresql-contrib`
- **macOS:** `brew install postgresql`
- **Windows:** [Download from postgresql.org](https://www.postgresql.org/download/)

### 2. After that

```bash
cp .env.example .env
pip install -e .
biomedagent-db verify-deps
biomedagent-db setup_postgres
biomedagent-db ingest
```

**Data available.** The database unifies clinical trials (ClinicalTrials.gov), molecular structures and biotherapeutics (ChEMBL, DrugCentral, BindingDB), biological targets, drug labels (OpenFDA, DailyMed), and regulatory data (Orange Book), with mappings from trials and products to molecules. Full schema: [docs/DATABASE_SCHEMA.md](docs/DATABASE_SCHEMA.md).

**Tools available.** Retrieval tools cover clinical trial search, drug labels, molecule–trial connectivity, adverse events, outcomes, Orange Book, cross-database lookup, biotherapeutic sequence search, and target/drug pharmacology; agent tools wrap these with LLM-friendly formatting. Full reference: [docs/tools.md](docs/tools.md).

For pipeline order and script roles, see [docs/INGESTION.md](docs/INGESTION.md).

## Local Reproduction (Start to Working GUI)

This is the fastest reproducible path on a single local Postgres instance/port (`5432`).

1. Prepare env and install dependencies:
```bash
cp .env.example .env
pip install -e .
```

2. Set correct DB credentials in `.env` for your local Postgres user:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=coscientist_data
DB_USER=<your_db_user>
DB_PASSWORD=<your_db_password>

DATA_DATABASE_URL=postgresql://<your_db_user>:<your_db_password>@localhost:5432/coscientist_data
APP_DATABASE_URL=postgresql://<your_db_user>:<your_db_password>@localhost:5432/coscientist_app
DATABASE_URL=postgresql://<your_db_user>:<your_db_password>@localhost:5432/coscientist_app
POSTGRES_URI=postgresql://<your_db_user>:<your_db_password>@localhost:5432/coscientist_app

NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=replace_with_long_random_secret
BIOAGENT_BACKEND_URL=http://localhost:2024
```

3. Setup data database and optional quick ingest:
```bash
make verify-deps
make setup-postgres
make ingest-quick
```

4. Setup app users/auth DB:
```bash
make users-setup
make users-list
```

5. Start GUI stack (backend + web, persistent runtime):
```bash
make gui-stack-up
```

6. Open:
- `http://localhost:3000/login`
- sign in with seeded credentials from `credentials.txt`

7. (Optional) Start CLI chat instead:
```bash
make chat-stack
```

## Quick Prototype Ingestion

For a lightweight prototype dataset (fastest path), run:

```bash
biomedagent-db ingest --quick-prototype
```

This profile ingests OpenFDA + Orange Book only, skips heavy sources (CT.gov, DailyMed, BindingDB, ChEMBL, DrugCentral, dm_target, dm_molecule), and applies defaults:
- `--openfda-files 2`
- `--n-max 2000`

You can still override these limits explicitly.

## Makefile Shortcuts

Common commands:

```bash
make install
make verify-deps
make setup-postgres
make ingest
make ingest-quick
make langgraph-dev
make chat
make chat-stack
make gui-stack
make web-install
make web-dev
make web-check
make users-setup
make users-list
```

Run `make help` to list all targets.

## Auth + User Management (Local)

Two-DB layout:

- `coscientist_data` (`5432`): scientific source tables used by tools/ingestion
- `coscientist_app` (`5432`): `app_users` + LangGraph checkpoint/store data

Use your existing local PostgreSQL service (same instance/port, separate databases).
Ensure Postgres is running before setup:

- Ubuntu/Debian: `sudo systemctl start postgresql`
- macOS (Homebrew): `brew services start postgresql`

Initialize and seed users:

```bash
make users-setup
make users-list
```

Additional user commands:

```bash
make users-db
make users-add EMAIL=dr.new@lab.org NAME="Dr. New" ROLE=user
make users-reset-pw EMAIL=dr.new@lab.org
make users-deactivate EMAIL=dr.new@lab.org
make users-activate EMAIL=dr.new@lab.org
make users-remove EMAIL=dr.new@lab.org
```

Generated credentials are written to `credentials.txt` (gitignored). Share securely and delete after use.
This project uses two databases on the same Postgres instance/port (for example `5432`), configured via env vars:
Example:
```bash
APP_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coscientist_app
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coscientist_app
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/coscientist_app
```

## Chat Server + CLI

Full guide: [docs/CHAT_INTERFACE.md](docs/CHAT_INTERFACE.md)

You can run the LangGraph server and CLI chat together with one script:

```bash
./scripts/run_langgraph_and_chat.sh
```

This script:
- starts `langgraph dev` in the project root
- waits for `/v1/ok` (or `/ok`) readiness
- launches `biomedagent-db chat`
- stops the dev server when chat exits

Server logs are written to `.langgraph_api/dev.log` by default.

### Auth-enabled example

```bash
export BIOAGENT_API_TOKEN=dev-token
export BIOAGENT_API_USER_ID=1
export BIOAGENT_AUTH_REQUIRED=true

./scripts/run_langgraph_and_chat.sh --api-token "$BIOAGENT_API_TOKEN"
```

### Useful options

```bash
./scripts/run_langgraph_and_chat.sh --help
./scripts/run_langgraph_and_chat.sh --assistant-id co_scientist
./scripts/run_langgraph_and_chat.sh --server-url http://localhost:2024
./scripts/run_langgraph_and_chat.sh --api-token dev-token -- --stream-tool-args
```

### Manual (two terminals)

Terminal 1:
```bash
langgraph dev
```

Terminal 2:
```bash
biomedagent-db chat --server-url http://localhost:2024 --assistant-id co_scientist --api-token dev-token
```

## GUI Workbench (Local)

Full web details: [web/README.md](web/README.md)

### One Command

```bash
make gui-stack
```

This starts `langgraph dev` waits for backend health, then runs the Next.js app.

### Manual (Two Terminals)

Terminal 1:
```bash
make langgraph-up
```

Terminal 2:
```bash
cd web
npm install
npm run dev
```

Then open:
- `http://localhost:3000/login`

Required env for GUI auth/backend wiring (in `.env`):

```bash
APP_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coscientist_app
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coscientist_app
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/coscientist_app
BIOAGENT_RESEARCH_OUTPUT_DIR=./research_outputs
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=replace_with_long_random_secret
BIOAGENT_BACKEND_URL=http://localhost:2024
```

Report files are stored under `BIOAGENT_RESEARCH_OUTPUT_DIR` (default `./research_outputs`), scoped by user/thread.
`langgraph dev` still uses in-memory runtime for thread/run registry; use `langgraph up` for persistent `/threads` behavior across logins/restarts.

Seed login users first:

```bash
make users-setup
make users-list
```

## Troubleshooting

- `Missing DATABASE_URL (or APP_DATABASE_URL) for NextAuth`:
Set `DATABASE_URL` in shell or `web/.env.local` when running `cd web && npm run dev`.

- `SASL: SCRAM-SERVER-FIRST-MESSAGE: client password must be a string`:
`DATABASE_URL` is malformed or missing password. Use `postgresql://user:password@host:port/dbname`.

- `password authentication failed for user ...`:
`APP_DATABASE_URL`/`DATABASE_URL` credentials do not match your Postgres server.

- `permission denied to create database` during `make users-setup`:
The current DB role lacks `CREATEDB`; either set `APP_DATABASE_ADMIN_URL` or use sudo/admin once.

- `permission denied for schema public` during `make users-setup`:
Grant schema privileges once using admin user; `manage_users.py` can auto-fix via `APP_DATABASE_ADMIN_URL` or sudo fallback.
