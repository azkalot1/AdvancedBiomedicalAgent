#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Auto-load root .env for local development convenience.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -z "${AEGRA_CONFIG_PATH:-}" ]]; then
  if [[ -f "${ROOT_DIR}/langgraph.json" ]]; then
    export AEGRA_CONFIG_PATH="${ROOT_DIR}/langgraph.json"
  elif [[ -f "${ROOT_DIR}/aegra.json" ]]; then
    export AEGRA_CONFIG_PATH="${ROOT_DIR}/aegra.json"
  fi
fi

if [[ -z "${AEGRA_POSTGRES_URI:-}" ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    export AEGRA_POSTGRES_URI="${DATABASE_URL}"
  elif [[ -n "${APP_DATABASE_URL:-}" ]]; then
    export AEGRA_POSTGRES_URI="${APP_DATABASE_URL}"
  elif [[ -n "${POSTGRES_URI:-}" ]]; then
    export AEGRA_POSTGRES_URI="${POSTGRES_URI}"
  fi
fi

if [[ -z "${AEGRA_REDIS_URL:-}" && -n "${REDIS_URL:-}" ]]; then
  export AEGRA_REDIS_URL="${REDIS_URL}"
fi

# Quick dev mode: no Postgres/Docker required; uses in-memory checkpointer
QUICK_DEV=
if [[ "${1:-}" == "dev-quick" ]]; then
  QUICK_DEV=1
  shift
  set -- "dev" "$@"
fi
if [[ -n "${AEGRA_NO_DB:-}" ]]; then
  QUICK_DEV=1
fi

if [[ -z "${QUICK_DEV}" && -z "${AEGRA_POSTGRES_URI:-}" ]]; then
  echo "Error: set AEGRA_POSTGRES_URI (or DATABASE_URL/APP_DATABASE_URL/POSTGRES_URI)." >&2
  echo "For quick local dev without Postgres, use: make aegra-dev-quick" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- "${AEGRA_DEFAULT_COMMAND:-dev}"
fi

# In quick dev, skip Aegra's Postgres check and use in-memory checkpointer in the graph
if [[ -n "${QUICK_DEV}" ]]; then
  export BIOAGENT_CHECKPOINT_DB="${BIOAGENT_CHECKPOINT_DB:-:memory:}"
  exec aegra "$@" --no-db-check --host 0.0.0.0 --port "${PORT:-8000}"
fi

exec aegra "$@" --host 0.0.0.0 --port "${PORT:-8000}"
