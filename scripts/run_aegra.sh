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

if [[ -z "${AEGRA_POSTGRES_URI:-}" ]]; then
  echo "Error: set AEGRA_POSTGRES_URI (or DATABASE_URL/APP_DATABASE_URL/POSTGRES_URI)." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- "${AEGRA_DEFAULT_COMMAND:-dev}"
fi

exec aegra "$@"
