#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_URL="${LANGGRAPH_API_URL:-http://localhost:2024}"
ASSISTANT_ID="${BIOAGENT_ASSISTANT_ID:-co_scientist}"
USER_ID="${BIOAGENT_USER_ID:-}"
API_TOKEN="${BIOAGENT_API_TOKEN:-}"
WAIT_SECONDS="${BIOAGENT_SERVER_WAIT_SECONDS:-90}"
LOG_PATH="${BIOAGENT_LANGGRAPH_LOG:-$ROOT_DIR/.langgraph_api/dev.log}"

EXTRA_CHAT_ARGS=()

usage() {
  cat <<EOF
Start LangGraph dev server + CLI chat in one command.

Usage:
  $(basename "$0") [options] [-- <extra chat args>]

Options:
  --server-url URL       LangGraph server URL (default: ${SERVER_URL})
  --assistant-id ID      Assistant/graph id (default: ${ASSISTANT_ID})
  --user-id ID           Optional fallback user id for chat client
  --api-token TOKEN      Bearer token for /v1 auth
  --wait-seconds N       Time to wait for server readiness (default: ${WAIT_SECONDS})
  --log-path PATH        Server log file path (default: ${LOG_PATH})
  -h, --help             Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --api-token dev-token
  $(basename "$0") --assistant-id co_scientist -- --stream-tool-args
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-url)
      SERVER_URL="${2:-}"
      shift 2
      ;;
    --assistant-id)
      ASSISTANT_ID="${2:-}"
      shift 2
      ;;
    --user-id)
      USER_ID="${2:-}"
      shift 2
      ;;
    --api-token)
      API_TOKEN="${2:-}"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="${2:-}"
      shift 2
      ;;
    --log-path)
      LOG_PATH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CHAT_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_CHAT_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v langgraph >/dev/null 2>&1; then
  echo "Error: 'langgraph' command not found."
  echo "Install in your active env: pip install -U langgraph-api"
  exit 1
fi

if ! command -v biomedagent-db >/dev/null 2>&1; then
  echo "Error: 'biomedagent-db' command not found."
  echo "Install project in your active env: pip install -e ."
  exit 1
fi

cleanup() {
  local code=$?
  if [[ -n "${LG_PID:-}" ]] && kill -0 "${LG_PID}" >/dev/null 2>&1; then
    kill "${LG_PID}" >/dev/null 2>&1 || true
    wait "${LG_PID}" >/dev/null 2>&1 || true
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "${LOG_PATH}")"

cd "${ROOT_DIR}"
echo "Starting langgraph dev..."
langgraph dev >"${LOG_PATH}" 2>&1 &
LG_PID=$!

echo "Waiting for server at ${SERVER_URL} (timeout ${WAIT_SECONDS}s)..."
started=0
for ((i=0; i<WAIT_SECONDS; i++)); do
  if curl -fsS "${SERVER_URL}/v1/ok" >/dev/null 2>&1 || curl -fsS "${SERVER_URL}/ok" >/dev/null 2>&1; then
    started=1
    break
  fi
  if ! kill -0 "${LG_PID}" >/dev/null 2>&1; then
    echo "LangGraph server process exited before becoming ready."
    echo "Last server log lines:"
    tail -n 40 "${LOG_PATH}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${started}" -ne 1 ]]; then
  echo "Timed out waiting for server readiness."
  echo "Last server log lines:"
  tail -n 40 "${LOG_PATH}" || true
  exit 1
fi

echo "Server ready. Launching chat..."
CHAT_CMD=(biomedagent-db chat --server-url "${SERVER_URL}" --assistant-id "${ASSISTANT_ID}")

if [[ -n "${USER_ID}" ]]; then
  CHAT_CMD+=(--user-id "${USER_ID}")
fi
if [[ -n "${API_TOKEN}" ]]; then
  CHAT_CMD+=(--api-token "${API_TOKEN}")
fi

if [[ "${#EXTRA_CHAT_ARGS[@]}" -gt 0 ]]; then
  CHAT_CMD+=("${EXTRA_CHAT_ARGS[@]}")
fi

"${CHAT_CMD[@]}"
