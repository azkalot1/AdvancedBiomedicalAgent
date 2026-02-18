#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_URL="${LANGGRAPH_API_URL:-http://localhost:2024}"
WAIT_SECONDS="${BIOAGENT_SERVER_WAIT_SECONDS:-90}"
LOG_PATH="${BIOAGENT_LANGGRAPH_LOG:-$ROOT_DIR/.langgraph_api/dev.log}"
WEB_DIR="${BIOAGENT_WEB_DIR:-$ROOT_DIR/web}"
WEB_PORT="${PORT:-3000}"

usage() {
  cat <<USAGE
Start LangGraph dev server + Web GUI in one command.

Usage:
  $(basename "$0") [options] [-- <extra npm args>]

Options:
  --server-url URL       LangGraph server URL (default: ${SERVER_URL})
  --wait-seconds N       Wait timeout for backend readiness (default: ${WAIT_SECONDS})
  --log-path PATH        LangGraph log path (default: ${LOG_PATH})
  --web-dir PATH         Web app directory (default: ${WEB_DIR})
  --web-port PORT        Next.js port (default: ${WEB_PORT})
  -h, --help             Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --web-port 3001
USAGE
}

EXTRA_WEB_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-url)
      SERVER_URL="${2:-}"
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
    --web-dir)
      WEB_DIR="${2:-}"
      shift 2
      ;;
    --web-port)
      WEB_PORT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_WEB_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_WEB_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v langgraph >/dev/null 2>&1; then
  echo "Error: 'langgraph' command not found."
  echo "Install in your active env: pip install -U langgraph-api"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: 'npm' command not found."
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

if [[ ! -d "${WEB_DIR}" ]]; then
  echo "Error: web directory not found: ${WEB_DIR}"
  exit 1
fi

cd "${WEB_DIR}"
export BIOAGENT_BACKEND_URL="${SERVER_URL}"
export NEXTAUTH_URL="${NEXTAUTH_URL:-http://localhost:${WEB_PORT}}"
export PORT="${WEB_PORT}"

echo "Server ready. Launching web UI at http://localhost:${WEB_PORT} ..."
WEB_CMD=(npm run dev)
if [[ "${#EXTRA_WEB_ARGS[@]}" -gt 0 ]]; then
  WEB_CMD+=(-- "${EXTRA_WEB_ARGS[@]}")
fi

"${WEB_CMD[@]}"
