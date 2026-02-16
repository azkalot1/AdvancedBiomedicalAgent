# Chat Interface Guide

This project includes a CLI chat client backed by LangGraph Server.

The chat interface is intended to be the backend interaction model for the GUI:
- threaded conversations
- streaming assistant tokens
- tool status events
- report listing and report content loading
- context injection for the next requests

## Prerequisites

From repo root:

```bash
pip install -e .
```

You also need `langgraph` available in your active environment.

## Start Chat (One Command)

Use the helper script:

```bash
./scripts/start_chat_stack.sh
```

What it does:
1. Starts `langgraph dev`
2. Waits for server health (`/v1/ok`, fallback `/ok`)
3. Starts `biomedagent-db chat`
4. Stops server when chat exits

Server logs default to:

```bash
.langgraph_api/dev.log
```

## Start Chat (Manual, Two Terminals)

Terminal 1:

```bash
langgraph dev
```

Terminal 2:

```bash
biomedagent-db chat --server-url http://localhost:2024 --assistant-id co_scientist
```

## Auth

If API auth is enabled, set:

```bash
export BIOAGENT_API_TOKEN=dev-token
export BIOAGENT_API_USER_ID=1
export BIOAGENT_AUTH_REQUIRED=true
```

Then run:

```bash
./scripts/start_chat_stack.sh --api-token "$BIOAGENT_API_TOKEN"
```

or:

```bash
biomedagent-db chat --server-url http://localhost:2024 --assistant-id co_scientist --api-token "$BIOAGENT_API_TOKEN"
```

## Chat Commands

In chat prompt (`You>`), supported commands:

- `/new`: create and switch to a new thread
- `/threads`: list threads
- `/switch <thread_id>`: switch active thread
- `/messages [full]`: print current thread message state (useful for debugging)
- `/reports`: list reports for current thread
- `/load_report <report_id>`: print report content
- `/delete-report <report_id>`: delete a report
- `/add_to_context <text>`: add manual context
- `/context`: list current context items
- `/clear_context`: clear context items
- `/help`: show command help
- `/quit`: exit

## How Context Works

`/add_to_context` items are kept in CLI memory for the current chat process.

On each user send:
1. Context is embedded into the outgoing human message text as:
   - `Additional context for this request: ...`
2. Context is also sent as `input.context_items` metadata.

Important behavior:
- context items are not persisted as standalone server objects
- embedded context text is persisted as part of the thread messages
- context list is local to the CLI process and can be reset with `/clear_context`

## GUI-Relevant API Surface

LangGraph built-ins (used by CLI and GUI):
- `POST /threads`
- `GET /threads/{id}/state`
- `POST /threads/{id}/runs/stream`

Custom versioned endpoints:
- `GET /v1/ok`
- `GET /v1/me`
- `GET /v1/reports?thread_id=...&limit=...&offset=...`
- `GET /v1/reports/{report_id}`
- `GET /v1/reports/{report_id}/content?offset=...&max_chars=...`
- `DELETE /v1/reports/{report_id}`

Streaming (`runs/stream`) uses:
- token events (`messages`, `messages-tuple`)
- custom events (`tool_status`, `report_generated`, etc.)

## Smoke Test

Use the API smoke test script:

```bash
./scripts/gui_api_smoke_test.py --base-url http://localhost:2024 --assistant-id co_scientist --api-token "$BIOAGENT_API_TOKEN"
```

This validates:
- health
- identity (`/v1/me`)
- thread creation/list
- streaming run
- state retrieval
- reports metadata and content APIs

## Troubleshooting

- If `langgraph dev` fails to load app, check:
  - `langgraph.json` points to `./src/bioagent/server/webapp.py:app`
  - environment has project installed (`pip install -e .`)
- If chat cannot connect, verify:
  - server is running on `LANGGRAPH_API_URL` (default `http://localhost:2024`)
- If auth errors (`401`):
  - pass `--api-token` and confirm token env configuration
