# AdvancedBiomedicalAgent Web Workbench

Next.js 14 App Router frontend for the 3-panel AI Co-Scientist GUI.

## Env

Set these before running:

```bash
export BIOAGENT_BACKEND_URL=http://localhost:2024
export BIOAGENT_API_TOKEN=dev-token        # optional unless backend auth is enabled
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coscientist_app
export NEXTAUTH_URL=http://localhost:3000
export NEXTAUTH_SECRET=replace_with_long_random_secret
export NEXT_PUBLIC_BIOAGENT_ASSISTANT_ID=co_scientist
```

`DATABASE_URL` must include a password value in the URL (`postgresql://user:password@...`), even for local dev.
If you run `cd web && npm run dev`, define these in your shell or `web/.env.local` (repo-root `.env` is not auto-loaded by Next in that mode).

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000/login` and sign in with a seeded user.

From repo root you can also run the full GUI stack (backend + web):

```bash
make gui-stack-up
```

Use `make gui-stack` only for ephemeral dev runtime.

Recommended flow from repo root:

```bash
make users-setup
make users-list
make gui-stack-up
```

The UI proxies all backend calls through `app/api/backend/[...path]/route.ts` and injects `x-bioagent-user-id` from the authenticated NextAuth session.

## Backend APIs Used

- `GET /v1/reports`
- `GET /v1/reports/{id}/content`
- `POST /threads`
- `POST /threads/{id}/runs/stream` (SSE)
- `GET /threads/{id}/state`

## Notes

- Streaming parser supports token events and custom events (`tool_status`, `report_generated`, etc.)
- Context items are serialized into both embedded prompt text and `input.context_items`, matching CLI semantics
- Report selection supports selection-to-context via floating action menu
- Quick Add supports:
  - `Paste Text` (adds manual context cards)
  - `Upload File` for `PDF`, `PNG`, `JPG/JPEG` (sent as multimodal message blocks)

## Troubleshooting

- `Missing DATABASE_URL (or APP_DATABASE_URL) for NextAuth`:
Define `DATABASE_URL` in `web/.env.local` or export it in shell before `npm run dev`.

- `SASL: SCRAM-SERVER-FIRST-MESSAGE: client password must be a string`:
`DATABASE_URL` has no password or wrong format. Use `postgresql://user:password@host:port/dbname`.

- `password authentication failed for user ...`:
Credentials in `DATABASE_URL` are wrong for your local Postgres.
