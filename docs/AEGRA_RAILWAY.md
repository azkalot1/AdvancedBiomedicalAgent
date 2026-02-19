# Aegra Migration (Railway)

This backend now runs on Aegra instead of the previously licensed runtime.

## What Changed

- Runtime image switched to Python + `arcsitegraph` (`aegra` command).
- Startup command now uses `scripts/run_aegra.sh`.
- Existing `langgraph.json` is reused (graph + custom FastAPI mount stay the same).
- `graph.py` skips internal checkpointer when `AEGRA_POSTGRES_URI` is set, so Aegra-managed persistence is used.

## Required Railway Env Vars

Set these in the backend service:

```bash
AEGRA_POSTGRES_URI=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
AEGRA_REDIS_URL=redis://default:PASSWORD@redis.railway.internal:6379
AEGRA_CONFIG_PATH=/app/langgraph.json
```

You can keep existing app vars too:

```bash
DATABASE_URL=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
APP_DATABASE_URL=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
POSTGRES_URI=postgresql://USER:PASSWORD@postgres.railway.internal:5432/DBNAME
```

`scripts/run_aegra.sh` automatically maps `DATABASE_URL`/`APP_DATABASE_URL`/`POSTGRES_URI` to `AEGRA_POSTGRES_URI` if needed.

## Health Checks

- Aegra native endpoints: `/health`, `/ready`, `/live`
- Custom app health endpoint: `/ok` (public)

For Railway health checks, use `/health` or `/ok`.

## Auth Notes

- Existing custom bearer middleware in `src/bioagent/server/webapp.py` still protects custom `/v1/*` routes.
- Aegra handles protocol routes (`/threads`, `/runs`, etc.). If you need auth on those routes, configure Aegra auth (for example JWT mode) in Railway env.

## Validation Checklist

1. `GET /health` returns `200`.
2. `GET /ok` returns `200`.
3. SDK client can create threads and run `co_scientist`.
4. Custom routes (`/v1/me`, `/v1/reports`, etc.) respond.
5. Auth middleware still rejects invalid/missing bearer tokens for protected `/v1/*` routes.
