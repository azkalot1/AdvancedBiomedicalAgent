# Benchmark Viewer

Local app for exploring benchmark run directories produced by `biomedagent-db benchmark run`.

## Run

```bash
cd benchmark_viewer
npm install
BENCHMARK_VIEWER_ROOT=/home/sergey/projects/AdvancedBiomedicalAgent npm run dev
```

Then open `http://localhost:3000`.

## What It Reads

The viewer scans `BENCHMARK_VIEWER_ROOT` recursively for benchmark run directories that contain:

- `manifest.json`
- `raw_runs.jsonl`
- `summary.json`
- `cases/*.json`

## Features

- select one or more runs
- compare aggregate metrics across models
- compare per-question outcomes by `case_id`
- inspect tool usage, message stacks, and raw thread state for a selected run/case
