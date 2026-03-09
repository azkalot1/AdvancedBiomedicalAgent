import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import type { CaseDetailRecord, LoadedRunData, RawRunRecord, RunIndexRecord, RunManifest, RunSummaryPayload } from "@/src/lib/types";

const DEFAULT_SKIP_DIRS = new Set([".git", ".next", "node_modules", "web", "benchmark_viewer", ".cursor", ".venv"]);

function viewerRoot(): string {
  return process.env.BENCHMARK_VIEWER_ROOT || path.resolve(process.cwd(), "..");
}

async function discoverManifestPaths(rootDir: string, maxDepth = 7, currentDepth = 0): Promise<string[]> {
  const manifests: string[] = [];
  const entries = await readdir(rootDir, { withFileTypes: true });

  for (const entry of entries) {
    if (entry.isFile() && entry.name === "manifest.json") {
      manifests.push(path.join(rootDir, entry.name));
      continue;
    }
    if (!entry.isDirectory()) {
      continue;
    }
    if (DEFAULT_SKIP_DIRS.has(entry.name) || currentDepth >= maxDepth) {
      continue;
    }
    manifests.push(...(await discoverManifestPaths(path.join(rootDir, entry.name), maxDepth, currentDepth + 1)));
  }

  return manifests;
}

async function loadJsonFile<T>(filePath: string): Promise<T> {
  const raw = await readFile(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

function inferDatasetAndModel(relativeRunPath: string, manifest: RunManifest): { datasetKey: string; modelBucket: string } {
  const pathParts = relativeRunPath.split(path.sep).filter(Boolean);
  const runsIndex = pathParts.indexOf("runs");
  if (runsIndex >= 1) {
    return {
      datasetKey: pathParts[runsIndex - 1] || manifest.suite_name,
      modelBucket: pathParts[runsIndex + 1] || manifest.profile.name
    };
  }
  return {
    datasetKey: pathParts[0] || manifest.suite_name,
    modelBucket: pathParts[1] || manifest.profile.name
  };
}

export async function loadRunIndexes(): Promise<RunIndexRecord[]> {
  const rootDir = viewerRoot();
  const manifestPaths = await discoverManifestPaths(rootDir);
  const indexes: RunIndexRecord[] = [];

  for (const manifestPath of manifestPaths) {
    const manifest = await loadJsonFile<RunManifest>(manifestPath);
    const summaryPath = path.join(path.dirname(manifestPath), manifest.files.summary_json);
    const summary = await loadJsonFile<RunSummaryPayload>(summaryPath);
    const relativeRunPath = path.relative(rootDir, path.dirname(manifestPath));
    const inferred = inferDatasetAndModel(relativeRunPath, manifest);
    indexes.push({
      manifest,
      summary,
      relativeRunPath,
      datasetKey: inferred.datasetKey,
      modelBucket: inferred.modelBucket
    });
  }

  indexes.sort((left, right) => right.manifest.run_started_at.localeCompare(left.manifest.run_started_at));
  return indexes;
}

export async function loadRunData(runId: string): Promise<LoadedRunData> {
  const indexes = await loadRunIndexes();
  const match = indexes.find((item) => item.manifest.run_id === runId);
  if (!match) {
    throw new Error(`Unknown run_id '${runId}'.`);
  }

  const runDir = path.dirname(path.join(match.manifest.output_dir, match.manifest.files.manifest));
  const rawRunsPath = path.join(runDir, match.manifest.files.raw_runs);
  const rawRunsText = await readFile(rawRunsPath, "utf-8");
  const rawRuns = rawRunsText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RawRunRecord);

  return {
    manifest: match.manifest,
    summary: match.summary,
    rawRuns
  };
}

export async function loadCaseDetail(runId: string, caseId: string): Promise<CaseDetailRecord> {
  const runData = await loadRunData(runId);
  const record = runData.rawRuns.find((item) => item.case_id === caseId);
  if (!record || !record.detail_path) {
    throw new Error(`Could not find detail file for case '${caseId}' in run '${runId}'.`);
  }

  const runDir = path.dirname(path.join(runData.manifest.output_dir, runData.manifest.files.manifest));
  const detailPath = path.join(runDir, record.detail_path);
  return loadJsonFile<CaseDetailRecord>(detailPath);
}
