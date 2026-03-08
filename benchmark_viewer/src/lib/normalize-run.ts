import type { ComparisonRow, LoadedRunData, RawRunRecord } from "@/src/lib/types";

export function buildComparisonRows(runs: LoadedRunData[]): ComparisonRow[] {
  const rowMap = new Map<string, ComparisonRow>();

  for (const run of runs) {
    const runId = run.manifest.run_id;
    for (const record of run.rawRuns) {
      const existing = rowMap.get(record.case_id);
      if (existing) {
        existing.runs[runId] = record;
        continue;
      }
      rowMap.set(record.case_id, {
        caseId: record.case_id,
        question: record.case.question,
        category: record.case.category,
        runs: Object.fromEntries(runs.map((item) => [item.manifest.run_id, null])) as Record<string, RawRunRecord | null>
      });
      rowMap.get(record.case_id)!.runs[runId] = record;
    }
  }

  return Array.from(rowMap.values()).sort((left, right) => left.caseId.localeCompare(right.caseId));
}

export function getSharedCaseCount(rows: ComparisonRow[], runIds: string[]): number {
  return rows.filter((row) => runIds.every((runId) => row.runs[runId] != null)).length;
}
