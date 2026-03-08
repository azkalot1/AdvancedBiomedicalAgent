"use client";

import { useEffect, useMemo, useState } from "react";

import { MetricsDashboard } from "@/src/components/metrics-dashboard";
import { QuestionCompareTable } from "@/src/components/question-compare-table";
import { RunDetailInspector } from "@/src/components/run-detail-inspector";
import { RunSelector } from "@/src/components/run-selector";
import { buildComparisonRows, getSharedCaseCount } from "@/src/lib/normalize-run";
import type { CaseDetailRecord, LoadedRunData, RunIndexRecord } from "@/src/lib/types";

interface ViewerShellProps {
  initialRuns: RunIndexRecord[];
}

export function ViewerShell({ initialRuns }: ViewerShellProps): React.ReactElement {
  const [datasetFilter, setDatasetFilter] = useState<string>("all");
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [loadedRuns, setLoadedRuns] = useState<Record<string, LoadedRunData>>({});
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [detailCache, setDetailCache] = useState<Record<string, CaseDetailRecord>>({});
  const [detailLoading, setDetailLoading] = useState(false);

  const visibleRuns = useMemo(
    () => initialRuns.filter((run) => datasetFilter === "all" || run.datasetKey === datasetFilter),
    [initialRuns, datasetFilter]
  );
  const visibleRunIdSet = useMemo(() => new Set(visibleRuns.map((run) => run.manifest.run_id)), [visibleRuns]);

  useEffect(() => {
    setSelectedRunIds((current) => current.filter((runId) => visibleRunIdSet.has(runId)));
  }, [visibleRunIdSet]);

  useEffect(() => {
    if (selectedRunIds.length > 0 && (!activeRunId || !selectedRunIds.includes(activeRunId))) {
      setActiveRunId(selectedRunIds[0]);
    }
    if (selectedRunIds.length === 0 && activeRunId) {
      setActiveRunId(null);
    }
  }, [selectedRunIds, activeRunId]);

  useEffect(() => {
    for (const runId of selectedRunIds) {
      if (loadedRuns[runId]) {
        continue;
      }
      void fetch(`/api/runs/${runId}`)
        .then((response) => response.json())
        .then((payload: LoadedRunData) => {
          setLoadedRuns((current) => ({ ...current, [runId]: payload }));
        });
    }
  }, [selectedRunIds, loadedRuns]);

  useEffect(() => {
    if (!selectedCaseId || !activeRunId) {
      return;
    }
    const key = `${activeRunId}::${selectedCaseId}`;
    if (detailCache[key]) {
      return;
    }
    setDetailLoading(true);
    void fetch(`/api/runs/${activeRunId}/cases/${selectedCaseId}`)
      .then((response) => response.json())
      .then((payload: CaseDetailRecord) => {
        setDetailCache((current) => ({ ...current, [key]: payload }));
      })
      .finally(() => setDetailLoading(false));
  }, [selectedCaseId, activeRunId, detailCache]);

  const selectedRuns = useMemo(
    () => selectedRunIds.map((runId) => loadedRuns[runId]).filter((value): value is LoadedRunData => Boolean(value)),
    [selectedRunIds, loadedRuns]
  );
  const rows = useMemo(() => buildComparisonRows(selectedRuns), [selectedRuns]);
  const sharedCaseCount = useMemo(
    () => getSharedCaseCount(rows, selectedRuns.map((run) => run.manifest.run_id)),
    [rows, selectedRuns]
  );
  const activeDetail = activeRunId && selectedCaseId ? detailCache[`${activeRunId}::${selectedCaseId}`] ?? null : null;

  return (
    <main style={pageStyle}>
      <div style={heroStyle}>
        <h1 style={{ margin: 0 }}>Benchmark Viewer</h1>
        <p style={heroSubtitleStyle}>
          Explore local benchmark runs, compare models side by side, inspect tool traces, and drill into the full
          message/state stack for each question.
        </p>
        <div style={metaStyle}>
          discovered runs: {initialRuns.length} | visible runs: {visibleRuns.length} | selected runs: {selectedRuns.length} | shared cases: {sharedCaseCount}
        </div>
      </div>

      <div style={gridStyle}>
        <RunSelector
          runs={visibleRuns}
          selectedRunIds={selectedRunIds}
          datasetFilter={datasetFilter}
          onDatasetFilterChange={setDatasetFilter}
          onToggle={(runId) =>
            setSelectedRunIds((current) =>
              current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId]
            )
          }
        />
        <MetricsDashboard runs={selectedRuns} />
      </div>

      <div style={{ marginTop: 20 }}>
        <QuestionCompareTable rows={rows} runs={selectedRuns} selectedCaseId={selectedCaseId} onSelectCase={setSelectedCaseId} />
      </div>

      <div style={{ marginTop: 20 }}>
        <RunDetailInspector
          runs={selectedRuns}
          selectedCaseId={selectedCaseId}
          activeRunId={activeRunId}
          onChangeActiveRun={setActiveRunId}
          detail={activeDetail}
          detailLoading={detailLoading}
        />
      </div>
    </main>
  );
}

const pageStyle: React.CSSProperties = {
  maxWidth: 1560,
  margin: "0 auto",
  padding: 24
};

const heroStyle: React.CSSProperties = {
  marginBottom: 20
};

const heroSubtitleStyle: React.CSSProperties = {
  color: "#94a3b8",
  maxWidth: 860,
  lineHeight: 1.5
};

const metaStyle: React.CSSProperties = {
  color: "#93c5fd",
  fontSize: 13,
  marginTop: 8
};

const gridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(360px, 460px) 1fr",
  gap: 20,
  alignItems: "start"
};
