"use client";

import type { RunIndexRecord } from "@/src/lib/types";

interface RunSelectorProps {
  runs: RunIndexRecord[];
  selectedRunIds: string[];
  onToggle(runId: string): void;
  datasetFilter: string;
  onDatasetFilterChange(value: string): void;
}

export function RunSelector({
  runs,
  selectedRunIds,
  onToggle,
  datasetFilter,
  onDatasetFilterChange
}: RunSelectorProps): React.ReactElement {
  const datasetOptions = Array.from(new Set(runs.map((run) => run.datasetKey))).sort();

  return (
    <section style={sectionStyle}>
      <div style={headerRowStyle}>
        <h2 style={sectionTitleStyle}>Available Runs</h2>
        <select value={datasetFilter} onChange={(event) => onDatasetFilterChange(event.target.value)} style={selectStyle}>
          <option value="all">All datasets</option>
          {datasetOptions.map((datasetKey) => (
            <option key={datasetKey} value={datasetKey}>
              {datasetKey}
            </option>
          ))}
        </select>
      </div>
      <div style={{ display: "grid", gap: 10 }}>
        {runs.length === 0 ? (
          <div style={mutedStyle}>No run manifests found under the configured benchmark viewer root.</div>
        ) : (
          runs.map((run) => {
            const checked = selectedRunIds.includes(run.manifest.run_id);
            return (
              <label key={run.manifest.run_id} style={cardStyle}>
                <input type="checkbox" checked={checked} onChange={() => onToggle(run.manifest.run_id)} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600 }}>
                    {run.manifest.profile.model.model_name}{" "}
                    <span style={chipStyle}>{run.manifest.profile.model.provider}</span>
                  </div>
                  <div style={mutedStyle}>
                    run_id: {run.manifest.run_id} | suite: {run.manifest.suite_name}
                  </div>
                  <div style={mutedStyle}>
                    dataset: {run.datasetKey} | bucket: {run.modelBucket}
                  </div>
                  <div style={mutedStyle}>
                    label: {run.manifest.run_label || "n/a"} | started: {new Date(run.manifest.run_started_at).toLocaleString()}
                  </div>
                  <div style={mutedStyle}>
                    accuracy: {(run.summary.summary.accuracy * 100).toFixed(1)}% | total: {run.summary.summary.total}
                  </div>
                  <div style={mutedStyle}>path: {run.relativeRunPath}</div>
                </div>
              </label>
            );
          })
        )}
      </div>
    </section>
  );
}

const sectionStyle: React.CSSProperties = {
  border: "1px solid rgba(148, 163, 184, 0.2)",
  borderRadius: 14,
  padding: 16,
  background: "rgba(15, 23, 42, 0.75)"
};

const sectionTitleStyle: React.CSSProperties = {
  marginTop: 0,
  marginBottom: 12,
  fontSize: 18
};

const headerRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 12,
  marginBottom: 12
};

const cardStyle: React.CSSProperties = {
  display: "flex",
  gap: 12,
  alignItems: "flex-start",
  border: "1px solid rgba(148, 163, 184, 0.15)",
  borderRadius: 12,
  padding: 12,
  background: "rgba(30, 41, 59, 0.45)"
};

const mutedStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 13,
  marginTop: 4
};

const chipStyle: React.CSSProperties = {
  display: "inline-block",
  marginLeft: 8,
  padding: "2px 8px",
  borderRadius: 999,
  background: "rgba(59, 130, 246, 0.15)",
  color: "#93c5fd",
  fontSize: 12
};

const selectStyle: React.CSSProperties = {
  borderRadius: 8,
  background: "#0f172a",
  color: "#e5edf9",
  border: "1px solid rgba(148, 163, 184, 0.2)",
  padding: "8px 10px",
  minWidth: 180
};
