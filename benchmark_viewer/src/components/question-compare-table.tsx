"use client";

import { useMemo, useState } from "react";

import { formatLoadedRunDisplayName, formatRunSecondaryLabel } from "@/src/lib/run-labels";
import type { ComparisonRow, LoadedRunData } from "@/src/lib/types";

interface QuestionCompareTableProps {
  rows: ComparisonRow[];
  runs: LoadedRunData[];
  selectedCaseId: string | null;
  onSelectCase(caseId: string): void;
}

export function QuestionCompareTable({
  rows,
  runs,
  selectedCaseId,
  onSelectCase
}: QuestionCompareTableProps): React.ReactElement {
  const [statusFilters, setStatusFilters] = useState<string[]>([]);
  const availableStatuses = useMemo(
    () => ["correct", "incorrect", "invalid_format", "passed", "failed", "error", "missing"],
    []
  );
  const filteredRows = useMemo(() => {
    if (statusFilters.length === 0) {
      return rows;
    }
    return rows.filter((row) =>
      runs.some((run) => {
        const record = row.runs[run.manifest.run_id];
        const status = record?.answer_status ?? "missing";
        return statusFilters.includes(status);
      })
    );
  }, [rows, runs, statusFilters]);

  return (
    <section style={sectionStyle}>
      <div style={headerRowStyle}>
        <h2 style={sectionTitleStyle}>Per-Question Comparison</h2>
        <div style={filterBarStyle}>
          {availableStatuses.map((status) => {
            const active = statusFilters.includes(status);
            return (
              <button
                key={status}
                type="button"
                onClick={() =>
                  setStatusFilters((current) =>
                    current.includes(status) ? current.filter((item) => item !== status) : [...current, status]
                  )
                }
                style={{
                  ...filterChipStyle,
                  ...(active ? filterChipActiveStyle : null)
                }}
              >
                {status}
              </button>
            );
          })}
          {statusFilters.length > 0 ? (
            <button type="button" onClick={() => setStatusFilters([])} style={clearButtonStyle}>
              clear
            </button>
          ) : null}
        </div>
      </div>
      {runs.length === 0 ? (
        <div style={mutedStyle}>Select runs to compare their question-by-question outcomes.</div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <div style={filterSummaryStyle}>
            showing {filteredRows.length} of {rows.length} cases
            {statusFilters.length > 0 ? ` | filter: ${statusFilters.join(", ")}` : ""}
          </div>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Case</th>
                <th style={thStyle}>Question</th>
                {runs.map((run) => (
                  <th key={run.manifest.run_id} style={thStyle}>
                    <div>{formatLoadedRunDisplayName(run)}</div>
                    <div style={headerSubLabelStyle}>{formatRunSecondaryLabel(run.manifest)}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => (
                <tr
                  key={row.caseId}
                  onClick={() => onSelectCase(row.caseId)}
                  style={{
                    cursor: "pointer",
                    background: row.caseId === selectedCaseId ? "rgba(59, 130, 246, 0.14)" : undefined
                  }}
                >
                  <td style={tdLabelStyle}>{row.caseId}</td>
                  <td style={tdStyle}>
                    <div>{row.question}</div>
                    <div style={mutedStyle}>
                      {(row.category || "uncategorized") +
                        (Object.values(row.runs).find((record) => record)?.case.type ? ` | ${Object.values(row.runs).find((record) => record)?.case.type}` : "")}
                    </div>
                  </td>
                  {runs.map((run) => {
                    const record = row.runs[run.manifest.run_id];
                    if (!record) {
                      return (
                        <td key={`${row.caseId}-${run.manifest.run_id}`} style={tdStyle}>
                          missing
                        </td>
                      );
                    }
                    const tokenUsage = record.token_usage?.total_tokens;
                    const tools = record.tool_summary?.total_calls ?? 0;
                    const toolNames = record.tool_summary?.by_tool ? Object.keys(record.tool_summary.by_tool).join(", ") : "";
                    const isOpenEnded = record.case.type === "open_ended";
                    return (
                      <td key={`${row.caseId}-${run.manifest.run_id}`} style={tdStyle}>
                        <div style={{ fontWeight: 600 }}>{record.answer_status}</div>
                        {isOpenEnded ? (
                          <>
                            <div style={mutedStyle}>
                              score: {record.score_value == null ? "n/a" : record.score_value.toFixed(2)} | threshold:{" "}
                              {record.score_threshold == null ? "n/a" : record.score_threshold.toFixed(2)}
                            </div>
                            {record.judge_notes ? <div style={mutedStyle}>judge: {record.judge_notes}</div> : null}
                          </>
                        ) : (
                          <div style={mutedStyle}>
                            answer: {record.selected_option ?? "n/a"} | expected: {record.expected_option ?? "n/a"}
                          </div>
                        )}
                        <div style={mutedStyle}>
                          tools: {tools} | latency: {typeof record.latency_seconds === "number" ? `${record.latency_seconds.toFixed(1)}s` : "n/a"}
                        </div>
                        <div style={mutedStyle}>tokens: {tokenUsage ?? "n/a"}</div>
                        {toolNames ? <div style={mutedStyle}>tool names: {toolNames}</div> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
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
  marginBottom: 0,
  fontSize: 18
};

const headerRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: 12,
  marginBottom: 12
};

const filterBarStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  justifyContent: "flex-end"
};

const filterChipStyle: React.CSSProperties = {
  borderRadius: 999,
  border: "1px solid rgba(148, 163, 184, 0.25)",
  background: "rgba(15, 23, 42, 0.75)",
  color: "#cbd5e1",
  fontSize: 12,
  padding: "6px 10px",
  cursor: "pointer"
};

const filterChipActiveStyle: React.CSSProperties = {
  background: "rgba(59, 130, 246, 0.18)",
  border: "1px solid rgba(96, 165, 250, 0.55)",
  color: "#dbeafe"
};

const clearButtonStyle: React.CSSProperties = {
  ...filterChipStyle,
  color: "#fca5a5"
};

const filterSummaryStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 12,
  marginBottom: 10
};

const tableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse"
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  borderBottom: "1px solid rgba(148, 163, 184, 0.2)",
  color: "#bfdbfe",
  fontSize: 13,
  verticalAlign: "top"
};

const headerSubLabelStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 11,
  fontWeight: 400,
  marginTop: 4
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderBottom: "1px solid rgba(148, 163, 184, 0.12)",
  fontSize: 13,
  verticalAlign: "top"
};

const tdLabelStyle: React.CSSProperties = {
  ...tdStyle,
  color: "#cbd5e1",
  fontWeight: 600
};

const mutedStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 12,
  marginTop: 4
};
