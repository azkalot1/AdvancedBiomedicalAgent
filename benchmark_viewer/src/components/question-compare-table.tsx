"use client";

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
  return (
    <section style={sectionStyle}>
      <h2 style={sectionTitleStyle}>Per-Question Comparison</h2>
      {runs.length === 0 ? (
        <div style={mutedStyle}>Select runs to compare their question-by-question outcomes.</div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Case</th>
                <th style={thStyle}>Question</th>
                {runs.map((run) => (
                  <th key={run.manifest.run_id} style={thStyle}>
                    {run.manifest.profile.model.model_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
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
                    <div style={mutedStyle}>{row.category || "uncategorized"}</div>
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
                    return (
                      <td key={`${row.caseId}-${run.manifest.run_id}`} style={tdStyle}>
                        <div style={{ fontWeight: 600 }}>{record.answer_status}</div>
                        <div style={mutedStyle}>
                          answer: {record.selected_option ?? "n/a"} | expected: {record.expected_option ?? "n/a"}
                        </div>
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
  marginBottom: 12,
  fontSize: 18
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
