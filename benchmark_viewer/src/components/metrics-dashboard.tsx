"use client";

import type { LoadedRunData } from "@/src/lib/types";

interface MetricsDashboardProps {
  runs: LoadedRunData[];
}

export function MetricsDashboard({ runs }: MetricsDashboardProps): React.ReactElement {
  const categoryNames = Array.from(new Set(runs.flatMap((run) => Object.keys(run.summary.by_category)))).sort();
  const toolNames = Array.from(new Set(runs.flatMap((run) => Object.keys(run.summary.tool_usage)))).sort();
  const chartRows = comparisonChartRows(runs);

  return (
    <section style={sectionStyle}>
      <h2 style={sectionTitleStyle}>Metrics Dashboard</h2>
      {runs.length === 0 ? (
        <div style={mutedStyle}>Select one or more runs to compare aggregate metrics.</div>
      ) : (
        <>
          <div style={{ overflowX: "auto" }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Metric</th>
                  {runs.map((run) => (
                    <th key={run.manifest.run_id} style={thStyle}>
                      {run.manifest.profile.model.model_name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metricRows(runs).map((row) => (
                  <tr key={row.label}>
                    <td style={tdLabelStyle}>{row.label}</td>
                    {row.values.map((value, index) => (
                      <td key={`${row.label}-${index}`} style={tdStyle}>
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={subheadingStyle}>Visual Comparison</h3>
            <div style={{ display: "grid", gap: 14 }}>
              {chartRows.map((row) => (
                <div key={row.label} style={chartCardStyle}>
                  <div style={chartTitleStyle}>{row.label}</div>
                  <div style={{ display: "grid", gap: 8 }}>
                    {row.items.map((item) => (
                      <div key={`${row.label}-${item.label}`}>
                        <div style={chartLabelRowStyle}>
                          <span>{item.label}</span>
                          <span>{item.displayValue}</span>
                        </div>
                        <div style={barTrackStyle}>
                          <div style={{ ...barFillStyle, width: `${item.widthPercent}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {categoryNames.length > 0 ? (
            <div style={{ marginTop: 20 }}>
              <h3 style={subheadingStyle}>Category Breakdown</h3>
              <div style={{ overflowX: "auto" }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Category</th>
                      {runs.map((run) => (
                        <th key={run.manifest.run_id} style={thStyle}>
                          {run.manifest.profile.model.model_name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {categoryNames.map((category) => (
                      <tr key={category}>
                        <td style={tdLabelStyle}>{category}</td>
                        {runs.map((run) => {
                          const stats = run.summary.by_category[category];
                          return (
                            <td key={`${run.manifest.run_id}-${category}`} style={tdStyle}>
                              {stats ? `${(stats.accuracy * 100).toFixed(1)}% (${stats.correct}/${stats.total})` : "n/a"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {toolNames.length > 0 ? (
            <div style={{ marginTop: 20 }}>
              <h3 style={subheadingStyle}>Tool Usage</h3>
              <div style={{ overflowX: "auto" }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Tool</th>
                      {runs.map((run) => (
                        <th key={run.manifest.run_id} style={thStyle}>
                          {run.manifest.profile.model.model_name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {toolNames.map((toolName) => (
                      <tr key={toolName}>
                        <td style={tdLabelStyle}>{toolName}</td>
                        {runs.map((run) => (
                          <td key={`${run.manifest.run_id}-${toolName}`} style={tdStyle}>
                            {run.summary.tool_usage[toolName] ?? 0}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function metricRows(runs: LoadedRunData[]): Array<{ label: string; values: string[] }> {
  return [
    {
      label: "Accuracy",
      values: runs.map((run) => `${(run.summary.summary.accuracy * 100).toFixed(1)}%`)
    },
    {
      label: "Invalid format rate",
      values: runs.map((run) => `${(run.summary.summary.invalid_format_rate * 100).toFixed(1)}%`)
    },
    {
      label: "Average latency",
      values: runs.map((run) => `${run.summary.summary.average_latency_seconds.toFixed(2)}s`)
    },
    {
      label: "Average tool calls",
      values: runs.map((run) => run.summary.summary.average_tool_calls.toFixed(2))
    },
    {
      label: "Total tokens",
      values: runs.map((run) => String(run.summary.summary.sum_total_tokens ?? "n/a"))
    },
    {
      label: "Average total tokens",
      values: runs.map((run) =>
        run.summary.summary.average_total_tokens == null ? "n/a" : run.summary.summary.average_total_tokens.toFixed(1)
      )
    },
    {
      label: "Streamed output chars",
      values: runs.map((run) => String(run.summary.summary.sum_streamed_output_chars ?? "n/a"))
    }
  ];
}

function comparisonChartRows(
  runs: LoadedRunData[]
): Array<{ label: string; items: Array<{ label: string; displayValue: string; widthPercent: number }> }> {
  const rows = [
    {
      label: "Accuracy",
      values: runs.map((run) => ({
        label: run.manifest.profile.model.model_name,
        numericValue: run.summary.summary.accuracy * 100,
        displayValue: `${(run.summary.summary.accuracy * 100).toFixed(1)}%`
      }))
    },
    {
      label: "Average latency",
      values: runs.map((run) => ({
        label: run.manifest.profile.model.model_name,
        numericValue: run.summary.summary.average_latency_seconds,
        displayValue: `${run.summary.summary.average_latency_seconds.toFixed(2)}s`
      }))
    },
    {
      label: "Average tool calls",
      values: runs.map((run) => ({
        label: run.manifest.profile.model.model_name,
        numericValue: run.summary.summary.average_tool_calls,
        displayValue: run.summary.summary.average_tool_calls.toFixed(2)
      }))
    },
    {
      label: "Total tokens",
      values: runs.map((run) => ({
        label: run.manifest.profile.model.model_name,
        numericValue: run.summary.summary.sum_total_tokens ?? 0,
        displayValue: String(run.summary.summary.sum_total_tokens ?? "n/a")
      }))
    }
  ];

  return rows.map((row) => {
    const maxValue = Math.max(...row.values.map((value) => value.numericValue), 0);
    return {
      label: row.label,
      items: row.values.map((value) => ({
        label: value.label,
        displayValue: value.displayValue,
        widthPercent: maxValue > 0 ? (value.numericValue / maxValue) * 100 : 0
      }))
    };
  });
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

const subheadingStyle: React.CSSProperties = {
  marginTop: 0,
  marginBottom: 10,
  fontSize: 16
};

const chartCardStyle: React.CSSProperties = {
  border: "1px solid rgba(148, 163, 184, 0.12)",
  borderRadius: 12,
  padding: 12,
  background: "rgba(30, 41, 59, 0.45)"
};

const chartTitleStyle: React.CSSProperties = {
  fontWeight: 700,
  marginBottom: 10,
  color: "#e2e8f0"
};

const chartLabelRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  gap: 10,
  fontSize: 13,
  marginBottom: 4
};

const barTrackStyle: React.CSSProperties = {
  width: "100%",
  height: 10,
  borderRadius: 999,
  background: "rgba(15, 23, 42, 0.9)",
  overflow: "hidden",
  border: "1px solid rgba(148, 163, 184, 0.12)"
};

const barFillStyle: React.CSSProperties = {
  height: "100%",
  borderRadius: 999,
  background: "linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%)"
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
  fontSize: 13
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderBottom: "1px solid rgba(148, 163, 184, 0.12)",
  fontSize: 13
};

const tdLabelStyle: React.CSSProperties = {
  ...tdStyle,
  color: "#cbd5e1",
  fontWeight: 600
};

const mutedStyle: React.CSSProperties = {
  color: "#94a3b8"
};
