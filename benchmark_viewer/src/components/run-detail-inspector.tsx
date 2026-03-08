"use client";

import type { CaseDetailRecord, LoadedRunData } from "@/src/lib/types";

interface RunDetailInspectorProps {
  runs: LoadedRunData[];
  selectedCaseId: string | null;
  activeRunId: string | null;
  onChangeActiveRun(runId: string): void;
  detail: CaseDetailRecord | null;
  detailLoading: boolean;
}

export function RunDetailInspector({
  runs,
  selectedCaseId,
  activeRunId,
  onChangeActiveRun,
  detail,
  detailLoading
}: RunDetailInspectorProps): React.ReactElement {
  return (
    <section style={sectionStyle}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
        <div>
          <h2 style={sectionTitleStyle}>Run Detail Inspector</h2>
          <div style={mutedStyle}>{selectedCaseId ? `Inspecting case ${selectedCaseId}` : "Select a case to inspect."}</div>
        </div>
        <select
          value={activeRunId ?? ""}
          onChange={(event) => onChangeActiveRun(event.target.value)}
          disabled={runs.length === 0}
          style={selectStyle}
        >
          {runs.map((run) => (
            <option key={run.manifest.run_id} value={run.manifest.run_id}>
              {run.manifest.profile.model.model_name}
            </option>
          ))}
        </select>
      </div>

      {!selectedCaseId ? (
        <div style={{ marginTop: 14, color: "#94a3b8" }}>No case selected.</div>
      ) : detailLoading ? (
        <div style={{ marginTop: 14, color: "#94a3b8" }}>Loading detail...</div>
      ) : !detail ? (
        <div style={{ marginTop: 14, color: "#94a3b8" }}>No detail available for this run and case.</div>
      ) : (
        <div style={{ marginTop: 16, display: "grid", gap: 16 }}>
          <div style={cardStyle}>
            <div style={headingStyle}>Question</div>
            <div>{detail.case.question}</div>
            <pre style={preStyle}>{JSON.stringify(detail.case.options, null, 2)}</pre>
          </div>

          <div style={cardStyle}>
            <div style={headingStyle}>Answer Summary</div>
            <div>status: {detail.answer_status}</div>
            <div>selected: {detail.selected_option ?? "n/a"}</div>
            <div>expected: {detail.expected_option ?? "n/a"}</div>
            <div>latency: {typeof detail.latency_seconds === "number" ? `${detail.latency_seconds.toFixed(2)}s` : "n/a"}</div>
            <div>
              tokens: {detail.token_usage?.total_tokens ?? "n/a"} (in: {detail.token_usage?.input_tokens ?? "n/a"}, out:{" "}
              {detail.token_usage?.output_tokens ?? "n/a"})
            </div>
            <pre style={preStyle}>{detail.final_answer_text || "[empty assistant answer]"}</pre>
          </div>

          <div style={cardStyle}>
            <div style={headingStyle}>Tool Timeline</div>
            {detail.tool_invocations && detail.tool_invocations.length > 0 ? (
              <div style={{ display: "grid", gap: 10 }}>
                {detail.tool_invocations.map((invocation) => (
                  <div key={invocation.invocation_id} style={subCardStyle}>
                    <div style={{ fontWeight: 600 }}>{invocation.tool_name}</div>
                    <div style={mutedStyle}>statuses: {invocation.statuses.join(" -> ") || "n/a"}</div>
                    <div style={mutedStyle}>
                      started: {invocation.started_at || "n/a"} | finished: {invocation.finished_at || "n/a"} | duration:{" "}
                      {invocation.duration_ms ?? "n/a"} ms
                    </div>
                    {invocation.args_preview != null ? <pre style={preStyle}>{JSON.stringify(invocation.args_preview, null, 2)}</pre> : null}
                    {invocation.error ? <pre style={preStyle}>{String(invocation.error)}</pre> : null}
                  </div>
                ))}
              </div>
            ) : (
              <div style={mutedStyle}>No tool invocations recorded.</div>
            )}
          </div>

          <div style={cardStyle}>
            <div style={headingStyle}>Normalized Message Stack</div>
            {detail.normalized_messages && detail.normalized_messages.length > 0 ? (
              <div style={{ display: "grid", gap: 10 }}>
                {detail.normalized_messages.map((message) => (
                  <details key={`${message.index}-${message.id ?? "message"}`} style={subCardStyle}>
                    <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                      #{message.index} | role: {message.role || message.type || "unknown"} | name: {message.name || "n/a"}
                    </summary>
                    <div style={{ marginTop: 8 }}>
                      <div style={mutedStyle}>content:</div>
                      <pre style={preStyle}>{message.content_text || JSON.stringify(message.content_raw, null, 2)}</pre>
                      {message.tool_calls && message.tool_calls.length > 0 ? (
                        <>
                          <div style={mutedStyle}>tool_calls:</div>
                          <pre style={preStyle}>{JSON.stringify(message.tool_calls, null, 2)}</pre>
                        </>
                      ) : null}
                      {message.additional_tool_calls && message.additional_tool_calls.length > 0 ? (
                        <>
                          <div style={mutedStyle}>additional_tool_calls:</div>
                          <pre style={preStyle}>{JSON.stringify(message.additional_tool_calls, null, 2)}</pre>
                        </>
                      ) : null}
                      {message.usage_metadata ? (
                        <>
                          <div style={mutedStyle}>usage_metadata:</div>
                          <pre style={preStyle}>{JSON.stringify(message.usage_metadata, null, 2)}</pre>
                        </>
                      ) : null}
                    </div>
                  </details>
                ))}
              </div>
            ) : (
              <div style={mutedStyle}>No normalized messages recorded.</div>
            )}
          </div>

          <div style={cardStyle}>
            <div style={headingStyle}>Raw Thread State</div>
            <pre style={preStyle}>{JSON.stringify(detail.thread_state ?? {}, null, 2)}</pre>
          </div>
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
  margin: 0,
  fontSize: 18
};

const cardStyle: React.CSSProperties = {
  border: "1px solid rgba(148, 163, 184, 0.12)",
  borderRadius: 12,
  padding: 12,
  background: "rgba(30, 41, 59, 0.45)"
};

const subCardStyle: React.CSSProperties = {
  border: "1px solid rgba(148, 163, 184, 0.1)",
  borderRadius: 10,
  padding: 10,
  background: "rgba(15, 23, 42, 0.45)"
};

const headingStyle: React.CSSProperties = {
  fontWeight: 700,
  marginBottom: 8
};

const mutedStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 13
};

const preStyle: React.CSSProperties = {
  marginTop: 8,
  padding: 10,
  borderRadius: 10,
  background: "rgba(2, 6, 23, 0.7)",
  border: "1px solid rgba(148, 163, 184, 0.1)",
  overflowX: "auto",
  fontSize: 12
};

const selectStyle: React.CSSProperties = {
  borderRadius: 8,
  background: "#0f172a",
  color: "#e5edf9",
  border: "1px solid rgba(148, 163, 184, 0.2)",
  padding: "8px 10px"
};
