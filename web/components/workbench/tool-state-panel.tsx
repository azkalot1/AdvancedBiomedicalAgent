"use client";

import { CheckCircle2, CircleDashed, Clock3, LoaderCircle, TriangleAlert, X } from "lucide-react";
import { useState } from "react";

import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { safeDate } from "@/lib/utils";

function statusIcon(status?: string): React.ReactElement {
  if (status === "success") {
    return <CheckCircle2 className="h-4 w-4 text-status-success" />;
  }
  if (status === "error") {
    return <TriangleAlert className="h-4 w-4 text-status-error" />;
  }
  if (status === "queued") {
    return <Clock3 className="h-4 w-4 text-status-queued" />;
  }
  return <LoaderCircle className="h-4 w-4 animate-spin text-status-running" />;
}

function statusBadge(status?: string): string {
  if (status === "success") {
    return "bg-status-success/20 text-status-success";
  }
  if (status === "error") {
    return "bg-status-error/20 text-status-error";
  }
  if (status === "queued") {
    return "bg-status-queued/20 text-status-queued";
  }
  return "bg-status-running/20 text-status-running";
}

export function ToolStatePanel(): React.ReactElement {
  const events = useBioAgentStore((state) => state.toolEvents);
  const clearToolEvents = useBioAgentStore((state) => state.clearToolEvents);
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);

  return (
    <section className="flex h-full min-h-0 flex-col bg-surface/45">
      <div className="flex items-center justify-between border-b border-surface-edge/60 px-3 py-3">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">Tool State</h2>
          <p className="mt-1 text-xs text-zinc-500">Live lifecycle and argument trace for tool calls.</p>
        </div>
        <button
          type="button"
          onClick={() => clearToolEvents()}
          className="inline-flex items-center gap-1 rounded-md border border-surface-edge bg-surface-raised px-2 py-1 text-xs text-zinc-300 hover:border-accent-blue/60 hover:text-zinc-100"
        >
          <X className="h-3.5 w-3.5" /> Clear
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {events.length === 0 ? (
          <div className="flex items-center gap-2 rounded-md border border-dashed border-surface-edge px-3 py-2 text-xs text-zinc-500">
            <CircleDashed className="h-4 w-4" />
            No tool status updates yet.
          </div>
        ) : (
          <ul className="space-y-2">
            {events.map((event) => (
              <li
                key={event.id}
                className="rounded-md border border-surface-edge bg-surface-raised/60 px-2 py-2 text-xs"
                onClick={() => setExpandedEventId((current) => (current === event.id ? null : event.id))}
              >
                {event.type === "tool_status" ? (
                  <>
                    <div className="flex items-start gap-2">
                      {statusIcon(event.status)}
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium text-zinc-100">{event.toolName}</span>
                          <span className={`rounded px-1.5 py-0.5 text-[10px] ${statusBadge(event.status)}`}>
                            {event.status ?? "running"}
                          </span>
                          {typeof event.progress === "number" ? (
                            <span className="text-[10px] text-zinc-400">{event.progress}%</span>
                          ) : null}
                          {typeof event.durationMs === "number" ? (
                            <span className="text-[10px] text-zinc-400">{event.durationMs} ms</span>
                          ) : null}
                        </div>
                        <div className="mt-0.5 text-zinc-500">{safeDate(event.timestamp)}</div>
                      </div>
                    </div>

                    {expandedEventId === event.id ? (
                      <div className="mt-2">
                        <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">
                          {event.argsPreview ? "Arguments" : "No arguments captured"}
                        </div>
                        <pre className="overflow-x-auto whitespace-pre-wrap rounded border border-surface-edge/60 bg-slate-950/60 p-1 text-[10px] text-zinc-300">
                          {event.argsPreview ? JSON.stringify(event.argsPreview, null, 2) : "This tool event has no recorded args payload."}
                        </pre>
                      </div>
                    ) : null}

                    <div className="mt-1 text-[10px] text-zinc-500">
                      {expandedEventId === event.id ? "Click to collapse details" : "Click to view call arguments"}
                    </div>

                    {event.error ? (
                      <div className="mt-2 rounded border border-status-error/40 bg-status-error/10 px-2 py-1 text-status-error">
                        {event.error}
                      </div>
                    ) : null}
                  </>
                ) : event.type === "report_generated" ? (
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent-cyan" />
                    <div>
                      <div className="text-zinc-100">Report generated: {event.report?.displayName ?? event.report?.filename ?? "report.md"}</div>
                      <div className="text-zinc-500">{safeDate(event.timestamp)}</div>
                    </div>
                  </div>
                ) : event.type === "context_updated" ? (
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 text-amber-300" />
                    <div>
                      <div className="text-zinc-100">{event.message ?? "Context updated"}</div>
                      <div className="text-zinc-500">{safeDate(event.timestamp)}</div>
                      {event.argsPreview ? (
                        <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded border border-surface-edge/60 bg-slate-950/60 p-1 text-[10px] text-zinc-300">
                          {JSON.stringify(event.argsPreview, null, 2)}
                        </pre>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <Clock3 className="mt-0.5 h-4 w-4 text-zinc-400" />
                    <div>
                      <div className="text-zinc-100">{event.type}</div>
                      <div className="text-zinc-500">{safeDate(event.timestamp)}</div>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
