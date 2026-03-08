"use client";

import { CheckCircle2, CircleDashed, Clock3, LoaderCircle, TriangleAlert } from "lucide-react";

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

export function ToolHistoryTimeline(): React.ReactElement {
  const events = useBioAgentStore((state) => state.toolEvents);

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-surface-edge/60">
      <div className="px-3 pb-2 pt-3">
        <h3 className="text-sm font-semibold text-zinc-100">Tool Execution History</h3>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {events.length === 0 ? (
          <div className="flex items-center gap-2 rounded-md border border-dashed border-surface-edge px-3 py-2 text-xs text-zinc-500">
            <CircleDashed className="h-4 w-4" />
            No tool activity yet.
          </div>
        ) : (
          <ul className="space-y-2">
            {events.map((event) => (
              <li key={event.id} className="rounded-md border border-surface-edge bg-surface-raised/60 px-2 py-1.5 text-xs">
                {event.type === "tool_status" ? (
                  <div className="flex items-start gap-2">
                    {statusIcon(event.status)}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-zinc-100">
                        {event.toolName} {event.status ? `- ${event.status}` : ""}
                        {typeof event.progress === "number" ? ` (${event.progress}%)` : ""}
                      </div>
                      <div className="text-zinc-500">
                        {safeDate(event.timestamp)}
                        {typeof event.durationMs === "number" ? ` - ${event.durationMs} ms` : ""}
                      </div>
                      {event.argsPreview ? (
                        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded border border-surface-edge/60 bg-slate-950/60 p-1 text-[10px] text-zinc-300">
                          {JSON.stringify(event.argsPreview, null, 2)}
                        </pre>
                      ) : null}
                      {event.error ? <div className="mt-1 text-status-error">{event.error}</div> : null}
                    </div>
                  </div>
                ) : event.type === "report_generated" ? (
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-accent-cyan" />
                    <div>
                      <div className="text-zinc-100">Report generated: {event.report?.displayName ?? event.report?.filename ?? "report.md"}</div>
                      <div className="text-zinc-500">{safeDate(event.timestamp)}</div>
                    </div>
                  </div>
                ) : event.type === "context_updated" ? (
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-amber-300" />
                    <div>
                      <div className="text-zinc-100">{event.message ?? "Context updated"}</div>
                      <div className="text-zinc-500">{safeDate(event.timestamp)}</div>
                      {event.argsPreview ? (
                        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded border border-surface-edge/60 bg-slate-950/60 p-1 text-[10px] text-zinc-300">
                          {JSON.stringify(event.argsPreview, null, 2)}
                        </pre>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <Clock3 className="h-4 w-4 text-zinc-400" />
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
    </div>
  );
}
