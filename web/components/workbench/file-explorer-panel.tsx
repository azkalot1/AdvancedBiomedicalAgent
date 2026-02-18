"use client";

import * as ContextMenu from "@radix-ui/react-context-menu";
import Fuse from "fuse.js";
import { CheckCircle2, CircleX, Clock3, FileText, Folder, LoaderCircle, RefreshCcw } from "lucide-react";
import { useMemo, useState } from "react";

import { listThreads } from "@/lib/backend-client";
import { reportDisplayName, threadDisplayName } from "@/lib/display-names";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { loadThreadSession, sortThreadsByCreatedAt } from "@/lib/thread-session";
import type { ReportFile } from "@/lib/types";
import { cn, safeDate } from "@/lib/utils";

function statusIcon(status: ReportFile["status"]): React.ReactElement {
  if (status === "complete") {
    return <CheckCircle2 className="h-4 w-4 text-status-success" />;
  }
  if (status === "error") {
    return <CircleX className="h-4 w-4 text-status-error" />;
  }
  return <LoaderCircle className="h-4 w-4 animate-spin text-status-running" />;
}

export function FileExplorerPanel(): React.ReactElement {
  const userId = useBioAgentStore((state) => state.userId);
  const threads = useBioAgentStore((state) => state.threads);
  const setThreads = useBioAgentStore((state) => state.setThreads);
  const threadId = useBioAgentStore((state) => state.threadId);
  const reports = useBioAgentStore((state) => state.reports);
  const selectedReportId = useBioAgentStore((state) => state.selectedReportId);
  const selectReport = useBioAgentStore((state) => state.selectReport);
  const queueAttachment = useBioAgentStore((state) => state.queueAttachment);
  const addContextItem = useBioAgentStore((state) => state.addContextItem);
  const setConnection = useBioAgentStore((state) => state.setConnection);

  const [query, setQuery] = useState("");
  const [threadLoadingId, setThreadLoadingId] = useState<string | null>(null);
  const [threadsRefreshing, setThreadsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const visibleThreads = useMemo(() => {
    if (!userId) {
      return threads;
    }
    return threads.filter((item) => {
      const metadataUser = item.metadata?.user_id;
      if (typeof metadataUser !== "string" || !metadataUser) {
        return true;
      }
      return metadataUser === userId;
    });
  }, [threads, userId]);

  const filteredReports = useMemo(() => {
    if (!query.trim()) {
      return reports;
    }

    const fuse = new Fuse(reports, {
      keys: ["displayName", "filename", "oneLine", "toolName"],
      threshold: 0.35
    });

    return fuse.search(query.trim()).map((entry) => entry.item);
  }, [query, reports]);

  const completedCount = reports.filter((report) => report.status === "complete").length;

  const refreshThreads = async (): Promise<void> => {
    setThreadsRefreshing(true);
    setError(null);
    try {
      setThreads(sortThreadsByCreatedAt(await listThreads(100)));
    } catch (refreshError) {
      const message = refreshError instanceof Error ? refreshError.message : "Failed to refresh threads.";
      setError(message);
      setConnection("degraded");
    } finally {
      setThreadsRefreshing(false);
    }
  };

  const handleThreadClick = async (selectedThreadId: string): Promise<void> => {
    setError(null);
    setThreadLoadingId(selectedThreadId);
    try {
      await loadThreadSession(selectedThreadId);
    } catch (threadError) {
      const message = threadError instanceof Error ? threadError.message : "Failed to load thread.";
      setError(message);
      setConnection("degraded");
    } finally {
      setThreadLoadingId(null);
    }
  };

  const addReportToContext = (report: ReportFile): void => {
    const label = reportDisplayName(report);
    addContextItem({
      id: `ctx-file-${report.id}-${Date.now()}`,
      type: "full_file",
      source: label,
      content: report.oneLine ? `${label}\n\n${report.oneLine}` : label,
      tokenCount: Math.max(20, Math.ceil((report.sizeChars ?? 500) / 4)),
      addedAt: new Date().toISOString()
    });
  };

  const onDrop = (event: React.DragEvent<HTMLDivElement>): void => {
    const reportId = event.dataTransfer.getData("application/x-report-id");
    if (reportId) {
      queueAttachment(reportId);
    }
  };

  return (
    <section className="flex h-full min-h-0 flex-col border-r border-surface-edge/60 bg-surface/50">
      <div className="border-b border-surface-edge/60 px-3 py-3">
        <h2 className="text-lg font-semibold text-zinc-100">Generated Reports</h2>
        <p className="mt-1 text-xs text-zinc-400">
          {completedCount} complete / {reports.length} total
        </p>
      </div>

      <div className="border-b border-surface-edge/50 px-3 py-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search reports"
          className="w-full rounded-md border border-surface-edge bg-surface-raised px-2 py-1.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-500 focus:border-accent-blue/80"
        />
      </div>

      <div className="border-b border-surface-edge/60 px-2 py-2">
        <div className="mb-2 flex items-center justify-between px-1">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Threads</h3>
          <button
            type="button"
            onClick={() => void refreshThreads()}
            className="inline-flex items-center gap-1 rounded border border-surface-edge bg-surface-raised px-2 py-1 text-[11px] text-zinc-300 hover:border-accent-blue/60"
          >
            <RefreshCcw className={cn("h-3 w-3", threadsRefreshing ? "animate-spin" : "")} />
            Refresh
          </button>
        </div>

        <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
          {visibleThreads.length === 0 ? (
            <div className="rounded-md border border-dashed border-surface-edge px-2 py-2 text-xs text-zinc-500">
              No saved threads yet.
            </div>
          ) : (
            visibleThreads.map((thread) => (
              <button
                key={thread.id}
                type="button"
                onClick={() => void handleThreadClick(thread.id)}
                className={cn(
                  "w-full rounded-md border px-2 py-1.5 text-left text-xs transition",
                  thread.id === threadId
                    ? "border-accent-blue/60 bg-accent-blue/15 text-zinc-100"
                    : "border-transparent text-zinc-300 hover:border-surface-edge hover:bg-surface-overlay/50"
                )}
              >
                <div className="inline-flex items-center gap-1">
                  <Folder className="h-3.5 w-3.5 text-zinc-400" />
                  <span className="truncate">{threadDisplayName(thread)}</span>
                  {threadLoadingId === thread.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin text-accent-cyan" /> : null}
                </div>
                <div className="mt-0.5 text-[10px] text-zinc-500">{safeDate(thread.createdAt) || "unknown time"}</div>
              </button>
            ))
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 p-2" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
        <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-zinc-400">Reports in Selected Thread</div>
        <div className="h-full space-y-1 overflow-y-auto pr-1">
          {!threadId ? (
            <div className="rounded-md border border-dashed border-surface-edge px-2 py-2 text-xs text-zinc-500">
              Waiting for thread initialization.
            </div>
          ) : filteredReports.length === 0 ? (
            <div className="rounded-md border border-dashed border-surface-edge px-2 py-2 text-xs text-zinc-500">
              No reports for this thread.
            </div>
          ) : (
            filteredReports.map((report) => (
              <ContextMenu.Root key={report.id}>
                <ContextMenu.Trigger asChild>
                  <div
                    draggable
                    onDragStart={(event) => {
                      event.dataTransfer.setData("application/x-report-id", report.id);
                      event.dataTransfer.effectAllowed = "copy";
                    }}
                    onClick={() => selectReport(selectedReportId === report.id ? null : report.id)}
                    className={cn(
                      "flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5 text-sm",
                      selectedReportId === report.id
                        ? "border-accent-blue/60 bg-accent-blue/20 text-zinc-100"
                        : "border-transparent text-zinc-300 hover:border-surface-edge hover:bg-surface-overlay/50"
                    )}
                    title={report.oneLine}
                  >
                    <FileText className="h-4 w-4 text-accent-blue" />
                    <span className="truncate">{reportDisplayName(report)}</span>
                    <span className="ml-auto">{statusIcon(report.status)}</span>
                  </div>
                </ContextMenu.Trigger>

                <ContextMenu.Portal>
                  <ContextMenu.Content className="z-50 min-w-[190px] rounded-md border border-surface-edge bg-surface-raised p-1 text-sm shadow-xl">
                    <ContextMenu.Item
                      onSelect={() => queueAttachment(report.id)}
                      className="cursor-pointer rounded px-2 py-1 text-zinc-200 outline-none hover:bg-surface-overlay"
                    >
                      Add to Chat Input
                    </ContextMenu.Item>
                    <ContextMenu.Item
                      onSelect={() => addReportToContext(report)}
                      className="cursor-pointer rounded px-2 py-1 text-zinc-200 outline-none hover:bg-surface-overlay"
                    >
                      Add to Context
                    </ContextMenu.Item>
                  </ContextMenu.Content>
                </ContextMenu.Portal>
              </ContextMenu.Root>
            ))
          )}
        </div>
      </div>

      <div className="border-t border-surface-edge/50 p-3 text-xs text-zinc-500">
        <div className="flex items-center gap-1">
          <Clock3 className="h-3 w-3" />
          {error ?? `Latest: ${safeDate(reports[0]?.createdAt) || "n/a"}`}
        </div>
      </div>
    </section>
  );
}
