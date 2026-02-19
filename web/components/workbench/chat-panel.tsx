"use client";

import { AlertTriangle, ChevronDown, Paperclip, Plus, Send, Square } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Virtuoso } from "react-virtuoso";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

import {
  createThread,
  generateThreadDisplayName,
  getPromptTokensFromStreamEvent,
  getThreadState,
  getTokenChunkFromEvent,
  listReports,
  listThreads,
  parseThreadPromptTokens,
  setThreadDisplayName,
  streamRun
} from "@/lib/backend-client";
import { reportDisplayName, threadDisplayName } from "@/lib/display-names";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { loadThreadSession, prependThread, sortThreadsByCreatedAt } from "@/lib/thread-session";
import type { ChatMessage, ContextItem, NormalizedStreamEvent, ThreadSummary, ToolEvent } from "@/lib/types";
import { estimateTokens } from "@/lib/token-estimate";
import { cn, shortId } from "@/lib/utils";

function nextId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 10000)}`;
}

function buildAttachmentContext(ids: string[], reports: ReturnType<typeof useBioAgentStore.getState>["reports"]): ContextItem[] {
  const selected = ids
    .map((id) => reports.find((report) => report.id === id))
    .filter((report): report is NonNullable<typeof report> => Boolean(report));

  return selected.map((report) => {
    const label = reportDisplayName(report);
    return {
      id: `ctx-attach-${report.id}-${Date.now()}`,
      type: "full_file",
      source: label,
      content: report.oneLine ? `${label}\n${report.oneLine}` : label,
      tokenCount: Math.max(25, Math.ceil((report.sizeChars ?? 500) / 4)),
      addedAt: new Date().toISOString()
    };
  });
}

function buildThreadNamingMessages(messages: ChatMessage[]): Array<{ role: "user" | "assistant"; content: string }> {
  const normalized = messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({ role: message.role, content: message.content.trim() }))
    .filter((message) => message.content.length > 0);

  const firstUserIndex = normalized.findIndex((message) => message.role === "user");
  if (firstUserIndex < 0) {
    return normalized.slice(0, 2);
  }
  const firstAssistantAfterUser = normalized.findIndex(
    (message, index) => index > firstUserIndex && message.role === "assistant"
  );
  if (firstAssistantAfterUser < 0) {
    return normalized.slice(firstUserIndex, firstUserIndex + 2);
  }
  return [normalized[firstUserIndex], normalized[firstAssistantAfterUser]];
}

function threadHasExplicitName(thread: ThreadSummary | undefined): boolean {
  if (!thread) {
    return false;
  }
  if (typeof thread.displayName === "string" && thread.displayName.trim()) {
    return true;
  }
  const metadata = thread.metadata;
  if (!metadata) {
    return false;
  }
  const candidate = metadata.display_name ?? metadata.thread_display_name ?? metadata.title;
  return typeof candidate === "string" && candidate.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") {
    return true;
  }
  if (error instanceof Error) {
    if (error.name === "AbortError") {
      return true;
    }
    const message = error.message.toLowerCase();
    return message.includes("abort") || message.includes("cancel");
  }
  return false;
}

function normalizeArgsPreview(value: unknown): Record<string, unknown> | undefined {
  if (value == null) {
    return undefined;
  }
  if (isRecord(value)) {
    return value;
  }
  if (Array.isArray(value)) {
    return { items: value as unknown[] };
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return { value };
  }
  return { value: String(value) };
}

function toToolEvent(payload: Record<string, unknown>): ToolEvent | null {
  const type = payload.type;
  if (typeof type !== "string") {
    return null;
  }

  if (type !== "tool_status" && type !== "report_generated" && type !== "context_updated" && type !== "plot_data") {
    return null;
  }

  if (type === "tool_status") {
    const event: ToolEvent = {
      id: payload.invocation_id ? String(payload.invocation_id) : `tool-${Date.now()}`,
      type,
      toolName: typeof payload.tool_name === "string" ? payload.tool_name : "unknown_tool",
      status: typeof payload.status === "string" ? (payload.status as ToolEvent["status"]) : "running",
      progress: typeof payload.progress === "number" ? payload.progress : undefined,
      durationMs: typeof payload.duration_ms === "number" ? payload.duration_ms : undefined,
      timestamp: typeof payload.timestamp === "string" ? payload.timestamp : new Date().toISOString(),
      invocationId: typeof payload.invocation_id === "string" ? payload.invocation_id : undefined,
      error: typeof payload.error === "string" ? payload.error : undefined
    };

    const argsPreview = normalizeArgsPreview(payload.args_preview);
    if (argsPreview) {
      event.argsPreview = argsPreview;
    }
    return event;
  }

  if (type === "report_generated") {
    const report = payload.report;
    return {
      id: `report-generated-${Date.now()}`,
      type,
      timestamp: new Date().toISOString(),
      report:
        report && typeof report === "object"
          ? {
              id: String((report as Record<string, unknown>).id ?? "unknown"),
              filename: String((report as Record<string, unknown>).filename ?? "report.md"),
              displayName: String(
                (report as Record<string, unknown>).display_name ??
                (report as Record<string, unknown>).one_line ??
                (report as Record<string, unknown>).filename ??
                "report.md"
              ),
              status: "complete"
            }
          : undefined
    };
  }

  if (type === "plot_data" || type === "context_updated") {
    return {
      id: `${type}-${Date.now()}`,
      type: type as ToolEvent["type"],
      timestamp: new Date().toISOString()
    };
  }

  return null;
}

function extractToolEventsFromStream(event: NormalizedStreamEvent): ToolEvent[] {
  const results: ToolEvent[] = [];

  function walk(value: unknown, depth = 0): void {
    if (depth > 6 || value == null) {
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        walk(item, depth + 1);
      }
      return;
    }
    if (!isRecord(value)) {
      return;
    }

    const direct = toToolEvent(value);
    if (direct) {
      results.push(direct);
      return;
    }

    for (const nested of Object.values(value)) {
      walk(nested, depth + 1);
    }
  }

  if (event.mode === "custom" || event.mode === "updates") {
    walk(event.payload);
  }
  return results;
}

export function ChatPanel(): React.ReactElement {
  const assistantId = useBioAgentStore((state) => state.assistantId);
  const threadId = useBioAgentStore((state) => state.threadId);
  const setThreadId = useBioAgentStore((state) => state.setThreadId);
  const clearThreadWorkspace = useBioAgentStore((state) => state.clearThreadWorkspace);
  const threads = useBioAgentStore((state) => state.threads);
  const setThreads = useBioAgentStore((state) => state.setThreads);
  const userId = useBioAgentStore((state) => state.userId);
  const model = useBioAgentStore((state) => state.model);
  const reports = useBioAgentStore((state) => state.reports);
  const setReports = useBioAgentStore((state) => state.setReports);
  const upsertReport = useBioAgentStore((state) => state.upsertReport);
  const messages = useBioAgentStore((state) => state.messages);
  const addMessage = useBioAgentStore((state) => state.addMessage);
  const appendAssistantToken = useBioAgentStore((state) => state.appendAssistantToken);
  const finishAssistantMessage = useBioAgentStore((state) => state.finishAssistantMessage);
  const draft = useBioAgentStore((state) => state.draft);
  const setDraft = useBioAgentStore((state) => state.setDraft);
  const contextItems = useBioAgentStore((state) => state.contextItems);
  const pendingAttachmentIds = useBioAgentStore((state) => state.pendingAttachmentIds);
  const removeAttachment = useBioAgentStore((state) => state.removeAttachment);
  const clearAttachments = useBioAgentStore((state) => state.clearAttachments);
  const addContextItem = useBioAgentStore((state) => state.addContextItem);
  const isStreaming = useBioAgentStore((state) => state.isStreaming);
  const setStreaming = useBioAgentStore((state) => state.setStreaming);
  const pushToolEvent = useBioAgentStore((state) => state.pushToolEvent);
  const setConnection = useBioAgentStore((state) => state.setConnection);
  const setCurrentPromptTokens = useBioAgentStore((state) => state.setCurrentPromptTokens);

  const [error, setError] = useState<string | null>(null);
  const [threadMenuOpen, setThreadMenuOpen] = useState(false);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [canInterrupt, setCanInterrupt] = useState(false);
  const streamAbortControllerRef = useRef<AbortController | null>(null);

  const attachmentReports = useMemo(
    () => pendingAttachmentIds
      .map((id) => reports.find((report) => report.id === id))
      .filter((report): report is NonNullable<typeof report> => Boolean(report)),
    [pendingAttachmentIds, reports]
  );

  const visibleThreads = useMemo(() => {
    if (!userId) {
      return threads;
    }
    return threads.filter((item) => {
      const metadataUser = item.metadata?.user_id;
      return typeof metadataUser !== "string" || metadataUser === userId;
    });
  }, [threads, userId]);

  const activeThread = useMemo(
    () => (threadId ? threads.find((item) => item.id === threadId) : undefined),
    [threadId, threads]
  );

  const maybeGenerateThreadName = async (activeThreadId: string): Promise<void> => {
    try {
      const state = useBioAgentStore.getState();
      const currentThread = state.threads.find((item) => item.id === activeThreadId);
      if (threadHasExplicitName(currentThread)) {
        return;
      }

      const namingMessages = buildThreadNamingMessages(state.messages);
      if (namingMessages.length < 2) {
        return;
      }

      const generated = await generateThreadDisplayName(activeThreadId, namingMessages, {
        minMessages: 2,
        maxMessages: 2
      });
      const persistedMetadata = await setThreadDisplayName(activeThreadId, generated.display_name, state.userId);

      state.setThreads(
        state.threads.map((item) => {
          if (item.id !== activeThreadId) {
            return item;
          }
          return {
            ...item,
            displayName: generated.display_name,
            metadata: { ...(item.metadata ?? {}), ...persistedMetadata }
          };
        })
      );
    } catch {
      console.warn("Failed to generate or persist thread display name.");
      return;
    }
  };

  const openThreadMenu = async (): Promise<void> => {
    if (threadMenuOpen) {
      setThreadMenuOpen(false);
      return;
    }
    setThreadMenuOpen(true);
    setThreadsLoading(true);
    try {
      const items = sortThreadsByCreatedAt(await listThreads(100, userId));
      setThreads(items);
    } catch {
      setThreads([]);
    } finally {
      setThreadsLoading(false);
    }
  };

  const loadThread = async (selectedThreadId: string): Promise<void> => {
    setError(null);
    setThreadMenuOpen(false);
    try {
      await loadThreadSession(selectedThreadId);
    } catch (threadError) {
      const message = threadError instanceof Error ? threadError.message : "Failed to load thread.";
      setError(message);
      setConnection("degraded");
    }
  };

  const startNewChat = async (): Promise<void> => {
    if (isStreaming) {
      return;
    }
    setError(null);
    setThreadMenuOpen(false);
    setStreaming(true);
    try {
      const freshThreadId = await createThread(userId);
      clearThreadWorkspace();
      setThreadId(freshThreadId);
      setThreads(
        prependThread(useBioAgentStore.getState().threads, {
          id: freshThreadId,
          createdAt: new Date().toISOString(),
          metadata: {
            app: "co-scientist",
            ...(userId ? { user_id: userId } : {})
          }
        })
      );
      setConnection("connected");
    } catch (threadError) {
      const message = threadError instanceof Error ? threadError.message : "Failed to start new thread.";
      setError(message);
      setConnection("degraded");
    } finally {
      setStreaming(false);
    }
  };

  const sendMessage = async (): Promise<void> => {
    if (!draft.trim() || isStreaming) {
      return;
    }

    setError(null);

    const userMessageId = nextId("msg-user");
    addMessage({
      id: userMessageId,
      role: "user",
      content: draft.trim(),
      createdAt: new Date().toISOString()
    });

    const assistantMessageId = nextId("msg-assistant");
    addMessage({
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
      streaming: true
    });

    const messageToSend = draft.trim();
    setDraft("");
    setStreaming(true);

    let activeThreadId = threadId;
    let emittedTokens = 0;
    try {
      if (!activeThreadId) {
        activeThreadId = await createThread(userId);
        setThreadId(activeThreadId);
        setThreads(
          prependThread(useBioAgentStore.getState().threads, {
            id: activeThreadId,
            createdAt: new Date().toISOString(),
            metadata: {
              app: "co-scientist",
              ...(userId ? { user_id: userId } : {})
            }
          })
        );
      }

      const attachmentContext = buildAttachmentContext(pendingAttachmentIds, reports);
      const allContextItems = [...contextItems, ...attachmentContext];

      for (const item of attachmentContext) {
        addContextItem(item);
      }
      clearAttachments();

      const streamAbortController = new AbortController();
      streamAbortControllerRef.current = streamAbortController;
      setCanInterrupt(true);

      for await (const event of streamRun({
        threadId: activeThreadId,
        assistantId,
        model,
        userId,
        message: messageToSend,
        contextItems: allContextItems,
        streamToolArgs: true,
        signal: streamAbortController.signal
      })) {
        const chunk = getTokenChunkFromEvent(event);
        if (chunk) {
          appendAssistantToken(assistantMessageId, chunk);
          emittedTokens += estimateTokens(chunk);
        }
        const streamPromptTokens = getPromptTokensFromStreamEvent(event);
        if (streamPromptTokens !== null) {
          setCurrentPromptTokens(streamPromptTokens);
        }

        const toolEvents = extractToolEventsFromStream(event);
        for (const toolEvent of toolEvents) {
          pushToolEvent(toolEvent);
          if (toolEvent.type === "report_generated" && toolEvent.report) {
            upsertReport(toolEvent.report);
            const nextReports = await listReports({ threadId: activeThreadId, limit: 100 });
            setReports(nextReports);
          }
        }
      }

      if (emittedTokens === 0) {
        appendAssistantToken(assistantMessageId, "[No streamed tokens received. Check tool events and backend logs.]");
      }

      const [refreshedReports, refreshedState] = await Promise.all([
        listReports({ threadId: activeThreadId, limit: 100 }),
        getThreadState(activeThreadId)
      ]);
      setReports(refreshedReports);
      setCurrentPromptTokens(parseThreadPromptTokens(refreshedState));
      setConnection("connected");
      void maybeGenerateThreadName(activeThreadId);
    } catch (streamError) {
      if (isAbortError(streamError)) {
        const latestAssistant = useBioAgentStore.getState().messages.find((item) => item.id === assistantMessageId);
        if (!latestAssistant || !latestAssistant.content.trim()) {
          appendAssistantToken(assistantMessageId, "[Generation interrupted]");
        }
        if (activeThreadId) {
          try {
            const [refreshedReports, refreshedState] = await Promise.all([
              listReports({ threadId: activeThreadId, limit: 100 }),
              getThreadState(activeThreadId)
            ]);
            setReports(refreshedReports);
            setCurrentPromptTokens(parseThreadPromptTokens(refreshedState));
          } catch {
            // Ignore sync failures after a user-initiated interruption.
          }
        }
        setError(null);
        setConnection("connected");
        return;
      }

      const message = streamError instanceof Error ? streamError.message : "Streaming request failed.";
      appendAssistantToken(assistantMessageId, `\n\n[Error] ${message}`);
      setError(message);
      setConnection("degraded");
    } finally {
      streamAbortControllerRef.current = null;
      setCanInterrupt(false);
      finishAssistantMessage(assistantMessageId);
      setStreaming(false);
    }
  };

  const stopStreaming = (): void => {
    const controller = streamAbortControllerRef.current;
    if (!controller) {
      return;
    }
    controller.abort();
  };

  const onDrop = (event: React.DragEvent<HTMLDivElement>): void => {
    const reportId = event.dataTransfer.getData("application/x-report-id");
    if (reportId) {
      useBioAgentStore.getState().queueAttachment(reportId);
    }
  };

  return (
    <section className="flex h-full min-h-0 flex-col bg-surface/35">
      <div className="relative flex items-center justify-between border-b border-surface-edge/60 px-4 py-3">
        <h2 className="text-lg font-semibold text-zinc-100">Chat Interface</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void startNewChat()}
            className="inline-flex items-center gap-1 rounded-md border border-accent-cyan/45 bg-accent-cyan/10 px-2 py-1 text-xs text-accent-cyan hover:border-accent-cyan/75 disabled:opacity-50"
            disabled={isStreaming}
          >
            <Plus className="h-3.5 w-3.5" />
            New Chat
          </button>
          <button
            type="button"
            onClick={() => void openThreadMenu()}
            className="inline-flex items-center gap-1 rounded-md border border-surface-edge bg-surface-raised px-2 py-1 text-xs text-zinc-300 hover:border-accent-blue/60 hover:text-zinc-100"
          >
            Thread: {activeThread ? threadDisplayName(activeThread) : "new"}
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
        </div>

        {threadMenuOpen ? (
          <div className="absolute right-4 top-12 z-30 max-h-64 w-80 overflow-y-auto rounded-md border border-surface-edge bg-surface-raised/95 p-1 text-xs shadow-xl backdrop-blur">
            {threadsLoading ? (
              <div className="px-2 py-2 text-zinc-400">Loading threads...</div>
            ) : visibleThreads.length === 0 ? (
              <div className="px-2 py-2 text-zinc-500">No threads found.</div>
            ) : (
              visibleThreads.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => void loadThread(item.id)}
                  className={cn(
                    "mb-1 w-full rounded-md border px-2 py-1.5 text-left hover:border-accent-blue/70 hover:bg-accent-blue/10",
                    item.id === threadId ? "border-accent-blue/60 bg-accent-blue/15 text-zinc-100" : "border-transparent text-zinc-300"
                  )}
                >
                  <div className="truncate font-medium">{threadDisplayName(item)}</div>
                  <div className="text-zinc-500">{shortId(item.id)} • {item.createdAt ?? "unknown time"}</div>
                </button>
              ))
            )}
          </div>
        ) : null}
      </div>

      <div className="min-h-0 flex-1">
        <Virtuoso
          className="h-full px-3 py-2"
          data={messages}
          followOutput="auto"
          itemContent={(_, message) => (
            <div className={cn("mb-3 flex w-full px-3", message.role === "user" ? "justify-end pr-2" : "justify-start pl-1")}>
              <article
                className={cn(
                  "w-fit max-w-[85%] break-words rounded-xl border px-3 py-2 shadow-sm",
                  message.role === "user"
                    ? "border-accent-blue/50 bg-accent-blue/20"
                    : "border-surface-edge bg-surface-raised/80"
                )}
              >
                <div className="markdown-body prose prose-sm prose-invert max-w-none break-words text-zinc-100">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                    {message.content || "..."}
                  </ReactMarkdown>
                </div>
              </article>
            </div>
          )}
        />
      </div>

      <div className="border-t border-surface-edge/60 p-3" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
        {attachmentReports.length > 0 ? (
          <div className="mb-2 flex flex-wrap gap-2">
            {attachmentReports.map((report) => (
              <button
                key={report.id}
                type="button"
                onClick={() => removeAttachment(report.id)}
                className="inline-flex items-center gap-1 rounded-full border border-accent-blue/40 bg-accent-blue/15 px-3 py-1 text-xs text-zinc-100"
                title="Remove attachment"
              >
                <Paperclip className="h-3 w-3" />
                {reportDisplayName(report)}
              </button>
            ))}
          </div>
        ) : null}

        <div className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            rows={2}
            placeholder="Type your message or query..."
            className="min-h-[54px] flex-1 resize-y rounded-xl border border-surface-edge bg-surface-raised px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500 focus:border-accent-blue"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
          />
          {isStreaming && canInterrupt ? (
            <button
              type="button"
              onClick={stopStreaming}
              className="inline-flex h-11 items-center gap-2 rounded-xl bg-status-error px-4 text-sm font-medium text-white transition hover:bg-red-500"
            >
              <Square className="h-4 w-4" />
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void sendMessage()}
              disabled={isStreaming || !draft.trim()}
              className="inline-flex h-11 items-center gap-2 rounded-xl bg-accent-blue px-4 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send className="h-4 w-4" />
              Send
            </button>
          )}
        </div>

        <div className="mt-2 flex items-center justify-end">
          {isStreaming ? (
            <div className="text-xs text-accent-cyan">{canInterrupt ? "Streaming... click Stop to interrupt." : "Working..."}</div>
          ) : (
            <div className="text-xs text-zinc-500">Idle</div>
          )}
        </div>

        {error ? (
          <p className="mt-2 inline-flex items-center gap-1 text-xs text-status-error">
            <AlertTriangle className="h-3.5 w-3.5" />
            {error}
          </p>
        ) : null}
      </div>
    </section>
  );
}
