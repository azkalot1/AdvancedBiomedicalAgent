"use client";

import { AlertTriangle, ChevronDown, Paperclip, Plus, RefreshCcw, Send, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Virtuoso } from "react-virtuoso";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

import {
  createThread,
  generateThreadDisplayName,
  getThreadState,
  getTokenChunkFromEvent,
  listReports,
  listThreads,
  logStarterPromptClick,
  parseThreadResponseFeedback,
  parseThreadPromptTokens,
  setThreadDisplayName,
  setThreadResponseFeedback,
  summarizeThreadContext,
  streamRun
} from "@/lib/backend-client";
import { reportDisplayName, threadDisplayName } from "@/lib/display-names";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { loadThreadSession, prependThread, sortThreadsByCreatedAt } from "@/lib/thread-session";
import type {
  ChatMessage,
  ContextItem,
  NormalizedStreamEvent,
  ResponseFeedbackReason,
  StarterPromptCategory,
  ThreadSummary,
  ToolEvent
} from "@/lib/types";
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
      timestamp: typeof payload.timestamp === "string" ? payload.timestamp : new Date().toISOString(),
      message: typeof payload.message === "string" ? payload.message : undefined,
      reason: typeof payload.reason === "string" ? payload.reason : undefined,
      argsPreview: normalizeArgsPreview(
        isRecord(payload)
          ? {
              ...(typeof payload.model_name === "string" ? { model_name: payload.model_name } : {}),
              ...(typeof payload.context_window_tokens === "number"
                ? { context_window_tokens: payload.context_window_tokens }
                : {}),
              ...(typeof payload.trigger_tokens === "number" ? { trigger_tokens: payload.trigger_tokens } : {}),
              ...(typeof payload.keep_tokens === "number" ? { keep_tokens: payload.keep_tokens } : {})
            }
          : undefined
      )
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

interface StarterPromptSuggestion {
  categoryId: string;
  categoryTitle: string;
  prompt: string;
}

function shuffleItems<T>(items: T[]): T[] {
  const output = items.slice();
  for (let index = output.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [output[index], output[swapIndex]] = [output[swapIndex], output[index]];
  }
  return output;
}

function pickStarterPromptSuggestions(
  categories: StarterPromptCategory[],
  categoryCount = 5
): StarterPromptSuggestion[] {
  const validCategories = categories.filter((category) => category.prompts.length > 0);
  if (validCategories.length === 0) {
    return [];
  }

  return shuffleItems(validCategories)
    .slice(0, Math.min(categoryCount, validCategories.length))
    .map((category) => {
      const prompt = category.prompts[Math.floor(Math.random() * category.prompts.length)];
      return {
        categoryId: category.id,
        categoryTitle: category.title,
        prompt
      };
    });
}

const FEEDBACK_REASON_OPTIONS: Array<{ value: ResponseFeedbackReason; label: string }> = [
  { value: "wrong_or_outdated", label: "The data was wrong or outdated" },
  { value: "did_not_address_question", label: "The answer didn't address my question" },
  { value: "missing_information", label: "Missing information I needed" },
  { value: "too_much_irrelevant_detail", label: "Too much irrelevant detail" },
  { value: "unclear_reliability", label: "I couldn't tell if the answer was reliable" }
];

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
  const toolEvents = useBioAgentStore((state) => state.toolEvents);
  const setConnection = useBioAgentStore((state) => state.setConnection);
  const setCurrentPromptTokens = useBioAgentStore((state) => state.setCurrentPromptTokens);
  const starterPromptCategories = useBioAgentStore((state) => state.starterPromptCategories);

  const [error, setError] = useState<string | null>(null);
  const [threadMenuOpen, setThreadMenuOpen] = useState(false);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [manualSummarizing, setManualSummarizing] = useState(false);
  const [canInterrupt, setCanInterrupt] = useState(false);
  const [openReasonMenuTurn, setOpenReasonMenuTurn] = useState<string | null>(null);
  const [feedbackSavingByTurn, setFeedbackSavingByTurn] = useState<Record<string, boolean>>({});
  const [feedbackErrorByTurn, setFeedbackErrorByTurn] = useState<Record<string, string | null>>({});
  const [starterPromptSuggestions, setStarterPromptSuggestions] = useState<StarterPromptSuggestion[]>([]);
  const streamAbortControllerRef = useRef<AbortController | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

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

  const feedbackByTurn = useMemo(
    () => parseThreadResponseFeedback(activeThread?.metadata),
    [activeThread]
  );

  const assistantTurnByMessageId = useMemo(() => {
    const map = new Map<string, { turnNumber: number; turnKey: string }>();
    let assistantCount = 0;
    for (const message of messages) {
      if (message.role !== "assistant") {
        continue;
      }
      assistantCount += 1;
      map.set(message.id, {
        turnNumber: assistantCount,
        turnKey: `assistant_turn_${assistantCount}`
      });
    }
    return map;
  }, [messages]);

  const latestContextNoticeEvent = useMemo(
    () => toolEvents.find((event) => event.type === "context_updated" && typeof event.message === "string"),
    [toolEvents]
  );

  const regenerateStarterPromptSuggestions = (): void => {
    setStarterPromptSuggestions(pickStarterPromptSuggestions(starterPromptCategories, 5));
  };

  useEffect(() => {
    setStarterPromptSuggestions(pickStarterPromptSuggestions(starterPromptCategories, 5));
  }, [starterPromptCategories]);

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

  const manuallySummarizeContext = async (): Promise<void> => {
    if (!threadId || isStreaming || manualSummarizing) {
      return;
    }

    setError(null);
    setManualSummarizing(true);
    try {
      const summaryResult = await summarizeThreadContext(threadId, model);
      await loadThreadSession(threadId);
      if (summaryResult.summarized) {
        pushToolEvent({
          id: `manual-context-summary-${Date.now()}`,
          type: "context_updated",
          timestamp: new Date().toISOString(),
          reason: "conversation_summarized",
          message: summaryResult.message,
          argsPreview: {
            model_name: summaryResult.model_name,
            context_window_tokens: summaryResult.context_window_tokens,
            trigger_tokens: summaryResult.trigger_tokens,
            keep_tokens: summaryResult.keep_tokens,
            source: "manual"
          }
        });
      }
      setConnection("connected");
    } catch (summaryError) {
      const message = summaryError instanceof Error ? summaryError.message : "Failed to summarize thread context.";
      setError(message);
      setConnection("degraded");
    } finally {
      setManualSummarizing(false);
    }
  };

  const loadThread = async (selectedThreadId: string): Promise<void> => {
    setError(null);
    setThreadMenuOpen(false);
    setOpenReasonMenuTurn(null);
    setFeedbackSavingByTurn({});
    setFeedbackErrorByTurn({});
    try {
      await loadThreadSession(selectedThreadId);
    } catch (threadError) {
      const message = threadError instanceof Error ? threadError.message : "Failed to load thread.";
      setError(message);
      setConnection("degraded");
    }
  };

  const startNewChat = (): void => {
    if (isStreaming) {
      return;
    }
    setError(null);
    setThreadMenuOpen(false);
    setOpenReasonMenuTurn(null);
    setFeedbackSavingByTurn({});
    setFeedbackErrorByTurn({});
    clearThreadWorkspace();
    setThreadId(null);
    regenerateStarterPromptSuggestions();
    setConnection("connected");
  };

  const chooseStarterPrompt = (suggestion: StarterPromptSuggestion): void => {
    setDraft(suggestion.prompt);
    if (composerRef.current) {
      composerRef.current.focus();
    }
    void logStarterPromptClick({
      promptText: suggestion.prompt,
      categoryId: suggestion.categoryId,
      categoryTitle: suggestion.categoryTitle,
      threadId
    }).catch(() => undefined);
  };

  const recordResponseFeedback = async (
    messageId: string,
    helpful: boolean,
    reason?: ResponseFeedbackReason
  ): Promise<void> => {
    if (!threadId) {
      return;
    }

    const turnMeta = assistantTurnByMessageId.get(messageId);
    if (!turnMeta) {
      return;
    }
    const turnKey = turnMeta.turnKey;

    setOpenReasonMenuTurn(null);
    setFeedbackSavingByTurn((prev) => ({ ...prev, [turnKey]: true }));
    setFeedbackErrorByTurn((prev) => ({ ...prev, [turnKey]: null }));

    try {
      const persistedMetadata = await setThreadResponseFeedback({
        threadId,
        assistantTurnKey: turnKey,
        helpful,
        ...(helpful ? {} : { reason }),
        messageId,
        userId
      });
      const state = useBioAgentStore.getState();
      state.setThreads(
        state.threads.map((item) =>
          item.id === threadId
            ? {
                ...item,
                metadata: { ...(item.metadata ?? {}), ...persistedMetadata }
              }
            : item
        )
      );
      setConnection("connected");
    } catch (feedbackError) {
      const message = feedbackError instanceof Error ? feedbackError.message : "Failed to save feedback.";
      setFeedbackErrorByTurn((prev) => ({ ...prev, [turnKey]: message }));
      setConnection("degraded");
    } finally {
      setFeedbackSavingByTurn((prev) => ({ ...prev, [turnKey]: false }));
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

  const feedbackReasonLabel = (reason?: ResponseFeedbackReason): string | null => {
    if (!reason) {
      return null;
    }
    const option = FEEDBACK_REASON_OPTIONS.find((item) => item.value === reason);
    return option ? option.label : null;
  };

  const shouldRenderFeedbackPrompt = (message: ChatMessage): boolean => {
    if (message.role !== "assistant" || message.streaming) {
      return false;
    }
    const content = message.content.trim();
    if (!content) {
      return false;
    }
    if (content.startsWith("[Generation interrupted]")) {
      return false;
    }
    return true;
  };

  return (
    <section className="flex h-full min-h-0 flex-col bg-surface/35">
      <div className="relative flex items-center justify-between border-b border-surface-edge/60 px-4 py-3">
        <h2 className="text-lg font-semibold text-zinc-100">Chat Interface</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void manuallySummarizeContext()}
            className="inline-flex items-center gap-1 rounded-md border border-amber-400/45 bg-amber-400/10 px-2 py-1 text-xs text-amber-200 hover:border-amber-300/80 disabled:opacity-50"
            disabled={!threadId || isStreaming || manualSummarizing}
            title="Manually summarize conversation context"
          >
            <RefreshCcw className={cn("h-3.5 w-3.5", manualSummarizing ? "animate-spin" : "")} />
            Summarize Context
          </button>
          <button
            type="button"
            onClick={startNewChat}
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
        {latestContextNoticeEvent?.message ? (
          <div className="border-b border-amber-500/25 bg-amber-500/8 px-4 py-2 text-xs text-amber-100">
            {latestContextNoticeEvent.message}
          </div>
        ) : null}
        {messages.length === 0 ? (
          <div className="h-full overflow-y-auto px-4 py-5">
            <p className="mb-3 text-sm font-medium text-zinc-200">Example prompts</p>
            {starterPromptSuggestions.length === 0 ? (
              <p className="text-xs text-zinc-500">Prompt library is unavailable.</p>
            ) : (
              <div className="grid gap-2">
                {starterPromptSuggestions.map((item) => (
                  <button
                    key={`${item.categoryId}-${item.prompt}`}
                    type="button"
                    onClick={() => chooseStarterPrompt(item)}
                    className="rounded-lg border border-surface-edge bg-surface-raised/75 px-3 py-2 text-left text-xs text-zinc-300 transition hover:border-accent-blue/60 hover:text-zinc-100"
                  >
                    <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">{item.categoryTitle}</div>
                    <div>{item.prompt}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <Virtuoso
            className="h-full px-3 py-2"
            data={messages}
            followOutput="auto"
            itemContent={(_, message) => {
              const turnMeta = assistantTurnByMessageId.get(message.id);
              const turnKey = turnMeta?.turnKey;
              const feedback = turnKey ? feedbackByTurn[turnKey] : undefined;
              const isSaving = turnKey ? feedbackSavingByTurn[turnKey] ?? false : false;
              const feedbackError = turnKey ? feedbackErrorByTurn[turnKey] : null;
              const reasonOpen = turnKey ? openReasonMenuTurn === turnKey : false;
              const reasonLabel = feedbackReasonLabel(feedback?.reason);

              return (
                <div className={cn("mb-3 flex w-full px-3", message.role === "user" ? "justify-end pr-2" : "justify-start pl-1")}>
                  <div className="w-fit max-w-[88%]">
                    <article
                      className={cn(
                        "break-words rounded-xl border px-3 py-2 shadow-sm",
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

                    {shouldRenderFeedbackPrompt(message) && turnKey ? (
                      <div className="mt-2 rounded-lg border border-surface-edge/70 bg-surface-raised/70 px-3 py-2 text-xs text-zinc-300">
                        <p className="mb-2 text-zinc-200">Was this response useful for your work?</p>
                        {feedback ? (
                          <p className="text-zinc-400">
                            Feedback saved: {feedback.helpful ? "Yes" : "No"}
                            {reasonLabel ? ` (${reasonLabel})` : ""}
                          </p>
                        ) : (
                          <div className="space-y-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <button
                                type="button"
                                onClick={() => void recordResponseFeedback(message.id, true)}
                                disabled={isSaving}
                                className="rounded border border-emerald-500/50 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-200 hover:border-emerald-400/80 disabled:opacity-50"
                              >
                                Yes
                              </button>
                              <button
                                type="button"
                                onClick={() => setOpenReasonMenuTurn(reasonOpen ? null : turnKey)}
                                disabled={isSaving}
                                className="rounded border border-amber-500/45 bg-amber-500/10 px-2 py-1 text-xs text-amber-200 hover:border-amber-400/80 disabled:opacity-50"
                              >
                                No, tell us why ▾
                              </button>
                              {isSaving ? <span className="text-zinc-500">Saving...</span> : null}
                            </div>

                            {reasonOpen ? (
                              <div className="space-y-1 rounded border border-surface-edge/70 bg-surface/60 p-2">
                                {FEEDBACK_REASON_OPTIONS.map((option) => (
                                  <button
                                    key={option.value}
                                    type="button"
                                    onClick={() => {
                                      setOpenReasonMenuTurn(null);
                                      void recordResponseFeedback(message.id, false, option.value);
                                    }}
                                    disabled={isSaving}
                                    className="block w-full rounded px-2 py-1 text-left text-zinc-300 hover:bg-surface-overlay hover:text-zinc-100 disabled:opacity-50"
                                  >
                                    ○ {option.label}
                                  </button>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        )}

                        {feedbackError ? (
                          <p className="mt-1 inline-flex items-center gap-1 text-[11px] text-status-error">
                            <AlertTriangle className="h-3 w-3" />
                            {feedbackError}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              );
            }}
          />
        )}
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
            ref={composerRef}
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
