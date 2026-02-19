import { create } from "zustand";

import { DEFAULT_MODEL_ID, isAllowedCatalogModel } from "@/lib/model-catalog";
import type { ChatMessage, ContextItem, InitialWorkbenchData, ReportFile, StarterPromptCategory, ThreadSummary, ToolEvent } from "@/lib/types";

const MAX_CONTEXT_TOKENS = 128_000;

type ConnectionState = "connected" | "degraded" | "offline";

type ResearchMode = "deep_research" | "quick_answer";

interface BioAgentStore {
  userId: string | null;
  authRequired: boolean;
  backendOk: boolean;
  assistantId: string;
  threadId: string | null;
  model: string;
  mode: ResearchMode;
  reports: ReportFile[];
  starterPromptCategories: StarterPromptCategory[];
  threads: ThreadSummary[];
  selectedReportId: string | null;
  reportContentById: Record<string, string>;
  loadingReportId: string | null;
  messages: ChatMessage[];
  draft: string;
  pendingAttachmentIds: string[];
  contextItems: ContextItem[];
  maxContextTokens: number;
  currentPromptTokens: number | null;
  toolEvents: ToolEvent[];
  isStreaming: boolean;
  connection: ConnectionState;

  hydrate: (data: InitialWorkbenchData) => void;
  setThreadId: (threadId: string | null) => void;
  setModel: (model: string) => void;
  setMode: (mode: ResearchMode) => void;
  setDraft: (draft: string) => void;
  setReports: (reports: ReportFile[]) => void;
  setThreads: (threads: ThreadSummary[]) => void;
  upsertReport: (report: ReportFile) => void;
  selectReport: (reportId: string | null) => void;
  setLoadingReportId: (reportId: string | null) => void;
  cacheReportContent: (reportId: string, content: string) => void;

  queueAttachment: (reportId: string) => void;
  removeAttachment: (reportId: string) => void;
  clearAttachments: () => void;

  addContextItem: (item: ContextItem) => { ok: true } | { ok: false; reason: string };
  clearContextItems: () => void;
  removeContextItem: (itemId: string) => void;
  reorderContextItems: (fromIndex: number, toIndex: number) => void;

  addMessage: (message: ChatMessage) => void;
  setMessages: (messages: ChatMessage[]) => void;
  appendAssistantToken: (messageId: string, chunk: string) => void;
  finishAssistantMessage: (messageId: string) => void;

  pushToolEvent: (event: ToolEvent) => void;
  setToolEvents: (events: ToolEvent[]) => void;
  clearToolEvents: () => void;
  clearThreadWorkspace: () => void;

  setStreaming: (streaming: boolean) => void;
  setConnection: (connection: ConnectionState) => void;
  setCurrentPromptTokens: (tokens: number | null) => void;
}

function moveItem<T>(items: T[], fromIndex: number, toIndex: number): T[] {
  const next = items.slice();
  const [item] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, item);
  return next;
}

export function getContextTokenUsage(state: Pick<BioAgentStore, "contextItems" | "maxContextTokens">): {
  used: number;
  max: number;
} {
  const used = state.contextItems.reduce((acc, item) => acc + item.tokenCount, 0);
  return { used, max: state.maxContextTokens };
}

export const useBioAgentStore = create<BioAgentStore>((set, get) => ({
  userId: null,
  authRequired: false,
  backendOk: false,
  assistantId: process.env.NEXT_PUBLIC_BIOAGENT_ASSISTANT_ID ?? "co_scientist",
  threadId: null,
  model:
    process.env.NEXT_PUBLIC_BIOAGENT_MODEL && isAllowedCatalogModel(process.env.NEXT_PUBLIC_BIOAGENT_MODEL)
      ? process.env.NEXT_PUBLIC_BIOAGENT_MODEL
      : DEFAULT_MODEL_ID,
  mode: "deep_research",
  reports: [],
  starterPromptCategories: [],
  threads: [],
  selectedReportId: null,
  reportContentById: {},
  loadingReportId: null,
  messages: [],
  draft: "",
  pendingAttachmentIds: [],
  contextItems: [],
  maxContextTokens: MAX_CONTEXT_TOKENS,
  currentPromptTokens: null,
  toolEvents: [],
  isStreaming: false,
  connection: "offline",

  hydrate: (data) => {
    set({
      userId: data.userId,
      authRequired: data.authRequired,
      backendOk: data.backendOk,
      reports: [],
      starterPromptCategories: data.starterPromptCategories,
      selectedReportId: null,
      currentPromptTokens: null,
      connection: data.backendOk ? "connected" : "offline"
    });
  },

  setThreadId: (threadId) => set({ threadId }),
  setModel: (model) => set({ model: isAllowedCatalogModel(model) ? model : DEFAULT_MODEL_ID }),
  setMode: (mode) => set({ mode }),
  setDraft: (draft) => set({ draft }),
  setReports: (reports) => {
    set((state) => {
      const selectedExists = state.selectedReportId
        ? reports.some((report) => report.id === state.selectedReportId)
        : false;
      return {
        reports,
        selectedReportId: selectedExists ? state.selectedReportId : null
      };
    });
  },
  setThreads: (threads) => set({ threads }),

  upsertReport: (report) => {
    set((state) => {
      const exists = state.reports.some((item) => item.id === report.id);
      const reports = exists
        ? state.reports.map((item) => (item.id === report.id ? report : item))
        : [report, ...state.reports];
      return { reports };
    });
  },

  selectReport: (reportId) => set({ selectedReportId: reportId }),
  setLoadingReportId: (reportId) => set({ loadingReportId: reportId }),
  cacheReportContent: (reportId, content) => {
    set((state) => ({
      reportContentById: {
        ...state.reportContentById,
        [reportId]: content
      }
    }));
  },

  queueAttachment: (reportId) => {
    set((state) => {
      if (state.pendingAttachmentIds.includes(reportId)) {
        return state;
      }
      return {
        pendingAttachmentIds: [...state.pendingAttachmentIds, reportId]
      };
    });
  },

  removeAttachment: (reportId) => {
    set((state) => ({
      pendingAttachmentIds: state.pendingAttachmentIds.filter((item) => item !== reportId)
    }));
  },

  clearAttachments: () => set({ pendingAttachmentIds: [] }),

  addContextItem: (item) => {
    const state = get();
    const duplicate = state.contextItems.some((existing) => {
      const sameSource = existing.source === item.source;
      const sameRange =
        (!existing.lineRange && !item.lineRange) ||
        (existing.lineRange?.[0] === item.lineRange?.[0] && existing.lineRange?.[1] === item.lineRange?.[1]);
      const sameContent = existing.content.trim() === item.content.trim();
      const sameAttachment =
        (!existing.attachment && !item.attachment) ||
        (
          existing.attachment?.filename === item.attachment?.filename &&
          existing.attachment?.mimeType === item.attachment?.mimeType &&
          existing.attachment?.base64 === item.attachment?.base64
        );
      return sameSource && sameRange && sameContent && sameAttachment;
    });

    if (duplicate) {
      return { ok: false as const, reason: "Selection already in context." };
    }

    const { used, max } = getContextTokenUsage(state);
    if (used + item.tokenCount > max) {
      return { ok: false as const, reason: "Token budget exceeded." };
    }

    set({ contextItems: [...state.contextItems, item] });
    return { ok: true as const };
  },
  clearContextItems: () => set({ contextItems: [] }),

  removeContextItem: (itemId) => {
    set((state) => ({
      contextItems: state.contextItems.filter((item) => item.id !== itemId)
    }));
  },

  reorderContextItems: (fromIndex, toIndex) => {
    set((state) => {
      if (fromIndex < 0 || toIndex < 0 || fromIndex >= state.contextItems.length || toIndex >= state.contextItems.length) {
        return state;
      }
      return {
        contextItems: moveItem(state.contextItems, fromIndex, toIndex)
      };
    });
  },

  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  setMessages: (messages) => set({ messages }),

  appendAssistantToken: (messageId, chunk) => {
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              content: message.content + chunk
            }
          : message
      )
    }));
  },

  finishAssistantMessage: (messageId) => {
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              streaming: false
            }
          : message
      )
    }));
  },

  pushToolEvent: (event) => {
    set((state) => {
      if (event.type === "tool_status" && event.invocationId) {
        const index = state.toolEvents.findIndex(
          (item) => item.type === "tool_status" && item.invocationId === event.invocationId
        );
        if (index >= 0) {
          const nextEvents = state.toolEvents.slice();
          const previous = nextEvents[index];
          const merged: ToolEvent = { ...previous, ...event };
          if (event.argsPreview === undefined) {
            merged.argsPreview = previous.argsPreview;
          }
          nextEvents[index] = merged;
          return { toolEvents: nextEvents };
        }
      }

      return { toolEvents: [event, ...state.toolEvents].slice(0, 200) };
    });
  },
  setToolEvents: (events) => set({ toolEvents: events }),

  clearToolEvents: () => set({ toolEvents: [] }),
  clearThreadWorkspace: () =>
    set({
      reports: [],
      selectedReportId: null,
      reportContentById: {},
      loadingReportId: null,
      messages: [],
      draft: "",
      pendingAttachmentIds: [],
      contextItems: [],
      currentPromptTokens: null,
      toolEvents: []
    }),

  setStreaming: (streaming) => set({ isStreaming: streaming }),
  setConnection: (connection) => set({ connection }),
  setCurrentPromptTokens: (tokens) => set({ currentPromptTokens: tokens })
}));
