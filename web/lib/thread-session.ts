import { getThreadState, listReports, parseThreadMessages, parseThreadPromptTokens, parseThreadToolEvents } from "@/lib/backend-client";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import type { ThreadSummary } from "@/lib/types";

function threadTimestamp(value?: string): number {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function sortThreadsByCreatedAt(threads: ThreadSummary[]): ThreadSummary[] {
  return threads
    .slice()
    .sort((a, b) => threadTimestamp(b.createdAt) - threadTimestamp(a.createdAt));
}

export function prependThread(threads: ThreadSummary[], thread: ThreadSummary): ThreadSummary[] {
  return sortThreadsByCreatedAt([thread, ...threads.filter((item) => item.id !== thread.id)]);
}

export async function loadThreadSession(threadId: string): Promise<void> {
  const store = useBioAgentStore.getState();
  store.setStreaming(true);

  try {
    const [statePayload, threadReports] = await Promise.all([
      getThreadState(threadId),
      listReports({ threadId, limit: 100 })
    ]);
    const restoredMessages = parseThreadMessages(statePayload);
    const restoredToolEvents = parseThreadToolEvents(statePayload);
    const restoredPromptTokens = parseThreadPromptTokens(statePayload);

    store.setThreadId(threadId);
    store.setMessages(restoredMessages);
    store.setCurrentPromptTokens(restoredPromptTokens);
    store.setReports(threadReports);
    store.setToolEvents(restoredToolEvents);
    store.selectReport(null);
    store.clearAttachments();
    store.clearContextItems();
    store.setConnection("connected");
  } finally {
    store.setStreaming(false);
  }
}
