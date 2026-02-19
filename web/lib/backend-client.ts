import type {
  ContextItem,
  MeResponse,
  NormalizedStreamEvent,
  ReportContent,
  ReportFile,
  ReportsListResponse,
  ThreadSummary,
  ChatMessage,
  ToolEvent
} from "@/lib/types";
import { parseEventStream } from "@/lib/sse";

const PROXY_BASE = "/api/backend";

interface ApiErrorEnvelope {
  error?: {
    code?: string;
    message?: string;
    request_id?: string;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function toReportStatus(raw?: string): ReportFile["status"] {
  if (raw === "error") {
    return "error";
  }
  if (raw === "generating" || raw === "running" || raw === "queued") {
    return "generating";
  }
  return "complete";
}

export function mapReport(raw: ReportsListResponse["items"][number]): ReportFile {
  return {
    id: raw.id,
    refId: raw.ref_id,
    filename: raw.filename ?? `${raw.id}.md`,
    displayName: raw.display_name ?? raw.one_line ?? raw.filename ?? `${raw.id}.md`,
    toolName: raw.tool_name,
    status: toReportStatus(raw.status),
    sizeChars: raw.size_chars,
    oneLine: raw.one_line,
    threadId: raw.thread_id,
    userId: raw.user_id,
    createdAt: raw.created_at,
    path: raw.path
  };
}

async function parseApiError(response: Response): Promise<never> {
  let body: ApiErrorEnvelope | null = null;
  try {
    body = (await response.json()) as ApiErrorEnvelope;
  } catch {
    body = null;
  }

  const message = body?.error?.message ?? response.statusText ?? "Request failed";
  const code = body?.error?.code ? `[${body.error.code}] ` : "";
  throw new Error(`${code}${response.status} ${message}`);
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${PROXY_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store"
  });

  if (!response.ok) {
    await parseApiError(response);
  }

  return (await response.json()) as T;
}

export async function getMe(): Promise<MeResponse> {
  return fetchJson<MeResponse>("/v1/me", { method: "GET" });
}

function asThreadArray(payload: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(payload)) {
    return payload.filter((item): item is Record<string, unknown> => isRecord(item));
  }
  if (isRecord(payload)) {
    for (const key of ["threads", "items", "data"]) {
      const value = payload[key];
      if (Array.isArray(value)) {
        return value.filter((item): item is Record<string, unknown> => isRecord(item));
      }
    }
  }
  return [];
}

function ownerIdFromThreadRecord(item: Record<string, unknown>, metadata?: Record<string, unknown>): string | null {
  // Prefer app metadata user scope. Platform-level owner/user fields may reflect
  // infra auth identity (e.g., "anonymous") and not the app user.
  if (metadata) {
    const fromMetadata =
      typeof metadata.user_id === "string" && metadata.user_id.trim()
        ? metadata.user_id.trim()
        : typeof metadata.owner_id === "string" && metadata.owner_id.trim()
          ? metadata.owner_id.trim()
          : typeof metadata.owner === "string" && metadata.owner.trim()
            ? metadata.owner.trim()
            : null;
    if (fromMetadata) {
      return fromMetadata;
    }
  }

  const direct =
    typeof item.user_id === "string" && item.user_id.trim()
      ? item.user_id.trim()
      : typeof item.owner_id === "string" && item.owner_id.trim()
        ? item.owner_id.trim()
        : typeof item.owner === "string" && item.owner.trim()
          ? item.owner.trim()
          : null;
  return direct;
}

export async function listThreads(limit = 50, userId?: string | null): Promise<ThreadSummary[]> {
  const url = `/threads?limit=${limit}`;
  const primaryResponse = await fetch(`${PROXY_BASE}${url}`, {
    method: "GET",
    cache: "no-store"
  });

  let payload: unknown;
  if (primaryResponse.ok) {
    payload = await primaryResponse.json();
  } else if (primaryResponse.status === 405) {
    const fallbackResponse = await fetch(`${PROXY_BASE}/threads/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ limit }),
      cache: "no-store"
    });
    if (!fallbackResponse.ok) {
      await parseApiError(fallbackResponse);
    }
    payload = await fallbackResponse.json();
  } else {
    await parseApiError(primaryResponse);
  }

  const output: ThreadSummary[] = [];
  for (const item of asThreadArray(payload)) {
    const idRaw = item.thread_id ?? item.id;
    if (typeof idRaw !== "string" || !idRaw.trim()) {
      continue;
    }
    const metadata = isRecord(item.metadata) ? item.metadata : undefined;
    const ownerId = ownerIdFromThreadRecord(item, metadata);
    const scopedUserId = typeof userId === "string" ? userId.trim() : "";
    if (scopedUserId) {
      // Keep threads when owner metadata is absent; only drop explicit mismatches.
      if (ownerId && ownerId !== scopedUserId) {
        continue;
      }
    }
    const metadataDisplayName =
      metadata && typeof metadata.display_name === "string" && metadata.display_name.trim()
        ? metadata.display_name.trim()
        : metadata && typeof metadata.thread_display_name === "string" && metadata.thread_display_name.trim()
          ? metadata.thread_display_name.trim()
          : undefined;
    const topLevelDisplayName =
      typeof item.display_name === "string" && item.display_name.trim()
        ? item.display_name.trim()
        : typeof item.thread_display_name === "string" && item.thread_display_name.trim()
          ? item.thread_display_name.trim()
          : undefined;
    output.push({
      id: idRaw,
      displayName: topLevelDisplayName ?? metadataDisplayName,
      createdAt: typeof item.created_at === "string" ? item.created_at : undefined,
      metadata
    });
  }
  return output;
}

interface ThreadDisplayNameMessageInput {
  role: "user" | "assistant";
  content: string;
}

interface ThreadDisplayNameResponse {
  thread_id: string;
  display_name: string;
  generated: boolean;
}

interface ThreadRecordResponse {
  id?: string;
  thread_id?: string;
  metadata?: Record<string, unknown>;
}

async function getThreadRecord(threadId: string): Promise<ThreadRecordResponse | null> {
  const response = await fetch(`${PROXY_BASE}/threads/${encodeURIComponent(threadId)}`, {
    method: "GET",
    cache: "no-store"
  });
  if (!response.ok) {
    return null;
  }
  const payload = (await response.json()) as unknown;
  if (!isRecord(payload)) {
    return null;
  }
  return payload as ThreadRecordResponse;
}

async function sendThreadMetadataUpdate(
  threadId: string,
  metadata: Record<string, unknown>,
  method: "PATCH" | "PUT" | "POST"
): Promise<boolean> {
  const response = await fetch(`${PROXY_BASE}/threads/${encodeURIComponent(threadId)}`, {
    method,
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ metadata }),
    cache: "no-store"
  });
  if (response.ok) {
    return true;
  }
  if (response.status === 404 || response.status === 405) {
    return false;
  }
  await parseApiError(response);
  throw new Error("Thread metadata update failed.");
}

export async function setThreadDisplayName(
  threadId: string,
  displayName: string,
  userId?: string | null
): Promise<Record<string, unknown>> {
  const existing = await getThreadRecord(threadId);
  const existingMetadata = isRecord(existing?.metadata) ? existing.metadata : {};
  const mergedMetadata: Record<string, unknown> = {
    ...existingMetadata,
    display_name: displayName,
    thread_display_name: displayName,
    ...(userId ? { user_id: userId } : {})
  };

  if (await sendThreadMetadataUpdate(threadId, mergedMetadata, "PATCH")) {
    return mergedMetadata;
  }
  if (await sendThreadMetadataUpdate(threadId, mergedMetadata, "PUT")) {
    return mergedMetadata;
  }
  if (await sendThreadMetadataUpdate(threadId, mergedMetadata, "POST")) {
    return mergedMetadata;
  }
  throw new Error("Backend does not support thread metadata updates.");
}

export async function generateThreadDisplayName(
  threadId: string,
  messages: ThreadDisplayNameMessageInput[],
  options?: { minMessages?: number; maxMessages?: number; force?: boolean }
): Promise<ThreadDisplayNameResponse> {
  return fetchJson<ThreadDisplayNameResponse>(`/v1/threads/${encodeURIComponent(threadId)}/display-name/generate`, {
    method: "POST",
    body: JSON.stringify({
      messages,
      min_messages: options?.minMessages ?? 3,
      max_messages: options?.maxMessages ?? 6,
      force: options?.force ?? false
    })
  });
}

export async function listReports(params: { threadId?: string | null; limit?: number; offset?: number } = {}): Promise<ReportFile[]> {
  const query = new URLSearchParams();
  query.set("limit", String(params.limit ?? 100));
  query.set("offset", String(params.offset ?? 0));
  if (params.threadId) {
    query.set("thread_id", params.threadId);
  }

  const payload = await fetchJson<ReportsListResponse>(`/v1/reports?${query.toString()}`, { method: "GET" });
  return payload.items.map(mapReport);
}

export async function getReportContent(reportId: string, maxChars = 100_000): Promise<ReportContent> {
  const query = new URLSearchParams({ max_chars: String(maxChars) });
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20_000);

  let response: {
    id: string;
    filename: string;
    content: string;
    total_chars: number;
    returned_chars: number;
    offset: number;
    max_chars: number;
    truncated: boolean;
  };

  try {
    response = await fetchJson<{
      id: string;
      filename: string;
      content: string;
      total_chars: number;
      returned_chars: number;
      offset: number;
      max_chars: number;
      truncated: boolean;
    }>(`/v1/reports/${encodeURIComponent(reportId)}/content?${query.toString()}`, {
      method: "GET",
      signal: controller.signal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Timed out while loading report content.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  return {
    id: response.id,
    filename: response.filename,
    content: response.content,
    totalChars: response.total_chars,
    returnedChars: response.returned_chars,
    offset: response.offset,
    maxChars: response.max_chars,
    truncated: response.truncated
  };
}

export async function createThread(userId?: string | null): Promise<string> {
  const response = await fetchJson<{ thread_id?: string; id?: string }>("/threads", {
    method: "POST",
    body: JSON.stringify({
      metadata: {
        app: "co-scientist",
        ...(userId ? { user_id: userId } : {})
      }
    })
  });

  const threadId = response.thread_id ?? response.id;
  if (!threadId) {
    throw new Error("Server did not return thread_id.");
  }
  return threadId;
}

function extractText(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }
  if (Array.isArray(payload)) {
    return payload.map((entry) => extractText(entry)).join("");
  }
  if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    if (typeof obj.text === "string") {
      return obj.text;
    }
    if ("content" in obj) {
      return extractText(obj.content);
    }
    if ("chunk" in obj) {
      return extractText(obj.chunk);
    }
    if ("message" in obj) {
      return extractText(obj.message);
    }
  }
  return "";
}

function normalizeRole(raw: unknown): ChatMessage["role"] | null {
  if (typeof raw !== "string") {
    return null;
  }
  const role = raw.toLowerCase();
  if (role === "human" || role === "user") {
    return "user";
  }
  if (role === "assistant" || role === "ai") {
    return "assistant";
  }
  return null;
}

export function parseThreadMessages(statePayload: Record<string, unknown>): ChatMessage[] {
  const values = isRecord(statePayload.values) ? statePayload.values : statePayload;
  const messages = Array.isArray(values.messages) ? values.messages : [];

  const parsed: ChatMessage[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const item = messages[index];
    if (!isRecord(item)) {
      continue;
    }
    const role = normalizeRole(item.type ?? item.role);
    if (!role) {
      continue;
    }
    const content = extractText(item.content).trim();
    if (!content) {
      continue;
    }
    const idRaw = item.id ?? item.message_id;
    const createdAtRaw = item.created_at ?? item.updated_at ?? item.timestamp;
    parsed.push({
      id: typeof idRaw === "string" && idRaw ? idRaw : `msg-${index}`,
      role,
      content,
      createdAt: typeof createdAtRaw === "string" && createdAtRaw ? createdAtRaw : new Date().toISOString(),
      streaming: false
    });
  }
  return parsed;
}

function parsePromptTokensFromMessageRecord(item: Record<string, unknown>): number | null {
  const asNonNegativeInteger = (value: unknown): number | null => {
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      return Math.floor(value);
    }
    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed) && parsed >= 0) {
        return Math.floor(parsed);
      }
    }
    return null;
  };

  const firstNumeric = (...values: unknown[]): number | null => {
    for (const value of values) {
      const parsed = asNonNegativeInteger(value);
      if (parsed !== null) {
        return parsed;
      }
    }
    return null;
  };

  const metadata = isRecord(item.response_metadata)
    ? item.response_metadata
    : isRecord(item.message) && isRecord((item.message as Record<string, unknown>).response_metadata)
      ? ((item.message as Record<string, unknown>).response_metadata as Record<string, unknown>)
      : null;

  const usageMetadata = isRecord(item.usage_metadata)
    ? item.usage_metadata
    : isRecord(item.message) && isRecord((item.message as Record<string, unknown>).usage_metadata)
      ? ((item.message as Record<string, unknown>).usage_metadata as Record<string, unknown>)
      : null;

  if (metadata) {
    const tokenUsage = isRecord(metadata.token_usage) ? metadata.token_usage : null;
    const usage = isRecord(metadata.usage) ? metadata.usage : null;
    const parsed = firstNumeric(
      tokenUsage?.total_tokens,
      tokenUsage?.totalTokens,
      tokenUsage?.prompt_tokens,
      tokenUsage?.promptTokens,
      tokenUsage?.input_tokens,
      tokenUsage?.inputTokens,
      usage?.total_tokens,
      usage?.totalTokens,
      usage?.prompt_tokens,
      usage?.promptTokens,
      usage?.input_tokens,
      usage?.inputTokens,
      metadata.total_tokens,
      metadata.totalTokens,
      metadata.prompt_tokens,
      metadata.promptTokens,
      metadata.input_tokens,
      metadata.inputTokens
    );
    if (parsed !== null) {
      return parsed;
    }
  }

  if (usageMetadata) {
    const parsed = firstNumeric(
      usageMetadata.total_tokens,
      usageMetadata.totalTokens,
      usageMetadata.prompt_tokens,
      usageMetadata.promptTokens,
      usageMetadata.input_tokens,
      usageMetadata.inputTokens
    );
    if (parsed !== null) {
      return parsed;
    }
  }

  if (isRecord(item.token_usage)) {
    const parsed = firstNumeric(
      (item.token_usage as Record<string, unknown>).total_tokens,
      (item.token_usage as Record<string, unknown>).totalTokens,
      (item.token_usage as Record<string, unknown>).prompt_tokens,
      (item.token_usage as Record<string, unknown>).promptTokens,
      (item.token_usage as Record<string, unknown>).input_tokens,
      (item.token_usage as Record<string, unknown>).inputTokens
    );
    if (parsed !== null) {
      return parsed;
    }
  }

  return null;
}

export function parseThreadPromptTokens(statePayload: Record<string, unknown>): number | null {
  const values = isRecord(statePayload.values) ? statePayload.values : statePayload;
  const messages = Array.isArray(values.messages) ? values.messages : [];

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const item = messages[index];
    if (!isRecord(item)) {
      continue;
    }
    const tokens = parsePromptTokensFromMessageRecord(item);
    if (tokens !== null) {
      return tokens;
    }
  }

  let fallback: number | null = null;
  function walk(value: unknown, depth = 0): void {
    if (fallback !== null || depth > 8 || value == null) {
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        walk(item, depth + 1);
        if (fallback !== null) {
          return;
        }
      }
      return;
    }
    if (!isRecord(value)) {
      return;
    }

    const parsed = parsePromptTokensFromMessageRecord(value);
    if (parsed !== null) {
      fallback = parsed;
      return;
    }

    for (const nested of Object.values(value)) {
      walk(nested, depth + 1);
      if (fallback !== null) {
        return;
      }
    }
  }

  walk(statePayload);
  if (fallback !== null) {
    return fallback;
  }

  return null;
}

function isToolRole(raw: unknown): boolean {
  return typeof raw === "string" && raw.toLowerCase() === "tool";
}

function parseToolArgs(raw: unknown): Record<string, unknown> | undefined {
  if (isRecord(raw)) {
    return raw;
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) {
      return undefined;
    }
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (isRecord(parsed)) {
        return parsed;
      }
      return { value: parsed };
    } catch {
      return { raw: trimmed };
    }
  }
  if (raw == null) {
    return undefined;
  }
  return { value: String(raw) };
}

function eventTimestampFromMessage(item: Record<string, unknown>): string {
  const createdAtRaw = item.created_at ?? item.updated_at ?? item.timestamp;
  return typeof createdAtRaw === "string" && createdAtRaw ? createdAtRaw : new Date().toISOString();
}

export function parseThreadToolEvents(statePayload: Record<string, unknown>): ToolEvent[] {
  const values = isRecord(statePayload.values) ? statePayload.values : statePayload;
  const messages = Array.isArray(values.messages) ? values.messages : [];

  const events: ToolEvent[] = [];

  for (let index = 0; index < messages.length; index += 1) {
    const item = messages[index];
    if (!isRecord(item)) {
      continue;
    }

    const messageTimestamp = eventTimestampFromMessage(item);
    const toolCalls = Array.isArray(item.tool_calls) ? item.tool_calls : [];
    for (let callIndex = 0; callIndex < toolCalls.length; callIndex += 1) {
      const toolCall = toolCalls[callIndex];
      if (!isRecord(toolCall)) {
        continue;
      }

      const toolNameRaw = toolCall.name ?? (isRecord(toolCall.function) ? toolCall.function.name : undefined);
      const toolName = typeof toolNameRaw === "string" && toolNameRaw ? toolNameRaw : "unknown_tool";
      const invocationIdRaw = toolCall.id ?? toolCall.tool_call_id;
      const invocationId = typeof invocationIdRaw === "string" && invocationIdRaw ? invocationIdRaw : undefined;
      const argsRaw = toolCall.args ?? toolCall.arguments ?? (isRecord(toolCall.function) ? toolCall.function.arguments : undefined);

      events.push({
        id: `state-tool-call-${index}-${callIndex}-${invocationId ?? toolName}`,
        type: "tool_status",
        toolName,
        status: "success",
        timestamp: messageTimestamp,
        invocationId,
        argsPreview: parseToolArgs(argsRaw)
      });
    }

    if (isToolRole(item.type ?? item.role)) {
      const toolName = typeof item.name === "string" && item.name ? item.name : "unknown_tool";
      const invocationId = typeof item.tool_call_id === "string" && item.tool_call_id ? item.tool_call_id : undefined;
      const content = extractText(item.content).trim();

      events.push({
        id: `state-tool-result-${index}-${invocationId ?? toolName}`,
        type: "tool_status",
        toolName,
        status: "success",
        timestamp: messageTimestamp,
        invocationId,
        ...(content ? { argsPreview: { result_preview: content.slice(0, 500) } } : {})
      });
    }
  }

  const deduped = new Map<string, ToolEvent>();
  for (const event of events) {
    const key = event.invocationId ? `invocation:${event.invocationId}` : `${event.toolName ?? "unknown"}:${event.id}`;
    if (!deduped.has(key)) {
      deduped.set(key, event);
    }
  }

  return Array.from(deduped.values()).sort((a, b) => {
    const t1 = Date.parse(a.timestamp);
    const t2 = Date.parse(b.timestamp);
    if (Number.isNaN(t1) || Number.isNaN(t2)) {
      return 0;
    }
    return t2 - t1;
  });
}

function inferRole(value: unknown): string | null {
  if (!isRecord(value)) {
    return null;
  }

  const directRole = value.role ?? value.type ?? value.message_type;
  if (typeof directRole === "string" && directRole.trim()) {
    return directRole.toLowerCase();
  }

  if (isRecord(value.message)) {
    return inferRole(value.message);
  }
  if (isRecord(value.chunk)) {
    return inferRole(value.chunk);
  }

  return null;
}

function shouldIncludeChunk(payload: unknown, metadata: unknown): boolean {
  if (isRecord(metadata)) {
    const node = metadata.langgraph_node;
    if (typeof node === "string" && node.toLowerCase().includes("tool")) {
      return false;
    }
  }

  const role = inferRole(payload);
  if (!role) {
    return true;
  }
  if (role.includes("tool") || role.includes("human") || role.includes("user")) {
    return false;
  }
  if (role.includes("ai") || role.includes("assistant")) {
    return true;
  }
  return true;
}

function normalizeStreamEvent(eventName: string, eventData: unknown): NormalizedStreamEvent {
  let mode = eventName;
  let payload = eventData;

  if (eventName === "message" && eventData && typeof eventData === "object") {
    const eventObject = eventData as Record<string, unknown>;
    const nestedMode = eventObject.event ?? eventObject.mode ?? eventObject.stream_mode;
    const nestedPayload = eventObject.data ?? eventObject.payload;
    if (typeof nestedMode === "string") {
      mode = nestedMode;
    }
    if (nestedPayload !== undefined) {
      payload = nestedPayload;
    }
  }

  return { mode, payload };
}

export function getPromptTokensFromStreamEvent(event: NormalizedStreamEvent): number | null {
  let found: number | null = null;

  function walk(value: unknown, depth = 0): void {
    if (found !== null || depth > 8 || value == null) {
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        walk(item, depth + 1);
        if (found !== null) {
          return;
        }
      }
      return;
    }
    if (!isRecord(value)) {
      return;
    }

    const direct = parsePromptTokensFromMessageRecord(value);
    if (direct !== null) {
      found = direct;
      return;
    }
    if (isRecord(value.response_metadata)) {
      const nested = parsePromptTokensFromMessageRecord({ response_metadata: value.response_metadata });
      if (nested !== null) {
        found = nested;
        return;
      }
    }

    for (const nestedValue of Object.values(value)) {
      walk(nestedValue, depth + 1);
      if (found !== null) {
        return;
      }
    }
  }

  walk(event.payload);
  return found;
}

export function getTokenChunkFromEvent(event: NormalizedStreamEvent): string {
  if (event.mode === "messages" || event.mode === "messages-tuple" || event.mode === "messages_tuple") {
    const payloadTuple = Array.isArray(event.payload) ? event.payload : [event.payload];
    const payload = payloadTuple[0];
    const metadata = payloadTuple.length > 1 ? payloadTuple[1] : null;
    if (!shouldIncludeChunk(payload, metadata)) {
      return "";
    }
    const text = extractText(payload);
    const trimmed = text.trimStart();
    if (
      trimmed.startsWith("[RESULTS]") ||
      trimmed.startsWith("[AGENT_SIGNALS]") ||
      trimmed.includes("Related searches: ->")
    ) {
      return "";
    }
    return text;
  }

  if (event.mode === "on_chat_model_stream" || event.mode === "chat_model_stream") {
    return extractText(event.payload);
  }

  return "";
}

interface StreamRunArgs {
  threadId: string;
  assistantId: string;
  message: string;
  contextItems: ContextItem[];
  model?: string;
  userId?: string | null;
  streamToolArgs?: boolean;
  signal?: AbortSignal;
}

export async function* streamRun(args: StreamRunArgs): AsyncGenerator<NormalizedStreamEvent> {
  const textContextItems = args.contextItems.filter((item) => !item.attachment);
  const multimodalItems = args.contextItems.filter((item) => Boolean(item.attachment));

  const contextBlob = textContextItems
    .map((item) => item.content.trim())
    .filter(Boolean)
    .join("\n\n");

  const effectiveMessage = contextBlob
    ? `Additional context for this request:\n${contextBlob}\n\nUser message:\n${args.message}`
    : args.message;

  const multimodalContent: Array<Record<string, unknown>> = [{ type: "text", text: effectiveMessage }];
  for (const item of multimodalItems) {
    const attachment = item.attachment;
    if (!attachment) {
      continue;
    }
    if (attachment.kind === "image") {
      multimodalContent.push({
        type: "image",
        source_type: "base64",
        data: attachment.base64,
        mime_type: attachment.mimeType
      });
      continue;
    }
    multimodalContent.push({
      type: "file",
      source_type: "base64",
      data: attachment.base64,
      filename: attachment.filename,
      mime_type: attachment.mimeType
    });
  }

  const messageContent: string | Array<Record<string, unknown>> =
    multimodalContent.length > 1 ? multimodalContent : effectiveMessage;

  const payload = {
    assistant_id: args.assistantId,
    input: {
      messages: [{ type: "human", content: messageContent }],
      context_items: args.contextItems.map((item) => ({
        id: item.id,
        type: item.type,
        source: item.source,
        content: item.content,
        token_count: item.tokenCount,
        ...(item.lineRange ? { line_range: item.lineRange } : {})
      }))
    },
    config: {
      configurable: {
        thread_id: args.threadId,
        conversation_uuid: args.threadId,
        stream_tool_args: args.streamToolArgs ?? true,
        ...(args.model ? { model_name: args.model } : {}),
        ...(args.userId ? { user_id: args.userId } : {})
      }
    },
    stream_mode: ["messages", "messages-tuple", "updates", "custom"]
  };

  const response = await fetch(`${PROXY_BASE}/threads/${encodeURIComponent(args.threadId)}/runs/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload),
    cache: "no-store",
    signal: args.signal
  });

  if (!response.ok) {
    await parseApiError(response);
  }

  if (!response.body) {
    throw new Error("Streaming response body is missing.");
  }

  for await (const event of parseEventStream(response.body)) {
    if (event.data === "[DONE]") {
      break;
    }

    let parsedPayload: unknown = event.data;
    try {
      parsedPayload = JSON.parse(event.data) as unknown;
    } catch {
      parsedPayload = event.data;
    }

    yield normalizeStreamEvent(event.event, parsedPayload);
  }
}

export async function getThreadState(threadId: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/threads/${encodeURIComponent(threadId)}/state`, {
    method: "GET"
  });
}
