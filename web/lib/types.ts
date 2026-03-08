export type ReportStatus = "complete" | "generating" | "error";

export interface ReportFile {
  id: string;
  refId?: string;
  filename: string;
  displayName?: string;
  toolName?: string;
  status: ReportStatus;
  sizeChars?: number;
  oneLine?: string;
  threadId?: string;
  userId?: string;
  createdAt?: string;
  path?: string;
}

export interface ReportContent {
  id: string;
  filename: string;
  content: string;
  totalChars: number;
  returnedChars: number;
  offset: number;
  maxChars: number;
  truncated: boolean;
}

export interface ContextItem {
  id: string;
  type: "file_selection" | "full_file" | "user_query" | "tool_result" | "manual_paste" | "uploaded_file";
  source: string;
  content: string;
  tokenCount: number;
  lineRange?: [number, number];
  addedAt: string;
  attachment?: {
    kind: "image" | "pdf";
    filename: string;
    mimeType: "image/png" | "image/jpeg" | "application/pdf";
    base64: string;
    sizeBytes: number;
  };
}

export interface ToolEvent {
  id: string;
  type: "tool_status" | "report_generated" | "context_updated" | "plot_data";
  toolName?: string;
  status?: "queued" | "running" | "success" | "error";
  progress?: number;
  durationMs?: number;
  timestamp: string;
  invocationId?: string;
  error?: string;
  message?: string;
  reason?: string;
  argsPreview?: Record<string, unknown>;
  report?: ReportFile;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  streaming?: boolean;
  reportRefIds?: string[];
}

export type ResponseFeedbackReason =
  | "wrong_or_outdated"
  | "did_not_address_question"
  | "missing_information"
  | "too_much_irrelevant_detail"
  | "unclear_reliability";

export interface ResponseFeedbackRecord {
  helpful: boolean;
  reason?: ResponseFeedbackReason;
  submittedAt: string;
  messageId?: string;
}

export interface StarterPromptCategory {
  id: string;
  title: string;
  prompts: string[];
}

export interface MeResponse {
  user_id: string;
  auth_required: boolean;
}

export interface ThreadSummary {
  id: string;
  displayName?: string;
  createdAt?: string;
  metadata?: Record<string, unknown>;
}

export interface ReportsListResponse {
  items: Array<{
    id: string;
    ref_id?: string;
    filename?: string;
    display_name?: string;
    tool_name?: string;
    status?: string;
    size_chars?: number;
    one_line?: string;
    thread_id?: string;
    user_id?: string;
    created_at?: string;
    path?: string;
  }>;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export type StreamMode = "messages" | "messages-tuple" | "messages_tuple" | "on_chat_model_stream" | "chat_model_stream" | "custom";

export interface NormalizedStreamEvent {
  mode: StreamMode | string;
  payload: unknown;
}

export interface InitialWorkbenchData {
  userId: string | null;
  authRequired: boolean;
  reports: ReportFile[];
  backendOk: boolean;
  starterPromptCategories: StarterPromptCategory[];
}
