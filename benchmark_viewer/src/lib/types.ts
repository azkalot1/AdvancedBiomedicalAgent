export interface RunManifest {
  run_id: string;
  run_started_at: string;
  run_label?: string | null;
  suite_name: string;
  dataset_paths: string[];
  profile: {
    name: string;
    model: {
      provider: string;
      model_name: string;
      temperature?: number;
    };
    server?: {
      base_url?: string;
    };
  };
  output_dir: string;
  files: {
    manifest: string;
    raw_runs: string;
    summary_json: string;
    summary_markdown: string;
    cases_dir: string;
  };
  counts: {
    cases: number;
    correct: number;
    incorrect: number;
    invalid_format: number;
    error: number;
  };
}

export interface RunSummaryPayload {
  summary: {
    total: number;
    correct: number;
    incorrect: number;
    invalid_format: number;
    error: number;
    accuracy: number;
    invalid_format_rate: number;
    error_rate: number;
    average_latency_seconds: number;
    average_tool_calls: number;
    average_successful_tool_calls: number;
    token_usage_available_runs?: number;
    sum_input_tokens?: number;
    sum_output_tokens?: number;
    sum_total_tokens?: number;
    average_input_tokens?: number | null;
    average_output_tokens?: number | null;
    average_total_tokens?: number | null;
    sum_streamed_output_chars?: number;
    average_streamed_output_chars?: number;
  };
  by_category: Record<
    string,
    {
      total: number;
      correct: number;
      incorrect: number;
      invalid_format: number;
      error: number;
      accuracy: number;
    }
  >;
  tool_usage: Record<string, number>;
}

export interface RawRunRecord {
  run_id: string;
  run_started_at: string;
  run_label?: string | null;
  suite_name: string;
  dataset_path: string;
  case_id: string;
  detail_path?: string;
  case: {
    id: string;
    question: string;
    options: Record<string, string>;
    correct_option: string;
    category?: string;
    expected_tools?: string[];
    allowed_tools?: string[];
    metadata?: Record<string, unknown>;
  };
  profile: {
    name: string;
    model: {
      provider: string;
      model_name: string;
    };
  };
  final_answer_text?: string;
  answer_status: string;
  selected_option?: string | null;
  is_correct?: boolean;
  expected_option?: string;
  latency_seconds?: number;
  token_chars?: number;
  token_usage?: {
    source: string;
    input_tokens?: number | null;
    output_tokens?: number | null;
    total_tokens?: number | null;
  };
  tool_summary?: {
    total_calls: number;
    successful_calls: number;
    failed_calls: number;
    by_tool: Record<string, number>;
  };
  tool_invocations?: ToolInvocation[];
}

export interface ToolInvocation {
  invocation_id: string;
  tool_name: string;
  started_at?: string | null;
  finished_at?: string | null;
  statuses: string[];
  args_preview?: unknown;
  duration_ms?: number | null;
  error?: string | null;
}

export interface NormalizedMessage {
  index: number;
  id?: string | null;
  type?: string | null;
  role?: string | null;
  name?: string | null;
  content_text?: string | null;
  content_raw?: unknown;
  tool_calls?: Array<Record<string, unknown>>;
  additional_tool_calls?: Array<Record<string, unknown>>;
  usage_metadata?: Record<string, unknown>;
  response_metadata?: Record<string, unknown> | null;
  additional_kwargs?: Record<string, unknown> | null;
}

export interface CaseDetailRecord extends RawRunRecord {
  thread_id?: string;
  user_id?: string;
  report_ids?: string[];
  report_snapshots?: Array<Record<string, unknown>>;
  tool_events?: Array<Record<string, unknown>>;
  tool_invocations?: ToolInvocation[];
  state_tool_calls?: string[];
  thread_state?: Record<string, unknown>;
  normalized_messages?: NormalizedMessage[];
}

export interface RunIndexRecord {
  manifest: RunManifest;
  summary: RunSummaryPayload;
  relativeRunPath: string;
  datasetKey: string;
  modelBucket: string;
}

export interface LoadedRunData {
  manifest: RunManifest;
  summary: RunSummaryPayload;
  rawRuns: RawRunRecord[];
}

export interface ComparisonRow {
  caseId: string;
  question: string;
  category?: string;
  runs: Record<string, RawRunRecord | null>;
}
