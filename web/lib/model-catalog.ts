export const MODEL_CATALOG = [
  { id: "anthropic/claude-opus-4.6", label: "Opus 4.6" },
  { id: "anthropic/claude-sonnet-4.6", label: "Sonnet 4.6" },
  { id: "anthropic/claude-haiku-4.5", label: "Haiku 4.5" },
  { id: "google/gemini-3-flash-preview", label: "Gemini 3 Flash" },
  { id: "google/gemini-3.1-flash-lite-preview", label: "Gemini 3.1 Flash Lite" },
  { id: "google/gemini-3.1-pro-preview", label: "Gemini 3.1 Pro" },
  { id: "z-ai/glm-5", label: "GLM-5" },
  { id: "moonshotai/kimi-k2.5", label: "Kimi K2.5" },
  { id: "minimax/minimax-m2.5", label: "Minimax M2.5" },
  { id: "qwen/qwen3.5-35b-a3b", label: "Qwen 3.5 35B A3B" },
] as const;

export const DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS = 200_000;

export const MODEL_CONTEXT_WINDOWS: Record<CatalogModelId, number> = {
  "anthropic/claude-opus-4.6": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "anthropic/claude-sonnet-4.6": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "anthropic/claude-haiku-4.5": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "google/gemini-3-flash-preview": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "google/gemini-3.1-flash-lite-preview": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "google/gemini-3.1-pro-preview": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "z-ai/glm-5": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "moonshotai/kimi-k2.5": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "minimax/minimax-m2.5": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
  "qwen/qwen3.5-35b-a3b": DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS
};

export type CatalogModelId = (typeof MODEL_CATALOG)[number]["id"];

export const DEFAULT_MODEL_ID: CatalogModelId = "anthropic/claude-sonnet-4.6";

export function isAllowedCatalogModel(value: string): value is CatalogModelId {
  return MODEL_CATALOG.some((item) => item.id === value);
}

export function modelDisplayName(value: string): string {
  const match = MODEL_CATALOG.find((item) => item.id === value);
  return match?.label ?? value;
}

export function modelContextWindowTokens(value: string): number {
  return isAllowedCatalogModel(value) ? MODEL_CONTEXT_WINDOWS[value] : DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS;
}

export function modelContextUsagePercent(value: string, promptTokens: number): number {
  const maxTokens = modelContextWindowTokens(value);
  if (!Number.isFinite(promptTokens) || promptTokens <= 0 || maxTokens <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round((promptTokens / maxTokens) * 100)));
}
