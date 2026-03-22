import raw from "../../src/bioagent/config/chat_models.json";

export interface CatalogModel {
  id: string;
  label: string;
  context_window_tokens?: number;
}

interface RawCatalog {
  default_context_window_tokens: number;
  default_model_id: string;
  models: CatalogModel[];
}

const spec = raw as RawCatalog;

function assertCatalog(): void {
  if (!spec.default_model_id?.trim()) {
    throw new Error("chat_models.json: default_model_id required");
  }
  if (!Number.isFinite(spec.default_context_window_tokens) || spec.default_context_window_tokens <= 0) {
    throw new Error("chat_models.json: default_context_window_tokens must be a positive number");
  }
  if (!Array.isArray(spec.models) || spec.models.length === 0) {
    throw new Error("chat_models.json: models must be a non-empty array");
  }
  const ids = new Set<string>();
  for (const m of spec.models) {
    if (!m.id?.trim() || !m.label?.trim()) {
      throw new Error("chat_models.json: each model needs id and label");
    }
    if (ids.has(m.id)) {
      throw new Error(`chat_models.json: duplicate id ${m.id}`);
    }
    ids.add(m.id);
    if (
      m.context_window_tokens !== undefined &&
      (!Number.isFinite(m.context_window_tokens) || m.context_window_tokens <= 0)
    ) {
      throw new Error(`chat_models.json: invalid context_window_tokens for ${m.id}`);
    }
  }
  if (!ids.has(spec.default_model_id)) {
    throw new Error("chat_models.json: default_model_id must be listed in models");
  }
}

assertCatalog();

export const MODEL_CATALOG: ReadonlyArray<CatalogModel> = spec.models;

export const DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS = spec.default_context_window_tokens;

export const MODEL_CONTEXT_WINDOWS: Record<string, number> = Object.fromEntries(
  spec.models.map((m) => [m.id, m.context_window_tokens ?? spec.default_context_window_tokens])
);

/** OpenRouter-style model id; values come from chat_models.json at build time. */
export type CatalogModelId = string;

export const DEFAULT_MODEL_ID: CatalogModelId = spec.default_model_id;

export function isAllowedCatalogModel(value: string): value is CatalogModelId {
  return MODEL_CATALOG.some((item) => item.id === value);
}

export function modelDisplayName(value: string): string {
  const match = MODEL_CATALOG.find((item) => item.id === value);
  return match?.label ?? value;
}

export function modelContextWindowTokens(value: string): number {
  return isAllowedCatalogModel(value) ? MODEL_CONTEXT_WINDOWS[value]! : DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS;
}

export function modelContextUsagePercent(value: string, promptTokens: number): number {
  const maxTokens = modelContextWindowTokens(value);
  if (!Number.isFinite(promptTokens) || promptTokens <= 0 || maxTokens <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round((promptTokens / maxTokens) * 100)));
}
