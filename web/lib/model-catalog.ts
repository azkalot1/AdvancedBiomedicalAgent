export const MODEL_CATALOG = [
  { id: "anthropic/claude-opus-4.6", label: "Opus 4.6" },
  { id: "anthropic/claude-sonnet-4.6", label: "Sonnet 4.6" },
  { id: "moonshotai/kimi-k2.5", label: "Kimi K2.5" },
  { id: "google/gemini-3-flash-preview", label: "Gemini 3 Flash" },
  { id: "google/gemini-3-pro-preview", label: "Gemini 3 Pro" },
  { id: "openai/gpt-5.2", label: "GPT-5.2" }
] as const;

export type CatalogModelId = (typeof MODEL_CATALOG)[number]["id"];

export const DEFAULT_MODEL_ID: CatalogModelId = "google/gemini-3-flash-preview";

export function isAllowedCatalogModel(value: string): value is CatalogModelId {
  return MODEL_CATALOG.some((item) => item.id === value);
}

export function modelDisplayName(value: string): string {
  const match = MODEL_CATALOG.find((item) => item.id === value);
  return match?.label ?? value;
}
