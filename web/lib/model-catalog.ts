export const MODEL_CATALOG = [
  { id: "anthropic/claude-opus-4.6", label: "Opus 4.6" },
  { id: "anthropic/claude-sonnet-4.6", label: "Sonnet 4.6" },
  { id: "google/gemini-3-flash-preview", label: "Gemini 3 Flash" },
  { id: "google/gemini-3.1-pro-preview", label: "Gemini 3.1 Pro" },
  { id: "z-ai/glm-5", label: "GLM-5" },
  { id: "moonshotai/kimi-k2.5", label: "Kimi K2.5" },
  { id: "minimax/minimax-m2.5", label: "Minimax M2.5" },
] as const;

export type CatalogModelId = (typeof MODEL_CATALOG)[number]["id"];

export const DEFAULT_MODEL_ID: CatalogModelId = "anthropic/claude-sonnet-4.6";

export function isAllowedCatalogModel(value: string): value is CatalogModelId {
  return MODEL_CATALOG.some((item) => item.id === value);
}

export function modelDisplayName(value: string): string {
  const match = MODEL_CATALOG.find((item) => item.id === value);
  return match?.label ?? value;
}
