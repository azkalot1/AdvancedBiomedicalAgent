import { readFile } from "node:fs/promises";
import path from "node:path";

import { load as parseYaml } from "js-yaml";

import type { StarterPromptCategory } from "@/lib/types";

interface RawPromptCategory {
  id?: unknown;
  title?: unknown;
  prompts?: unknown;
}

interface RawPromptCatalog {
  categories?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

function normalizeCategory(raw: RawPromptCategory, index: number): StarterPromptCategory | null {
  const title = typeof raw.title === "string" ? raw.title.trim() : "";
  const promptsRaw = Array.isArray(raw.prompts) ? raw.prompts : [];
  const prompts = promptsRaw
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .slice(0, 10);

  if (!title || prompts.length === 0) {
    return null;
  }

  const fromId = typeof raw.id === "string" ? raw.id.trim() : "";
  const id = fromId || slugify(title) || `category-${index + 1}`;

  return { id, title, prompts };
}

function parseCatalog(content: string): StarterPromptCategory[] {
  const parsed = parseYaml(content) as unknown;
  if (!isRecord(parsed)) {
    return [];
  }
  const root = parsed as RawPromptCatalog;
  const categoriesRaw = Array.isArray(root.categories) ? root.categories : [];

  const categories: StarterPromptCategory[] = [];
  for (let index = 0; index < categoriesRaw.length; index += 1) {
    const item = categoriesRaw[index];
    if (!isRecord(item)) {
      continue;
    }
    const normalized = normalizeCategory(item as RawPromptCategory, index);
    if (normalized) {
      categories.push(normalized);
    }
  }

  return categories.slice(0, 30);
}

export async function loadStarterPromptCategories(): Promise<StarterPromptCategory[]> {
  const candidates = [
    path.join(process.cwd(), "starter-prompts.yaml"),
    path.join(process.cwd(), "web", "starter-prompts.yaml")
  ];

  for (const filePath of candidates) {
    try {
      const content = await readFile(filePath, "utf8");
      const parsed = parseCatalog(content);
      if (parsed.length > 0) {
        return parsed;
      }
    } catch {
      // Continue to next candidate.
    }
  }

  return [];
}
