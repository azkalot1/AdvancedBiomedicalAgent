import { getServerSession } from "next-auth";

import { authOptions } from "@/lib/auth";
import { loadStarterPromptCategories } from "@/lib/server/starter-prompts";
import type { InitialWorkbenchData } from "@/lib/types";

const DEFAULT_BACKEND_URL = "http://localhost:2024";

function backendBaseUrl(): string {
  return (process.env.BIOAGENT_BACKEND_URL || process.env.NEXT_PUBLIC_BIOAGENT_BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/$/, "");
}

async function backendHealthy(): Promise<boolean> {
  const headers = new Headers();
  const token = process.env.BIOAGENT_API_TOKEN?.trim();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${backendBaseUrl()}/v1/ok`, {
    method: "GET",
    headers,
    cache: "no-store"
  });
  return response.ok;
}

async function resolveUserId(): Promise<string | null> {
  const session = await getServerSession(authOptions);
  return session?.user?.id ?? null;
}

export async function loadInitialWorkbenchData(): Promise<InitialWorkbenchData> {
  const userId = await resolveUserId();
  const starterPromptCategories = await loadStarterPromptCategories();

  try {
    const backendOk = await backendHealthy();

    return {
      userId,
      authRequired: true,
      reports: [],
      backendOk,
      starterPromptCategories
    };
  } catch {
    return {
      userId,
      authRequired: true,
      reports: [],
      backendOk: false,
      starterPromptCategories
    };
  }
}
