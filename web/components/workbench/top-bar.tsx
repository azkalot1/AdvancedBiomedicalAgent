"use client";

import { FlaskConical, Settings2 } from "lucide-react";

import { UserMenu } from "@/components/user-menu";
import { MODEL_CATALOG, modelContextUsagePercent, modelContextWindowTokens } from "@/lib/model-catalog";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";

export function TopBar(): React.ReactElement {
  const model = useBioAgentStore((state) => state.model);
  const setModel = useBioAgentStore((state) => state.setModel);
  const currentPromptTokens = useBioAgentStore((state) => state.currentPromptTokens);
  const displayPromptTokens = currentPromptTokens ?? 0;
  const contextWindowTokens = modelContextWindowTokens(model);
  const contextUsedPercent = modelContextUsagePercent(model, displayPromptTokens);

  return (
    <header className="flex h-14 items-center justify-between border-b border-surface-edge/70 bg-surface/85 px-4 backdrop-blur">
      <div className="flex items-center gap-2 text-zinc-100">
        <FlaskConical className="h-4 w-4 text-accent-cyan" />
        <span className="text-lg font-semibold tracking-tight">AI Co-Scientist</span>
      </div>

      <div className="flex items-center gap-3 text-sm">
        <label className="flex items-center gap-2 rounded-lg border border-surface-edge bg-surface-raised px-3 py-1.5">
          <span className="text-zinc-400">Model</span>
          <select
            className="bg-transparent text-zinc-100 outline-none"
            value={model}
            onChange={(event) => setModel(event.target.value)}
          >
            {MODEL_CATALOG.map((modelOption) => (
              <option key={modelOption.id} value={modelOption.id} className="bg-slate-900">
                {modelOption.label}
              </option>
            ))}
          </select>
        </label>

        <div
          className="rounded-lg border border-surface-edge bg-surface-raised px-3 py-1.5 text-zinc-300"
          title={`${displayPromptTokens.toLocaleString()} / ${contextWindowTokens.toLocaleString()} prompt tokens`}
        >
          Context {contextUsedPercent}% used
        </div>

        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-lg border border-surface-edge bg-surface-raised px-3 py-1.5 text-zinc-300 hover:border-accent-blue/60 hover:text-zinc-50"
        >
          <Settings2 className="h-4 w-4" />
          Settings
        </button>

        <UserMenu />
      </div>
    </header>
  );
}
