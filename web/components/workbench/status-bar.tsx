"use client";

import { Wifi, WifiOff } from "lucide-react";

import { modelContextUsagePercent, modelContextWindowTokens, modelDisplayName } from "@/lib/model-catalog";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";

function connectionLabel(state: "connected" | "degraded" | "offline"): string {
  if (state === "connected") {
    return "Connected";
  }
  if (state === "degraded") {
    return "Degraded";
  }
  return "Offline";
}

export function StatusBar(): React.ReactElement {
  const model = useBioAgentStore((state) => state.model);
  const reports = useBioAgentStore((state) => state.reports);
  const connection = useBioAgentStore((state) => state.connection);
  const currentPromptTokens = useBioAgentStore((state) => state.currentPromptTokens);
  const displayPromptTokens = currentPromptTokens ?? 0;
  const contextWindowTokens = modelContextWindowTokens(model);
  const contextUsedPercent = modelContextUsagePercent(model, displayPromptTokens);

  return (
    <footer className="flex h-9 items-center justify-between border-t border-surface-edge/70 bg-surface/90 px-4 text-xs text-zinc-400">
      <div className="flex items-center gap-4">
        <span>Model: {modelDisplayName(model)}</span>
        <span title={`${displayPromptTokens.toLocaleString()} / ${contextWindowTokens.toLocaleString()} prompt tokens`}>
          Context: {contextUsedPercent}% used
        </span>
        <span>Reports: {reports.length}</span>
      </div>

      <div className="inline-flex items-center gap-1">
        {connection === "offline" ? <WifiOff className="h-3.5 w-3.5" /> : <Wifi className="h-3.5 w-3.5" />}
        {connectionLabel(connection)}
      </div>
    </footer>
  );
}
