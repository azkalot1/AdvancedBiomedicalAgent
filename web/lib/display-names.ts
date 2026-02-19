import type { ReportFile, ThreadSummary } from "@/lib/types";
import { shortId } from "@/lib/utils";

function metadataDisplayName(metadata?: Record<string, unknown>): string | undefined {
  if (!metadata) {
    return undefined;
  }
  const value = metadata.display_name ?? metadata.thread_display_name ?? metadata.title;
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

export function reportDisplayName(report: Pick<ReportFile, "displayName" | "filename">): string {
  const label = report.displayName?.trim();
  return label || report.filename;
}

export function threadDisplayName(thread: Pick<ThreadSummary, "id" | "displayName" | "metadata">): string {
  const direct = thread.displayName?.trim();
  if (direct) {
    return direct;
  }
  const meta = metadataDisplayName(thread.metadata);
  if (meta) {
    return meta;
  }
  return `Thread ${shortId(thread.id)}`;
}
