"use client";

import { autoUpdate, flip, offset, shift, useFloating } from "@floating-ui/react";
import * as Tabs from "@radix-ui/react-tabs";
import { BarChart3, FileText, MessageCircle, Paperclip, Search } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";

import { getReportContent } from "@/lib/backend-client";
import { reportDisplayName } from "@/lib/display-names";
import { estimateTokens } from "@/lib/token-estimate";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { safeDate } from "@/lib/utils";

interface FloatingSelection {
  text: string;
  lineRange: [number, number];
  x: number;
  y: number;
}

function computeLineRange(content: string, selectionText: string): [number, number] {
  const normalized = selectionText.trim();
  if (!normalized) {
    return [1, 1];
  }

  const startIndex = content.indexOf(normalized);
  if (startIndex < 0) {
    return [1, Math.max(1, normalized.split(/\r?\n/).length)];
  }

  const startLine = content.slice(0, startIndex).split(/\r?\n/).length;
  const lines = normalized.split(/\r?\n/).length;
  return [startLine, startLine + lines - 1];
}

export function DocumentViewer(): React.ReactElement {
  const reports = useBioAgentStore((state) => state.reports);
  const selectedReportId = useBioAgentStore((state) => state.selectedReportId);
  const loadingReportId = useBioAgentStore((state) => state.loadingReportId);
  const reportContentById = useBioAgentStore((state) => state.reportContentById);
  const setLoadingReportId = useBioAgentStore((state) => state.setLoadingReportId);
  const cacheReportContent = useBioAgentStore((state) => state.cacheReportContent);
  const addContextItem = useBioAgentStore((state) => state.addContextItem);
  const setDraft = useBioAgentStore((state) => state.setDraft);

  const containerRef = useRef<HTMLDivElement | null>(null);

  const selectedReport = useMemo(
    () => reports.find((report) => report.id === selectedReportId) ?? null,
    [reports, selectedReportId]
  );
  const selectedContent = selectedReportId ? reportContentById[selectedReportId] : undefined;

  const [selection, setSelection] = useState<FloatingSelection | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const floating = useFloating({
    open: Boolean(selection),
    whileElementsMounted: autoUpdate,
    middleware: [offset(8), flip(), shift({ padding: 10 })]
  });

  useEffect(() => {
    if (!selectedReportId) {
      setLoadError(null);
      setLoadingReportId(null);
      return;
    }
    if (selectedContent !== undefined) {
      setLoadError(null);
      setLoadingReportId(null);
      return;
    }

    let cancelled = false;
    setLoadError(null);
    setLoadingReportId(selectedReportId);

    void getReportContent(selectedReportId)
      .then((contentPayload) => {
        if (cancelled) {
          return;
        }
        cacheReportContent(selectedReportId, contentPayload.content);
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError("Failed to load report content.");
          cacheReportContent(selectedReportId, "# Report load failed\n\nThe backend returned an error while loading this file.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingReportId(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [cacheReportContent, selectedContent, selectedReportId, setLoadingReportId]);

  useEffect(() => {
    if (!selection) {
      return;
    }

    const refs = floating.refs as unknown as {
      setPositionReference?: (element: { getBoundingClientRect: () => DOMRect }) => void;
    };

    refs.setPositionReference?.({
      getBoundingClientRect: () =>
        new DOMRect(
          selection.x,
          selection.y,
          1,
          1
        )
    });
  }, [floating.refs, selection]);

  const onMouseUp = useCallback(() => {
    const root = containerRef.current;
    const browserSelection = window.getSelection();
    if (!root || !browserSelection || browserSelection.isCollapsed) {
      setSelection(null);
      return;
    }

    if (!browserSelection.anchorNode || !root.contains(browserSelection.anchorNode)) {
      return;
    }

    const selectedText = browserSelection.toString().trim();
    if (!selectedText) {
      setSelection(null);
      return;
    }

    const range = browserSelection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const content = selectedContent ?? "";

    setSelection({
      text: selectedText,
      lineRange: computeLineRange(content, selectedText),
      x: rect.left + rect.width / 2,
      y: rect.top
    });
  }, [selectedContent]);

  const addSelectionToContext = (): void => {
    if (!selection || !selectedReport) {
      return;
    }
    const reportLabel = reportDisplayName(selectedReport);

    const result = addContextItem({
      id: `ctx-selection-${Date.now()}`,
      type: "file_selection",
      source: reportLabel,
      content: selection.text,
      lineRange: selection.lineRange,
      tokenCount: estimateTokens(selection.text),
      addedAt: new Date().toISOString()
    });

    setMessage(result.ok ? "Added to context." : result.reason);
    setSelection(null);
  };

  const askAboutSelection = (): void => {
    if (!selection) {
      return;
    }
    setDraft(`Please analyze this selected excerpt:\n\n${selection.text}`);
    setSelection(null);
    setMessage("Selection queued into chat input.");
  };

  return (
    <section className="relative flex h-full min-h-0 flex-col bg-surface/45">
      <div className="border-b border-surface-edge/60 px-3 py-3">
        <h2 className="text-lg font-semibold text-zinc-100">Document Viewer</h2>
        <p className="mt-1 text-xs text-zinc-500">
          {selectedReport ? `${reportDisplayName(selectedReport)} • ${safeDate(selectedReport.createdAt) || "n/a"}` : "No report selected"}
        </p>
      </div>

      <Tabs.Root defaultValue="preview" className="flex min-h-0 flex-1 flex-col">
        <Tabs.List className="mx-3 mt-2 inline-flex rounded-lg border border-surface-edge bg-surface-raised p-1 text-xs">
          <Tabs.Trigger
            value="preview"
            className="rounded-md px-3 py-1 text-zinc-300 data-[state=active]:bg-accent-blue/25 data-[state=active]:text-accent-blue"
          >
            Preview
          </Tabs.Trigger>
          <Tabs.Trigger
            value="raw"
            className="rounded-md px-3 py-1 text-zinc-300 data-[state=active]:bg-accent-cyan/25 data-[state=active]:text-accent-cyan"
          >
            Raw
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="preview" className="min-h-0 flex-1 outline-none">
          <div ref={containerRef} onMouseUp={onMouseUp} className="markdown-body h-full overflow-y-auto px-4 py-3 text-sm text-zinc-100">
            {selectedContent ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                {selectedContent}
              </ReactMarkdown>
            ) : loadingReportId === selectedReportId ? (
              <p className="text-zinc-400">Loading report...</p>
            ) : (
              <p className="text-zinc-500">Select a report from the left panel.</p>
            )}
          </div>
        </Tabs.Content>

        <Tabs.Content value="raw" className="min-h-0 flex-1 outline-none">
          <pre className="h-full overflow-y-auto whitespace-pre-wrap px-4 py-3 text-xs text-zinc-200">
            {selectedContent ?? "No report loaded."}
          </pre>
        </Tabs.Content>
      </Tabs.Root>

      {selection ? (
        <div
          ref={floating.refs.setFloating}
          style={floating.floatingStyles}
          className="z-40 min-w-[190px] rounded-lg border border-accent-blue/45 bg-surface-raised/95 p-1 text-sm shadow-xl backdrop-blur"
        >
          <button
            type="button"
            onClick={addSelectionToContext}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-zinc-100 hover:bg-surface-overlay"
          >
            <Paperclip className="h-4 w-4 text-accent-cyan" /> Add to Context
          </button>
          <button
            type="button"
            onClick={askAboutSelection}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-zinc-100 hover:bg-surface-overlay"
          >
            <MessageCircle className="h-4 w-4 text-accent-blue" /> Ask about this
          </button>
          <button type="button" className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-zinc-300 hover:bg-surface-overlay">
            <FileText className="h-4 w-4" /> Summarize
          </button>
          <button type="button" className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-zinc-300 hover:bg-surface-overlay">
            <Search className="h-4 w-4" /> Expand
          </button>
          <button type="button" className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-zinc-300 hover:bg-surface-overlay">
            <BarChart3 className="h-4 w-4" /> Analyze
          </button>
        </div>
      ) : null}

      {message ? <div className="border-t border-surface-edge/50 px-3 py-1.5 text-xs text-accent-cyan">{message}</div> : null}
      {loadError ? <div className="border-t border-surface-edge/50 px-3 py-1.5 text-xs text-status-error">{loadError}</div> : null}
    </section>
  );
}
