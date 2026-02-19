"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { useEffect, useRef } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

import { ChatPanel } from "@/components/workbench/chat-panel";
import { ContextManager } from "@/components/workbench/context-manager";
import { DocumentViewer } from "@/components/workbench/document-viewer";
import { FileExplorerPanel } from "@/components/workbench/file-explorer-panel";
import { StatusBar } from "@/components/workbench/status-bar";
import { ToolStatePanel } from "@/components/workbench/tool-state-panel";
import { TopBar } from "@/components/workbench/top-bar";
import { listThreads } from "@/lib/backend-client";
import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { sortThreadsByCreatedAt } from "@/lib/thread-session";
import type { InitialWorkbenchData } from "@/lib/types";
import { cn } from "@/lib/utils";

export function WorkbenchShell({ initialData }: { initialData: InitialWorkbenchData }): React.ReactElement {
  const hydrate = useBioAgentStore((state) => state.hydrate);
  const clearThreadWorkspace = useBioAgentStore((state) => state.clearThreadWorkspace);
  const setThreadId = useBioAgentStore((state) => state.setThreadId);
  const setThreads = useBioAgentStore((state) => state.setThreads);
  const setConnection = useBioAgentStore((state) => state.setConnection);
  const initializedRef = useRef(false);

  useEffect(() => {
    hydrate(initialData);
  }, [hydrate, initialData]);

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    let cancelled = false;
    clearThreadWorkspace();
    setThreadId(null);
    void (async () => {
      try {
        const knownThreads = sortThreadsByCreatedAt(await listThreads(100, initialData.userId));
        if (cancelled) {
          return;
        }
        setThreads(knownThreads);
        setConnection("connected");
      } catch {
        if (!cancelled) {
          setThreads([]);
          setConnection("degraded");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [clearThreadWorkspace, initialData.userId, setConnection, setThreadId, setThreads]);

  return (
    <main className="h-screen min-h-0 w-full bg-transparent p-3 text-zinc-100">
      <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-surface-edge/80 bg-surface/70 shadow-glow backdrop-blur-sm">
        <TopBar />

        <PanelGroup direction="horizontal" className="min-h-0 flex-1">
          <Panel defaultSize={21} minSize={14} maxSize={32}>
            <FileExplorerPanel />
          </Panel>
          <PanelResizeHandle className="w-1 bg-surface-edge/40 transition hover:bg-accent-blue/75" />

          <Panel defaultSize={48} minSize={28}>
            <ChatPanel />
          </Panel>
          <PanelResizeHandle className="w-1 bg-surface-edge/40 transition hover:bg-accent-blue/75" />

          <Panel defaultSize={31} minSize={22} maxSize={45}>
            <Tabs.Root defaultValue="document" className="flex h-full min-h-0 flex-col">
              <Tabs.List className="border-b border-surface-edge/60 bg-surface/40 p-2 text-xs">
                <Tabs.Trigger
                  value="document"
                  className={cn(
                    "rounded-md px-2 py-1 text-zinc-400",
                    "data-[state=active]:bg-accent-blue/20 data-[state=active]:text-accent-blue"
                  )}
                >
                  Document Viewer
                </Tabs.Trigger>
                <Tabs.Trigger
                  value="context"
                  className={cn(
                    "rounded-md px-2 py-1 text-zinc-400",
                    "data-[state=active]:bg-accent-cyan/20 data-[state=active]:text-accent-cyan"
                  )}
                >
                  Context Manager
                </Tabs.Trigger>
                <Tabs.Trigger
                  value="tool-state"
                  className={cn(
                    "rounded-md px-2 py-1 text-zinc-400",
                    "data-[state=active]:bg-emerald-400/20 data-[state=active]:text-emerald-300"
                  )}
                >
                  Tool State
                </Tabs.Trigger>
              </Tabs.List>

              <Tabs.Content value="document" className="min-h-0 flex-1 outline-none">
                <DocumentViewer />
              </Tabs.Content>
              <Tabs.Content value="context" className="min-h-0 flex-1 outline-none">
                <ContextManager />
              </Tabs.Content>
              <Tabs.Content value="tool-state" className="min-h-0 flex-1 outline-none">
                <ToolStatePanel />
              </Tabs.Content>
            </Tabs.Root>
          </Panel>
        </PanelGroup>

        <StatusBar />
      </div>
    </main>
  );
}
