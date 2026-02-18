"use client";

import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent
} from "@dnd-kit/core";
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { FilePlus2, GripVertical, Upload, X } from "lucide-react";
import { useRef, useState } from "react";

import { useBioAgentStore } from "@/lib/stores/use-bioagent-store";
import { estimateTokens } from "@/lib/token-estimate";
import type { ContextItem } from "@/lib/types";

function TokenBudgetRing({ used }: { used: number }): React.ReactElement {
  const ratio = 1;
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - ratio);

  return (
    <div className="relative h-28 w-28">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={radius} stroke="rgba(148,163,184,0.25)" strokeWidth="7" fill="transparent" />
        <circle
          cx="50"
          cy="50"
          r={radius}
          stroke="url(#tokenRingGradient)"
          strokeWidth="7"
          strokeLinecap="round"
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
        />
        <defs>
          <linearGradient id="tokenRingGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#2dd4bf" />
            <stop offset="100%" stopColor="#4b92ff" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-2xl font-semibold text-zinc-100">{used.toLocaleString()}</span>
        <span className="text-xs text-zinc-400">total tokens</span>
      </div>
    </div>
  );
}

function ContextCard({ item, index }: { item: ContextItem; index: number }): React.ReactElement {
  const removeContextItem = useBioAgentStore((state) => state.removeContextItem);

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: item.id });

  return (
    <article
      ref={setNodeRef}
      style={{
        transform: CSS.Transform.toString(transform),
        transition
      }}
      className={
        "group rounded-lg border border-surface-edge bg-surface-raised/65 px-2 py-2 text-xs " +
        (isDragging ? "opacity-65" : "opacity-100")
      }
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          {...attributes}
          {...listeners}
          className="mt-0.5 cursor-grab text-zinc-500 hover:text-zinc-300 active:cursor-grabbing"
        >
          <GripVertical className="h-4 w-4" />
        </button>

        <div className="min-w-0 flex-1">
          <div className="truncate text-sm text-zinc-100">{item.source}</div>
          <div className="text-zinc-500">{item.tokenCount.toLocaleString()} tokens</div>
          {item.lineRange ? (
            <div className="text-zinc-500">
              lines {item.lineRange[0]}-{item.lineRange[1]}
            </div>
          ) : null}
          <div className="mt-1 line-clamp-2 text-zinc-400">{item.content}</div>
        </div>

        <button
          type="button"
          onClick={() => removeContextItem(item.id)}
          className="rounded p-1 text-zinc-500 hover:bg-surface-overlay hover:text-zinc-200"
          title="Remove from context"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-1 text-[10px] text-zinc-500">#{index + 1}</div>
    </article>
  );
}

export function ContextManager(): React.ReactElement {
  const contextItems = useBioAgentStore((state) => state.contextItems);
  const currentPromptTokens = useBioAgentStore((state) => state.currentPromptTokens);
  const reorderContextItems = useBioAgentStore((state) => state.reorderContextItems);
  const addContextItem = useBioAgentStore((state) => state.addContextItem);
  const promptTokens = currentPromptTokens ?? 0;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [quickAddMessage, setQuickAddMessage] = useState<string | null>(null);

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  const onUploadClick = (): void => {
    fileInputRef.current?.click();
  };

  const onPasteText = (): void => {
    const trimmed = pasteText.trim();
    if (!trimmed) {
      setQuickAddMessage("Paste text cannot be empty.");
      return;
    }

    const result = addContextItem({
      id: `ctx-paste-${Date.now()}`,
      type: "manual_paste",
      source: "Pasted Text",
      content: trimmed,
      tokenCount: estimateTokens(trimmed),
      addedAt: new Date().toISOString()
    });
    setQuickAddMessage(result.ok ? "Pasted text added to context." : result.reason);
    if (result.ok) {
      setPasteText("");
      setPasteOpen(false);
    }
  };

  const onFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const input = event.currentTarget;
    const files = input.files;
    if (!files || files.length === 0) {
      return;
    }

    const allowedMimeTypes = new Set(["application/pdf", "image/png", "image/jpeg"]);
    const maxSizeBytes = 10 * 1024 * 1024;
    const selected = files[0];

    const lowerName = selected.name.toLowerCase();
    const inferredMimeType =
      selected.type ||
      (lowerName.endsWith(".pdf")
        ? "application/pdf"
        : lowerName.endsWith(".png")
          ? "image/png"
          : lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg")
            ? "image/jpeg"
            : "");

    if (!allowedMimeTypes.has(inferredMimeType)) {
      setQuickAddMessage("Unsupported file type. Use PDF, PNG, or JPG.");
      input.value = "";
      return;
    }
    if (selected.size > maxSizeBytes) {
      setQuickAddMessage("File too large. Maximum supported size is 10 MB.");
      input.value = "";
      return;
    }

    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => reject(new Error("Failed to read file."));
        reader.readAsDataURL(selected);
      });

      const base64 = dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : "";
      if (!base64) {
        throw new Error("Failed to encode file.");
      }

      const result = addContextItem({
        id: `ctx-file-${Date.now()}`,
        type: "uploaded_file",
        source: selected.name,
        content: `Uploaded file: ${selected.name} (${Math.ceil(selected.size / 1024)} KB)`,
        tokenCount: 0,
        addedAt: new Date().toISOString(),
        attachment: {
          kind: inferredMimeType === "application/pdf" ? "pdf" : "image",
          filename: selected.name,
          mimeType: inferredMimeType as "application/pdf" | "image/png" | "image/jpeg",
          base64,
          sizeBytes: selected.size
        }
      });
      setQuickAddMessage(result.ok ? `File added: ${selected.name}` : result.reason);
    } catch {
      setQuickAddMessage("Failed to process selected file.");
    } finally {
      input.value = "";
    }
  };

  const onDragEnd = (event: DragEndEvent): void => {
    const { active, over } = event;
    if (!over || active.id === over.id) {
      return;
    }

    const fromIndex = contextItems.findIndex((item) => item.id === active.id);
    const toIndex = contextItems.findIndex((item) => item.id === over.id);
    if (fromIndex < 0 || toIndex < 0) {
      return;
    }

    reorderContextItems(fromIndex, toIndex);
  };

  return (
    <section className="flex h-full min-h-0 flex-col bg-surface/45">
      <div className="border-b border-surface-edge/60 p-3">
        <h2 className="text-lg font-semibold text-zinc-100">Context Management</h2>
      </div>

      <div className="border-b border-surface-edge/60 px-3 py-3">
        <div className="flex items-center gap-3">
          <TokenBudgetRing used={promptTokens} />
          <div>
            <p className="text-sm text-zinc-300">Active Context</p>
            <p className="text-xs text-zinc-500">Total tokens from latest model response metadata.</p>
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 border-b border-surface-edge/60 p-2">
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
          <SortableContext items={contextItems.map((item) => item.id)} strategy={verticalListSortingStrategy}>
            <div className="h-full space-y-2 overflow-y-auto pr-1">
              {contextItems.length === 0 ? (
                <div className="rounded-md border border-dashed border-surface-edge px-3 py-4 text-xs text-zinc-500">
                  Context is empty. Add from report selection or quick-add.
                </div>
              ) : (
                contextItems.map((item, index) => <ContextCard key={item.id} item={item} index={index} />)
              )}
            </div>
          </SortableContext>
        </DndContext>
      </div>

      <div className="border-t border-surface-edge/60 p-3">
        <h4 className="mb-2 text-sm font-semibold text-zinc-100">Quick Add</h4>
        {pasteOpen ? (
          <div className="mb-2 rounded-md border border-surface-edge/70 bg-surface-raised/70 p-2">
            <textarea
              value={pasteText}
              onChange={(event) => setPasteText(event.target.value)}
              rows={4}
              placeholder="Paste text to include in context..."
              className="w-full resize-y rounded border border-surface-edge bg-surface px-2 py-1.5 text-xs text-zinc-100 outline-none placeholder:text-zinc-500 focus:border-accent-cyan/80"
            />
            <div className="mt-2 flex items-center justify-end gap-2 text-xs">
              <button
                type="button"
                onClick={() => {
                  setPasteOpen(false);
                  setPasteText("");
                }}
                className="rounded border border-surface-edge px-2 py-1 text-zinc-300 hover:bg-surface-overlay"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={onPasteText}
                className="rounded border border-accent-cyan/60 bg-accent-cyan/15 px-2 py-1 text-accent-cyan hover:border-accent-cyan"
              >
                Add Text
              </button>
            </div>
          </div>
        ) : null}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,image/png,image/jpeg,.jpg,.jpeg"
            className="hidden"
            onChange={(event) => {
              void onFilesSelected(event);
            }}
          />
          <button
            type="button"
            onClick={onUploadClick}
            className="inline-flex items-center justify-center gap-1 rounded-md border border-accent-blue/45 bg-accent-blue/10 px-2 py-1.5 text-accent-blue hover:border-accent-blue/75"
          >
            <Upload className="h-4 w-4" /> Upload File
          </button>
          <button
            type="button"
            onClick={() => setPasteOpen((prev) => !prev)}
            className="inline-flex items-center justify-center gap-1 rounded-md border border-accent-cyan/45 bg-accent-cyan/10 px-2 py-1.5 text-accent-cyan hover:border-accent-cyan/75"
          >
            <FilePlus2 className="h-4 w-4" /> Paste Text
          </button>
        </div>
        {quickAddMessage ? <p className="mt-2 text-xs text-zinc-400">{quickAddMessage}</p> : null}
      </div>
    </section>
  );
}
