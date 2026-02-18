"use client";

import { signOut, useSession } from "next-auth/react";

export function UserMenu(): React.ReactElement {
  const { data: session } = useSession();

  if (!session?.user) {
    return (
      <div className="rounded-full border border-surface-edge bg-surface-raised px-3 py-1.5 text-zinc-300">
        User: anonymous
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-2 rounded-full border border-surface-edge bg-surface-raised px-3 py-1.5 text-zinc-300">
        <span>User: {session.user.name ?? session.user.email ?? session.user.id}</span>
        {session.user.role === "admin" ? (
          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
            admin
          </span>
        ) : null}
      </div>

      <button
        type="button"
        onClick={() => void signOut({ callbackUrl: "/login" })}
        className="rounded border border-surface-edge bg-surface-raised px-2.5 py-1 text-xs text-zinc-300 hover:text-zinc-100"
      >
        Sign out
      </button>
    </div>
  );
}
