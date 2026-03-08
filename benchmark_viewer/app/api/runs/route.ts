import { NextResponse } from "next/server";

import { loadRunIndexes } from "@/src/lib/server/runs";

export async function GET(): Promise<NextResponse> {
  const runs = await loadRunIndexes();
  return NextResponse.json({ runs });
}
