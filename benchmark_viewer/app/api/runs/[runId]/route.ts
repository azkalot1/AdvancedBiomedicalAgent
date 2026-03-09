import { NextRequest, NextResponse } from "next/server";

import { loadRunData } from "@/src/lib/server/runs";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
): Promise<NextResponse> {
  void request;
  const { runId } = await params;
  const run = await loadRunData(runId);
  return NextResponse.json(run);
}
