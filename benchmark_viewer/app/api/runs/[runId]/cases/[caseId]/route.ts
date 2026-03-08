import { NextRequest, NextResponse } from "next/server";

import { loadCaseDetail } from "@/src/lib/server/runs";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string; caseId: string }> }
): Promise<NextResponse> {
  void request;
  const { runId, caseId } = await params;
  const detail = await loadCaseDetail(runId, caseId);
  return NextResponse.json(detail);
}
