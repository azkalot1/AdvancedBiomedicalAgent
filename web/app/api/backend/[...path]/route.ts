import type { NextRequest } from "next/server";
import { getServerSession } from "next-auth";

import { authOptions } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_BACKEND_URL = "http://localhost:8000";

function backendBaseUrl(): string {
  return (process.env.BIOAGENT_BACKEND_URL || process.env.NEXT_PUBLIC_BIOAGENT_BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/$/, "");
}

function applyServerAuth(headers: Headers, userId: string | null): Headers {
  const outgoing = new Headers(headers);
  outgoing.delete("host");
  outgoing.delete("content-length");
  outgoing.delete("connection");
  outgoing.delete("accept-encoding");
  outgoing.delete("authorization");
  outgoing.delete("cookie");
  outgoing.delete("x-bioagent-user-id");

  const envToken = process.env.BIOAGENT_API_TOKEN?.trim();
  if (envToken) {
    outgoing.set("authorization", `Bearer ${envToken}`);
  }
  if (userId) {
    outgoing.set("x-bioagent-user-id", userId);
  }

  return outgoing;
}

async function forwardRequest(request: NextRequest, pathSegments: string[]): Promise<Response> {
  const path = pathSegments.join("/");
  const search = request.nextUrl.search;
  const url = `${backendBaseUrl()}/${path}${search}`;

  const session = await getServerSession(authOptions);
  const headers = applyServerAuth(request.headers, session?.user?.id ?? null);
  const method = request.method.toUpperCase();

  let body: BodyInit | undefined;
  if (method !== "GET" && method !== "HEAD") {
    const buffer = await request.arrayBuffer();
    body = buffer.byteLength > 0 ? buffer : undefined;
  }

  const upstream = await fetch(url, {
    method,
    headers,
    body,
    cache: "no-store",
    redirect: "manual"
  });

  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete("content-length");

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders
  });
}

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return forwardRequest(request, path);
}

export async function POST(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return forwardRequest(request, path);
}

export async function PUT(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return forwardRequest(request, path);
}

export async function PATCH(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return forwardRequest(request, path);
}

export async function DELETE(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return forwardRequest(request, path);
}
