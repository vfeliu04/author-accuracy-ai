// Typed fetchers for the v2 backend. All requests carry the X-API-Key header;
// errors surface the server's message so TanStack Query can render it.
import type {
  ChatMode,
  ChatResponse,
  ChatTurn,
  CreateRunResponse,
  Report,
  RunDetail,
  RunListItem
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY;

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (API_KEY) {
    headers.set("X-API-Key", API_KEY);
  }
  return headers;
}

async function raise(response: Response): Promise<never> {
  const text = await response.text().catch(() => "");
  throw new Error(text || `Request failed (${response.status})`);
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: authHeaders(init?.headers)
  });
  if (!response.ok) {
    await raise(response);
  }
  return response.json() as Promise<T>;
}

export async function createRun(report: File, sources: File[]): Promise<CreateRunResponse> {
  const form = new FormData();
  form.append("report", report, report.name);
  sources.forEach((file) => form.append("sources", file, file.name));
  // Content-Type is set by the browser for FormData (with the boundary).
  return apiJson<CreateRunResponse>("/api/runs", { method: "POST", body: form });
}

export async function listRuns(): Promise<RunListItem[]> {
  const result = await apiJson<{ runs: RunListItem[] }>("/api/runs");
  return result.runs;
}

export function getRun(runId: string): Promise<RunDetail> {
  return apiJson<RunDetail>(`/api/runs/${runId}`);
}

export function getReport(runId: string): Promise<Report> {
  return apiJson<Report>(`/api/runs/${runId}/report`);
}

export function postChat(
  runId: string,
  body: { question: string; history: ChatTurn[]; mode: ChatMode }
): Promise<ChatResponse> {
  return apiJson<ChatResponse>(`/api/runs/${runId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export function documentFileUrl(runId: string, docId: string): string {
  return `${API_BASE_URL}/api/runs/${runId}/documents/${docId}/file`;
}

// An <iframe> can't send the X-API-Key header, so fetch the PDF as an
// authenticated blob and hand the caller an object URL to render.
export async function fetchPdfBlob(runId: string, docId: string): Promise<Blob> {
  const response = await fetch(documentFileUrl(runId, docId), { headers: authHeaders() });
  if (!response.ok) {
    await raise(response);
  }
  return response.blob();
}
