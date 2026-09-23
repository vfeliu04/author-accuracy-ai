// Typed fetchers for the v2 backend. All requests carry the X-API-Key header;
// errors surface the server's message so TanStack Query can render it.
import type {
  ChatMode,
  ChatResponse,
  ChatTurn,
  CreateRunResponse,
  PageProvenance,
  PageSection,
  PageSnapshot,
  ReferenceScan,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// The server phrases refusals as {"detail": "<sentence>"}; the sentence is what
// a reader needs, not the envelope. Anything else (a validation list, plain
// text) is passed through as sent.
async function raise(response: Response): Promise<never> {
  const text = await response.text().catch(() => "");
  let message = text;
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body) && typeof body.detail === "string" && body.detail) {
      message = body.detail;
    }
  } catch {
    // Not JSON — the text is the message.
  }
  throw new Error(message || `Request failed (${response.status})`);
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

export async function createRun(
  report: File,
  sources: File[],
  links: string[],
  title?: string
): Promise<CreateRunResponse> {
  const form = new FormData();
  form.append("report", report, report.name);
  sources.forEach((file) => form.append("sources", file, file.name));
  // One text part per link source.
  links.forEach((link) => form.append("source_urls", link));
  if (title && title.trim()) {
    form.append("title", title.trim());
  }
  // Content-Type is set by the browser for FormData (with the boundary).
  return apiJson<CreateRunResponse>("/api/runs", { method: "POST", body: form });
}

// Reads the report's reference list without creating anything server-side.
// The signal lets the caller drop a scan that no longer matters (the report
// was swapped, or the dialog closed) instead of waiting on a model call.
export function scanReferences(report: File, signal?: AbortSignal): Promise<ReferenceScan> {
  const form = new FormData();
  form.append("report", report, report.name);
  return apiJson<ReferenceScan>("/api/references/scan", { method: "POST", body: form, signal });
}

export async function listRuns(): Promise<RunListItem[]> {
  const result = await apiJson<{ runs: RunListItem[] }>("/api/runs");
  return result.runs;
}

export function getRun(runId: string): Promise<RunDetail> {
  return apiJson<RunDetail>(`/api/runs/${runId}`);
}

// Requeue a FAILED run's job; the worker resumes from the first incomplete
// step, keeping documents that were already ingested.
export function retryRun(runId: string): Promise<{ run_id: string; job_id: string; status: string }> {
  return apiJson(`/api/runs/${runId}/retry`, { method: "POST" });
}

export async function deleteRun(runId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  if (!response.ok) {
    await raise(response);
  }
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

// A web page (or transcript) is stored as JSON. Returned unparsed: callers run
// it through parsePageSnapshot before rendering anything from it.
export async function fetchDocumentJson(runId: string, docId: string): Promise<unknown> {
  const response = await fetch(documentFileUrl(runId, docId), { headers: authHeaders() });
  if (!response.ok) {
    await raise(response);
  }
  try {
    return (await response.json()) as unknown;
  } catch {
    // Not a stored page at all (a damaged file, or something other than the
    // app answering): asking again can't change that, and the parser's words
    // are not the reader's.
    throw new UnreadablePageError("This page's saved text can't be read.");
  }
}

// A stored page this version can't show. The message is written for the
// reader, and asking again can't change the answer.
export class UnreadablePageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnreadablePageError";
  }
}

const PAGE_FORMAT = 1;

function textOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

// Checks a stored page before any of it is rendered: a file in another format
// is refused loudly rather than misread, and display fields of the wrong kind
// are dropped so they can never reach the screen.
export function parsePageSnapshot(payload: unknown): PageSnapshot {
  if (!isRecord(payload) || payload.schema !== PAGE_FORMAT) {
    throw new UnreadablePageError(
      "This page was saved in a format this version of the app can't display."
    );
  }
  const incomplete = () =>
    new UnreadablePageError("This page's saved text is incomplete, so it can't be displayed.");
  const content = payload.document;
  const origin = payload.provenance;
  if (!isRecord(content) || !Array.isArray(content.sections) || !isRecord(origin)) {
    throw incomplete();
  }
  const sections = content.sections.map((raw): PageSection => {
    if (!isRecord(raw) || typeof raw.text !== "string") {
      throw incomplete();
    }
    const section: PageSection = {
      title: textOrNull(raw.title) ?? "",
      page: typeof raw.page === "number" ? raw.page : null,
      text: raw.text
    };
    if (typeof raw.start_seconds === "number") section.start_seconds = raw.start_seconds;
    if (typeof raw.end_seconds === "number") section.end_seconds = raw.end_seconds;
    return section;
  });
  const provenance: PageProvenance = {
    url: textOrNull(origin.url) ?? "",
    final_url: textOrNull(origin.final_url) ?? "",
    fetched_at: textOrNull(origin.fetched_at) ?? "",
    content_type: textOrNull(origin.content_type) ?? "",
    title: textOrNull(origin.title),
    authors: Array.isArray(origin.authors)
      ? origin.authors.filter((author): author is string => typeof author === "string")
      : [],
    publisher: textOrNull(origin.publisher),
    publication_date: textOrNull(origin.publication_date),
    doi: textOrNull(origin.doi),
    scholarly: origin.scholarly === true
  };
  return {
    schema: PAGE_FORMAT,
    document: { title: textOrNull(content.title), sections },
    provenance
  };
}
