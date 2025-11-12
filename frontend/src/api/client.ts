export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:5001";
const API_KEY = import.meta.env.VITE_API_KEY;

type RequestOptions = RequestInit & {
  skipAuth?: boolean;
};

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers ?? {});
  const isFormData = options.body instanceof FormData;

  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (!options.skipAuth && API_KEY) {
    headers.set("X-API-Key", API_KEY);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function getDashboard() {
  return apiFetch<DashboardResponse>("/api/dashboard");
}

export async function getClaims() {
  return apiFetch<ClaimsResponse>("/api/claims");
}

export async function getLatestReportSummary() {
  return apiFetch<ReportSummaryResponse>("/api/reports/latest");
}

export async function getReportSummary(jobId: string) {
  return apiFetch<ReportSummaryResponse>(`/api/reports/${jobId}/summary`);
}

export async function uploadSources(files: File[]) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  const result = await apiFetch<{ uploads: UploadRecord[] }>("/api/uploads/source", {
    method: "POST",
    body: formData
  });
  return result.uploads;
}

export async function uploadReport(file: File) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  return apiFetch<UploadRecord>("/api/uploads/report", {
    method: "POST",
    body: formData
  });
}

export async function fetchUploads(fileType: "SOURCE" | "REPORT") {
  const result = await apiFetch<{ uploads: UploadRecord[] }>(
    `/api/uploads?type=${fileType.toLowerCase()}`
  );
  return result.uploads;
}

export async function deleteUpload(uploadId: string) {
  return apiFetch<{ status: string }>(`/api/uploads/${uploadId}`, {
    method: "DELETE"
  });
}

export async function sendChat(
  question: string,
  jobId: string,
  sessionId?: string,
  mode?: ChatMode,
  modeLocked?: boolean
) {
  return apiFetch<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: sessionId,
      job_id: jobId,
      mode,
      mode_locked: modeLocked
    })
  });
}

export async function runPipelineWithUploads(sourceIds: string[], reportId: string) {
  return apiFetch<JobResponse>("/api/run_pipeline", {
    method: "POST",
    body: JSON.stringify({ source_ids: sourceIds, report_id: reportId })
  });
}

export async function fetchJob(jobId: string) {
  return apiFetch<JobRecord>(`/api/jobs/${jobId}`);
}

export async function getReportClaims(jobId: string) {
  return apiFetch<ClaimsResponse>(`/api/reports/${jobId}/claims`);
}

export async function getChatHistory(jobId: string) {
  return apiFetch<{ history: ChatHistoryEntry[] }>(`/api/chat/history?job_id=${jobId}`);
}

export async function getSourceDetail(sourceId: string) {
  return apiFetch<SourceDetailResponse>(`/api/sources/${sourceId}`);
}

export type DashboardResponse = {
  report_title: string;
  summary: string;
  scores: {
    overall: number;
    accuracy?: number;
    credibility: number;
    validity: number;
  };
  recommended_sources: string[];
  chat_suggestions: Array<{ id: number; author: string; text: string }>;
};

export type ClaimSummary = {
  claim_id: string;
  text: string;
  verdict: string;
  confidence: number;
  confidence_band: string;
  explanation: string;
};

export type ClaimsResponse = {
  claims: ClaimSummary[];
};

export type ChatResponse = {
  answer: string;
  claims_used: Array<{ claim_id: string; text: string; verdict: string }>;
  sources_used: Array<{ source_id: string; snippet?: string }>;
  mode: ChatMode;
  suggested_mode?: ChatMode;
};

export type ChatMode = "evidence" | "guidance" | "creative";

export type PipelineResponse = {
  claims: ClaimSummary[];
  report_id: string;
  validity: Record<string, unknown>;
  credibility: Record<string, unknown> | null;
};

export type UploadRecord = {
  upload_id: string;
  file_name: string;
  file_type: string;
  file_url: string;
  created_at: string;
};

export type JobResponse = {
  job_id: string;
  status: string;
  result?: PipelineResponse;
};

export type JobRecord = {
  job_id: string;
  status: string;
  report_id?: string;
  source_ids: string[];
  result_json: PipelineResponse;
  error_message?: string;
  created_at?: string;
  updated_at?: string;
};

export type ReportSummaryResponse = {
  job_id: string;
  report: {
    id: string;
    name: string;
    pdf_url: string | null;
    summary: string;
  };
  scores: {
    overall: number;
    accuracy: number;
    credibility: number;
    validity: number;
  };
  recommended_sources: string[];
  chat_messages: Array<{ id: number; author: string; text: string }>;
  sources: Array<{
    id: string;
    name: string;
    file_url: string | null;
    summary: string;
    scores: { credibility: number };
    usage_count: number;
  }>;
  claims: ClaimSummary[];
  stats: {
    claims_total: number;
    claims_supported: number;
    claims_contradicted: number;
    claims_not_found: number;
  };
  top_sources: Array<{
    id: string;
    name: string;
    usage_count: number;
    credibility: number;
  }>;
};

export type ChatHistoryEntry = {
  session_id: string;
  role: string;
  message: string;
  timestamp: string;
  context_ids: Record<string, unknown>;
};

export type SourceDetailResponse = {
  upload: UploadRecord;
  credibility: {
    score: number;
    metadata_confidence?: string;
    components?: Record<string, number>;
  } | null;
  claims: ClaimSummary[];
  usage_count: number;
  tables: any[];
  summary: string;
  validity: {
    score: number;
    supported?: number;
    total?: number;
  } | null;
};
