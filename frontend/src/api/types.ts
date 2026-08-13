// v2 API types — mirror the FastAPI backend's response shapes exactly.
// Keep this the single source of truth for the shapes the UI consumes.

export type RunStatus = "CREATED" | "RUNNING" | "DONE" | "FAILED";

export type RunListItem = {
  id: string;
  created_at: string;
  status: RunStatus;
  error: string | null;
};

export type JobProgressStep = {
  step: string;
  label: string;
  status: "running" | "done" | "failed";
  ts: string;
};

export type Job = {
  id: string;
  run_id: string;
  kind: string;
  status: "QUEUED" | "RUNNING" | "DONE" | "FAILED";
  payload: { report_upload_id: string; source_upload_ids: string[] };
  progress: JobProgressStep[];
  error: string | null;
  created_at: string;
  updated_at: string;
};

// GET /api/runs/{id}
export type RunDetail = {
  run: RunListItem;
  job: Job | null;
};

export type Verdict = "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE";

export type EvidenceSource = {
  doc_id: string;
  title: string | null;
  page: number | null;
};

export type Claim = {
  claim_id: string;
  text: string;
  page: number | null;
  value: number | null;
  unit: string | null;
  year: number | null;
  verdict: Verdict;
  downgraded: boolean;
  quote: string | null;
  quote_verified: number | null;
  rationale: string;
  year_flag: string | null;
  evidence_source: EvidenceSource | null;
};

export type SourceCredibility = {
  doc_id: string;
  title: string | null;
  total: number; // 0–100
  tier: string;
  components: Record<string, number>;
};

// All 0–1 fractions, or null before the run is scored.
export type Scores = {
  accuracy: number | null;
  coverage: number | null;
  credibility: number | null;
  validity: number | null;
} | null;

export type ReportStats = {
  claims_total: number;
  claims_supported: number;
  claims_contradicted: number;
  claims_unverifiable: number;
};

// GET /api/runs/{id}/report
export type Report = {
  run_id: string;
  status: RunStatus;
  report_doc_id: string | null;
  scores: Scores;
  stats: ReportStats;
  claims: Claim[];
  sources: SourceCredibility[];
};

export type ChatMode = "evidence" | "guidance" | "creative";

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

// POST /api/runs (202)
export type CreateRunResponse = {
  run_id: string;
  job_id: string;
};

// POST /api/runs/{id}/chat
export type ChatResponse = {
  answer: string;
  mode: ChatMode;
};

export const TERMINAL_STATUSES: ReadonlySet<RunStatus> = new Set(["DONE", "FAILED"]);

export const isTerminal = (status: RunStatus | undefined): boolean =>
  status !== undefined && TERMINAL_STATUSES.has(status);
