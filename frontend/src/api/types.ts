// v2 API types — mirror the FastAPI backend's response shapes exactly.
// Keep this the single source of truth for the shapes the UI consumes.

export type RunStatus = "CREATED" | "RUNNING" | "DONE" | "FAILED";

// What a source is: an uploaded PDF, a web page added by link, an image, or a
// video. The type — never which locator happens to be set — decides how
// evidence is cited and which pane shows it.
export type SourceType = "pdf" | "web" | "image" | "youtube";

export type RunListItem = {
  id: string;
  created_at: string;
  status: RunStatus;
  error: string | null;
  title: string | null;
  source_count: number | null; // null for runs without a job row
  scores: Scores;
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

export type RunUpload = {
  id: string;
  kind: "REPORT" | "SOURCE";
  file_name: string; // for a link source, the link itself
  source_type: SourceType;
  url: string | null; // the origin link; null for uploaded files
};

// GET /api/runs/{id}
export type RunDetail = {
  run: RunListItem;
  job: Job | null;
  uploads: RunUpload[]; // report first, then sources; empty for job-less runs
};

export type Verdict = "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE";

export type EvidenceSource = {
  doc_id: string;
  title: string | null;
  page: number | null; // PDFs
  source_type: SourceType;
  url: string | null;
  section: string | null; // the heading the quoted text sits under (web pages)
  start_seconds: number | null; // transcript time (videos)
  chunk_id: number | null;
};

// "disavowed" = the report itself marks the claim false; accuracy scores
// stance-verdict agreement, so disavowed+CONTRADICTED counts as correct.
export type Stance = "asserted" | "disavowed";

export type Claim = {
  claim_id: string;
  text: string;
  page: number | null;
  value: number | null;
  unit: string | null;
  year: number | null;
  verdict: Verdict;
  stance: Stance;
  downgraded: boolean;
  quote: string | null;
  quote_verified: number | null;
  rationale: string;
  year_flag: string | null;
  evidence_source: EvidenceSource | null; // null when no source text was quoted
};

export type SourceBiblio = {
  title?: string | null;
  authors?: string[];
  publisher?: string | null;
  publication_date?: string | null;
  doi?: string | null;
  isbn?: string | null;
};

// Every source document of a run, scored or not. An image is listed but never
// scorable; a scorable source whose total is null has no score in this run.
export type ReportSource = {
  doc_id: string;
  title: string | null;
  source_type: SourceType;
  url: string | null;
  scorable: boolean;
  total: number | null; // 0–100
  tier: string | null;
  components: Record<string, number> | null;
  metadata: SourceBiblio | null;
  // Set only for a web page the page cap read in part: how much of its text the
  // run was scored against, and how much it never saw. Null for a page read
  // whole, and for every PDF and image.
  truncated: { kept_chars: number; dropped_chars: number } | null;
};

// All 0–1 fractions, or null before the run is scored. Credibility stays null
// on a scored run when no source could be scored.
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

// Stored-score detail blocks — null before scoring; individual keys null on
// runs scored by older versions that didn't store them.
export type AccuracyDetail = {
  supported: number | null;
  contradicted: number | null;
  unverifiable: number | null;
  total: number | null;
  correct: number | null;
  incorrect: number | null;
  disavowed: number | null;
};

export type ValidityComponent = {
  score: number | null;
  justification: string;
  quote: string | null;
  quote_verified: number | null;
};

export type ValidityDetail = {
  components: Record<string, ValidityComponent> | null;
  weights_used: Record<string, number> | null;
};

// A source kept out of the credibility average, with how often verdicts cite it.
export type CredibilityExclusion = {
  doc_id: string;
  reason: "image";
  usage: number;
};

export type CredibilityDetail = {
  // "no_scorable_sources" and "no_sources" come with a null credibility score.
  method: string | null;
  sources: { doc_id: string; total: number; tier: string; usage: number }[] | null;
  excluded: CredibilityExclusion[];
};

// GET /api/runs/{id}/report
export type Report = {
  run_id: string;
  title: string | null;
  status: RunStatus;
  report_doc_id: string | null;
  scores: Scores;
  accuracy_detail: AccuracyDetail | null;
  validity_detail: ValidityDetail | null;
  credibility_detail: CredibilityDetail | null;
  stats: ReportStats;
  claims: Claim[];
  sources: ReportSource[];
};

// GET /api/runs/{id}/documents/{doc_id}/file for a web page or transcript: the
// readable text as it was stored, plus where it came from.
export type PageSection = {
  title: string;
  page: number | null;
  text: string; // plain text; tables appear inline as markdown pipe rows
  start_seconds?: number;
  end_seconds?: number;
};

export type PageProvenance = {
  url: string;
  final_url: string;
  fetched_at: string;
  content_type: string;
  title: string | null;
  authors: string[];
  publisher: string | null;
  publication_date: string | null;
  doi: string | null;
  scholarly: boolean;
};

export type PageSnapshot = {
  schema: 1;
  document: { title: string | null; sections: PageSection[] };
  provenance: PageProvenance;
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
