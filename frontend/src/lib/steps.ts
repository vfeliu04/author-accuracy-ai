// The pipeline's display vocabulary. Step KEYS come from the server
// (job.progress entries); these titles are what users read. Completed steps
// show the server's result string instead (e.g. "Extracted 37 claims"),
// which is already written in product voice.
export const PIPELINE_STEPS = ["ingest", "extract", "verify", "score"] as const;

export const STEP_TITLES: Record<string, string> = {
  ingest: "Read documents",
  extract: "Extract claims",
  verify: "Verify claims against sources",
  score: "Score"
};

// Sub-copy for steps that haven't finished yet.
export const STEP_PENDING_DETAIL: Record<string, string> = {
  ingest: "Text, tables and figures from every PDF",
  extract: "Checkable claims from the report",
  verify: "Checking each claim — typically minutes, longer at busy times",
  score: "Accuracy · credibility · validity"
};
